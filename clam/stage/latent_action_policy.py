import re
from pathlib import Path
from typing import Callable, Dict, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Key
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from prompt_dtla.models.base import Base
from prompt_dtla.models.policy import Policy
from prompt_dtla.stage.action_decoder import LatentActionDecoder
from prompt_dtla.stage.lam.lam import LatentActionModel
from prompt_dtla.utils.data_utils import Batch, PRNGKeyDict, get_latent_action_dim
from prompt_dtla.utils.logger import log
from prompt_dtla.utils.training import (
    TrainState,
    default_weight_init,
    get_AdamW_optimizer,
)


class LatentActionAgent(Base):
    """
    BC Agent that maps observations to latent actions using a pretrained IDM
    """

    def _load_lam_and_decoder(self, key: Key) -> TrainState:
        kwargs = dict(
            load_from_ckpt=True,
            key=key,
            observation_shape=self.observation_shape,
            continuous_actions=self.continuous_actions,
            input_action_dim=self.input_action_dim,
            init_kwargs=default_weight_init,
        )

        key, lam_key, decoder_key = jax.random.split(key, 3)

        # loading both the LAM and latent action decoder
        if (
            hasattr(self.cfg, "lam_and_decoder_ckpt")
            and self.cfg.lam_and_decoder_ckpt is not None
        ):
            ckpt_path = self.cfg.lam_and_decoder_ckpt

            log("Loading LAM and decoder", color="blue")

            config_file = Path(ckpt_path) / "config.yaml"
            lam_and_decoder_cfg = OmegaConf.load(config_file)
            latent_action_dim = get_latent_action_dim(lam_and_decoder_cfg.stage.idm)
            kwargs["action_dim"] = latent_action_dim

            log("loading LAM from ckpt")
            lam_stage = LatentActionModel(
                cfg=lam_and_decoder_cfg.stage, ckpt_dir=Path(ckpt_path), **kwargs
            )

            ts_lam = lam_stage.create_train_state(lam_key)

            log("loading action decoder from ckpt")
            # update action dim
            kwargs["action_dim"] = self.action_dim

            decoder_stage = LatentActionDecoder(
                cfg=lam_and_decoder_cfg.stage, ckpt_dir=Path(ckpt_path), **kwargs
            )

            ts_decoder = decoder_stage.create_train_state(decoder_key)
        elif hasattr(self.cfg, "lam_ckpt") and self.cfg.lam_ckpt is not None:
            import ipdb

            ipdb.set_trace()
            # load the LAM and decoder separately
            assert hasattr(self.cfg, "la_decoder_ckpt")
            lam_ckpt_path = self.cfg.lam_ckpt
            decoder_ckpt_path = self.cfg.la_decoder_ckpt

        elif hasattr(self.cfg, "decoder_ckpt") and self.cfg.decoder_ckpt is not None:
            # load pretrained latent action decoder
            # this assumes we relabelled the full dataset with latent actions already
            ckpt_path = self.cfg.decoder_ckpt
            config_file = Path(ckpt_path) / "config.yaml"
            decoder_cfg = OmegaConf.load(config_file)

            kwargs["action_dim"] = self.action_dim
            latent_action_dim = (
                8  # TODO: fix this, need to get this from ckpt somewhere
            )

            decoder_stage = LatentActionDecoder(
                cfg=decoder_cfg.stage, ckpt_dir=Path(ckpt_path), **kwargs
            )

            ts_decoder = decoder_stage.create_train_state(key)
            ts_lam = None
        else:
            raise ValueError("No ckpt path provided")

        return ts_lam, ts_decoder, latent_action_dim

    def _init_model(self, keys: PRNGKeyDict) -> Tuple[dict, nn.Module]:
        """
        Load latent action decoder from ckpt, then initialize BC policy
        """
        log(f"action dim: {self.action_dim}")

        # * policy
        bs = 2
        dummy_state = jnp.zeros((bs, *self.observation_shape), dtype=jnp.float32)

        # * decoder
        ts_lam, ts_decoder, latent_action_dim = self._load_lam_and_decoder(
            keys["decoder_params"]
        )

        if ts_lam is None:
            lam_apply = None
        else:
            lam_apply = jax.jit(
                ts_lam.apply_fn, static_argnames=["self", "is_training"]
            )

        object.__setattr__(self, "lam_apply", lam_apply)
        object.__setattr__(self, "ts_lam", ts_lam)

        decoder_apply = jax.jit(
            ts_decoder.apply_fn, static_argnames=["self", "is_training"]
        )

        model_def = LatentActionPolicy(
            ts_decoder=ts_decoder,
            decoder_apply=decoder_apply,
            cfg=self.cfg.policy,
            latent_action_dim=latent_action_dim,
        )

        params = model_def.init(keys, x=dummy_state, is_training=True)
        return params, model_def

    def create_train_state(self, key: Key) -> TrainState:
        params_key, decoder_params_key, sample_key = jax.random.split(key, 3)
        keys = {
            "params": params_key,
            "decoder_params": decoder_params_key,
            "sample": sample_key,
        }
        variables, model_def = self._init_model(keys)
        tx = get_AdamW_optimizer(self.cfg)

        mparams = {}
        if "batch_stats" in variables:
            mparams["batch_stats"] = variables["batch_stats"]

        ts = TrainState.create(
            apply_fn=model_def.apply,
            params=variables["params"],
            tx=tx,
            mparams=mparams,
            keys={"sample": sample_key},
        )

        return ts

    def update(
        self,
        key: Key,
        ts: TrainState,
        batch: Batch,
        is_training: bool,
        **kwargs: Dict,
    ) -> Tuple[TrainState, Dict, Dict]:
        key, vq_key, sample_key, dropout_key = jax.random.split(key, 4)

        def loss_fn(params, batch: Batch, is_training: bool = True):
            # first check if the latent action field is present
            if hasattr(batch, "latent_actions") and batch.latent_actions is not None:
                idm_latent_actions = batch.latent_actions  # BDa
                gt_actions = batch.actions  # B,
                observations = batch.observations  # BHWC

            else:  # otherwise we use the LAM to first infer the latent actions
                assert self.lam_apply is not None, "LAM is not loaded"
                rngs = {
                    "vq": vq_key,
                    "sample": sample_key,
                    "dropout": dropout_key,
                }
                lam_output = self.lam_apply(
                    {"params": self.ts_lam.params, **self.ts_lam.mparams},
                    x=batch.observations,
                    is_training=False,
                    rngs=rngs,
                )
                idm_latent_actions = lam_output.idm.latent_actions
                gt_actions = batch.actions[
                    :, -2
                ]  # we are trying to predict the action between o_t and o_t+1
                observations = batch.observations[
                    :, -2
                ]  # we use the second to last observation

            policy_output = ts.apply_fn(
                {"params": params, **ts.mparams},
                x=observations,
                is_training=is_training,
                mutable=list(ts.mparams.keys()) if is_training else False,
                rngs={"sample": sample_key},
            )

            if is_training:
                policy_output, mparams = policy_output
            else:
                policy_output, mparams = (policy_output, None)

            loss = optax.squared_error(policy_output.latent_action, idm_latent_actions)
            loss = jnp.mean(loss)

            metrics = {"bc_loss": loss}

            if self.continuous_actions:
                decoded_acc = policy_output.action == gt_actions.squeeze()
                decoded_acc = jnp.mean(decoded_acc)
                metrics["decoded_acc"] = decoded_acc

            extras = {}
            return loss, (metrics, extras, mparams)

        value_grad_fn = jax.value_and_grad(
            lambda params: loss_fn(params, batch, is_training=is_training),
            has_aux=True,
        )
        (loss, (metrics, extra, mparams)), grads = value_grad_fn(ts.params)

        ts = ts.apply_gradients(grads=grads)
        ts = ts.replace(mparams=mparams)
        return ts, metrics, extra


class LatentActionPolicy(nn.Module):
    ts_decoder: TrainState
    decoder_apply: Callable
    cfg: DictConfig
    latent_action_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, is_training: bool, rng=None):
        """
        Predicts the latent action and decodes it to the original action space.
        Potentially two separate action sampling steps: latent then ground truth.
        """

        # predict latent action
        latent_action_out = Policy(
            self.cfg,
            is_continuous=True,
            action_dim=self.latent_action_dim,
            init_kwargs=default_weight_init,
        )(x=x, is_training=is_training)
        latent_action_preds = latent_action_out.action

        # decode latent action back to original action space using pretrained action decoder
        policy_output = self.decoder_apply(
            {"params": self.ts_decoder.params},
            x=latent_action_preds,
            is_training=False,
            rng=rng,
        )
        policy_output = policy_output.replace(latent_action=latent_action_preds)

        return policy_output

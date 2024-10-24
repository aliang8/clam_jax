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

from prompt_dtla.baselines.vpt.vpt import VPT
from prompt_dtla.models.base import Base
from prompt_dtla.models.policy import Policy
from prompt_dtla.stage.action_decoder import LatentActionDecoder
from prompt_dtla.stage.lam.lam import LatentActionModel
from prompt_dtla.utils.data_utils import Batch, PRNGKeyDict
from prompt_dtla.utils.logger import log
from prompt_dtla.utils.training import (
    TrainState,
    default_weight_init,
    get_AdamW_optimizer,
)


class VPTAgent(Base):
    """
    BC Agent that maps observations to actions using a pretrained IDM
    """

    def _load_vpt_idm(self, key: Key) -> TrainState:
        kwargs = dict(
            load_from_ckpt=True,
            key=key,
            action_dim=self.action_dim,
            observation_shape=self.observation_shape,
            continuous_actions=self.continuous_actions,
            input_action_dim=self.input_action_dim,
            init_kwargs=default_weight_init,
        )

        key, vpt_key = jax.random.split(key, 2)

        # load pretrained latent action decoder
        # this assumes we relabelled the full dataset with latent actions already
        ckpt_path = self.cfg.vpt_ckpt
        if "vpt_ne" in self.cfg and self.cfg.vpt_ne != -1:
            ckpt_path = re.sub(r"ne-\d+", f"ne-{self.cfg.vpt_ne}", ckpt_path)

        config_file = Path(ckpt_path) / "config.yaml"
        vpt_cfg = OmegaConf.load(config_file)

        vpt_stage = VPT(config=vpt_cfg.stage, ckpt_dir=Path(ckpt_path), **kwargs)

        ts_vpt = vpt_stage.create_train_state(key)
        return ts_vpt

    def _init_model(self, keys: PRNGKeyDict) -> Tuple[dict, nn.Module]:
        """
        Initialize BC policy
        """
        log(f"action dim: {self.action_dim}")

        # * policy
        bs = 2
        dummy_state = jnp.zeros((bs, *self.observation_shape), dtype=jnp.float32)

        model_def = Policy(
            self.cfg.policy,
            is_continuous=self.continuous_actions,
            action_dim=self.action_dim,
            init_kwargs=default_weight_init,
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

        # load pretrained vpt
        ts_vpt = self._load_vpt_idm(decoder_params_key)
        vpt_apply = jax.jit(ts_vpt.apply_fn, static_argnames=["self", "is_training"])
        object.__setattr__(self, "ts_vpt", ts_vpt)
        object.__setattr__(self, "vpt_apply", vpt_apply)

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

            else:  # otherwise we use the IDM to first infer the latent actions
                rngs = {
                    "vq": vq_key,
                    "sample": sample_key,
                    "dropout": dropout_key,
                }
                vpt_output = self.vpt_apply(
                    {"params": self.ts_vpt.params},
                    x=batch.observations,
                    is_training=False,
                    rngs=rngs,
                )
                vpt_actions = vpt_output.action
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

            metrics = {}

            if self.continuous_actions:
                loss = optax.squared_error(policy_output.action, vpt_actions)
                loss = jnp.mean(loss)
            else:
                raise NotImplementedError("Only continuous actions are supported")

                decoded_acc = policy_output.action == gt_actions.squeeze()
                decoded_acc = jnp.mean(decoded_acc)
                metrics["acc"] = decoded_acc

            metrics["bc_loss"] = loss

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

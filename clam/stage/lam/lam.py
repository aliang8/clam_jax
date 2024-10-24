from typing import Dict, Tuple

import distrax
import einops
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from einops import rearrange
from jaxtyping import Key
from loguru import logger
from omegaconf import DictConfig

from prompt_dtla.models.base import Base
from prompt_dtla.stage.lam.lam_cnn import ConvLAM
from prompt_dtla.stage.lam.lam_hierarchical import HierarchicalStateBasedLAM
from prompt_dtla.stage.lam.lam_rnn import RNNStateBasedLAM
from prompt_dtla.stage.lam.lam_state import StateBasedLAM
from prompt_dtla.stage.lam.lam_vit import ViTLAM
from prompt_dtla.utils.data_utils import Batch, PRNGKeyDict
from prompt_dtla.utils.logger import log
from prompt_dtla.utils.training import (
    TrainState,
    default_weight_init,
    get_AdamW_optimizer,
)


class LatentActionModel(Base):
    weight_init_kwargs = default_weight_init

    def _init_model(self, keys: PRNGKeyDict) -> Tuple[dict, nn.Module]:
        T = self.cfg.seq_len
        B = 2

        dummy_x = jnp.zeros((B, T, *self.observation_shape), dtype=jnp.float32)
        timesteps = jnp.arange(T, dtype=jnp.int32).reshape(1, -1).repeat(B, axis=0)

        if "mlp" in self.cfg.idm.encoder_cfg.name:
            state_dim = self.observation_shape[-1]
            if self.cfg.hierarchical_vq:
                model_def = HierarchicalStateBasedLAM(
                    self.cfg, state_dim, self.weight_init_kwargs
                )
            elif self.cfg.multistep_prediction_rnn:
                model_def = RNNStateBasedLAM(
                    self.cfg, state_dim, self.weight_init_kwargs
                )
            else:
                model_def = StateBasedLAM(self.cfg, state_dim, self.weight_init_kwargs)
        elif "vit" in self.cfg.idm.encoder_cfg.name:
            model_def = ViTLAM(self.cfg, self.weight_init_kwargs)
        elif "cnn_encoder" in self.cfg.idm.encoder_cfg.name:
            model_def = ConvLAM(self.cfg, self.weight_init_kwargs)
        else:
            raise ValueError(f"Invalid encoder name: {self.cfg.idm.encoder_cfg.name}")

        if self.load_from_ckpt:
            ts = self.load_model_from_ckpt()

            if "lam" in ts["params"]:
                params = ts["params"]["lam"]
            else:
                params = ts["params"]

            if "vq" in ts["params"] and "lam" in ts["mparams"]["vq"]:
                # TODO: maybe fix this
                mparams = {"vq": ts["mparams"]["vq"]["lam"]}
            else:
                mparams = ts["mparams"]

            # TODO: maybe fix this
            params = {"params": params, **mparams}
        else:
            params = model_def.init(
                keys,
                x=dummy_x,
                is_training=True,
            )

        return params, model_def

    def create_train_state(self, key: Key) -> TrainState:
        param_key, vq_key, sample_key, dropout_key = jax.random.split(key, 4)
        keys = {
            "params": param_key,
            "vq": vq_key,
            "sample": sample_key,
            "dropout": dropout_key,
        }

        variables, model_def = self._init_model(keys)
        tx = get_AdamW_optimizer(self.cfg)

        mparams = {}
        if "vq" in variables:
            mparams["vq"] = variables["vq"]
        if "batch_stats" in variables:
            mparams["batch_stats"] = variables["batch_stats"]

        ts = TrainState.create(
            apply_fn=model_def.apply,
            params=variables["params"],
            tx=tx,
            mparams=mparams,
            keys=keys,
        )
        return ts

    def update(
        self,
        key: Key,
        ts: TrainState,
        batch: Batch,
        is_training: bool,
        labelled: bool = False,
    ) -> Tuple[TrainState, Dict, Dict]:
        key, vq_key, sample_key = jax.random.split(key, 3)

        def loss_fn(params, batch: Batch, is_training: bool = True):
            apply_out = ts.apply_fn(
                {"params": params, **ts.mparams},
                x=batch.observations,
                actions=batch.actions,
                timesteps=batch.timestep,
                is_training=is_training,
                mutable=list(ts.mparams.keys()) if is_training else False,
                rngs={"vq": vq_key, "sample": sample_key},
            )

            if is_training:
                lam_output, mparams = apply_out
            else:
                lam_output, mparams = apply_out, None

            next_obs_pred = lam_output.next_obs_pred

            loss = 0.0

            ## KL LOSS
            if self.cfg.idm.distributional_la:
                # compute KL divergence loss to prior
                la_dist = lam_output.idm.latent_action_dist

                if self.cfg.idm.latent_action_prior == "uniform":
                    normal_dist = distrax.Normal(
                        jnp.zeros_like(la_dist.mean()), jnp.ones_like(la_dist.stddev())
                    )

                    kl_div = la_dist.kl_divergence(normal_dist)
                    kl_div = jnp.mean(kl_div)

                    kl_loss = self.cfg.idm.kl_weight * kl_div
                    loss += kl_loss
                else:
                    raise NotImplementedError("Only uniform prior is supported")
            else:
                kl_div = 0.0

            use_vq = (
                self.cfg.idm.apply_quantization or self.cfg.idm.separate_categorical_la
            )

            ## RECONSTRUCTION LOSS
            if self.cfg.idm.image_obs:
                # make sure same format as the ground truth, HWC
                if self.cfg.fdm.decoder_cfg.name in ["transformer", "vit"]:
                    # this should have T-1 predictions where T is our sequence length
                    next_obs_pred = einops.rearrange(
                        next_obs_pred, "b t c h w -> b t h w c"
                    )
                    gt_next_obs = batch.observations[:, 1:]
                    mask = batch.mask[:, 1:]

                    recon_loss = optax.squared_error(next_obs_pred, gt_next_obs)
                    recon_loss = jnp.mean(recon_loss, axis=(2, 3, 4))
                    recon_loss *= mask
                    recon_loss = jnp.sum(recon_loss) / jnp.sum(mask)
                else:
                    next_obs_gt = batch.observations[:, -1]
                    recon_loss = optax.squared_error(next_obs_pred, next_obs_gt).mean()
            else:
                next_obs_gt = batch.observations[:, -1]
                recon_loss = optax.squared_error(next_obs_pred, next_obs_gt).mean()

            recon_loss = self.cfg.recon_loss_weight * recon_loss
            loss += recon_loss

            if use_vq:
                vq_loss = self.cfg.commitment_loss_weight * lam_output.idm.vq.loss
                loss += vq_loss
            else:
                vq_loss = 0.0

            metrics = {
                "recon_loss": recon_loss,
                "vq_loss": vq_loss,
                "kl_loss": kl_div,
                "total_loss": loss,
            }

            if use_vq:
                metrics.update(
                    {
                        "vq_loss": vq_loss,
                        "perplexity": lam_output.idm.vq.perplexity,
                        "num_expired_codes": lam_output.idm.vq.num_expired_codes
                        if lam_output.idm.vq.num_expired_codes is not None
                        else 0,
                    }
                )

            latent_actions = lam_output.idm.latent_actions

            # compute sanity check loss value
            # what happens if the FDM just predicts the previous observation
            cheating_loss = optax.squared_error(
                batch.observations[:, -2], batch.observations[:, -1]
            )

            cheating_loss = jnp.mean(cheating_loss)

            extra = {
                "next_obs_pred": next_obs_pred,
                "latent_actions": latent_actions,
                "action_output": lam_output.action_output,
            }
            return loss, (metrics, extra, mparams)

        value_grad_fn = jax.value_and_grad(
            lambda params: loss_fn(params, batch, is_training=is_training),
            has_aux=True,
        )
        (loss, (metrics, extra, mparams)), grads = value_grad_fn(ts.params)

        ts = ts.apply_gradients(grads=grads)
        ts = ts.replace(mparams=mparams)

        extra["grads"] = grads
        return ts, metrics, extra

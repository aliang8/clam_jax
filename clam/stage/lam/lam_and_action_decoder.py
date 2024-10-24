from typing import Any, Dict, Tuple

import distrax
import einops
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxtyping import Key
from omegaconf import DictConfig

from prompt_dtla.models.base import Base
from prompt_dtla.stage.action_decoder import ActionDecoderNetwork
from prompt_dtla.stage.lam.lam_cnn import ConvLAM
from prompt_dtla.stage.lam.lam_rnn import RNNStateBasedLAM
from prompt_dtla.stage.lam.lam_state import StateBasedLAM

# from prompt_dtla.stage.lam.lam_vit import ViTLAM
from prompt_dtla.stage.lam.utils import LAMOutput
from prompt_dtla.utils.data_utils import Batch, PRNGKeyDict
from prompt_dtla.utils.logger import log
from prompt_dtla.utils.training import (
    TrainState,
    default_weight_init,
    get_AdamW_optimizer,
)


class LAMAndDecoder(Base):
    gaussian_policy: bool = False
    weight_init_kwargs = default_weight_init

    def _init_model(self, keys: PRNGKeyDict) -> Tuple[dict, nn.Module]:
        t, bs = 2 + self.cfg.context_len, 2
        dummy_x = np.zeros((bs, t, *self.observation_shape), dtype=np.float32)
        dummy_actions = np.zeros((bs, t, self.input_action_dim), dtype=np.int32)
        timesteps = np.arange(t, dtype=np.int32).reshape(1, -1).repeat(bs, axis=0)

        state_dim = self.observation_shape[-1]

        model_def = LAMAndDecoderNetwork(
            cfg=self.cfg,
            action_dim=self.input_action_dim,
            state_dim=state_dim,
            init_kwargs=self.weight_init_kwargs,
            is_continuous=self.continuous_actions,
            gaussian_policy=self.gaussian_policy,
            observation_shape=self.observation_shape,
        )

        params = model_def.init(
            keys, dummy_x, dummy_actions, timesteps=timesteps, is_training=True
        )

        return params, model_def

    def create_train_state(self, key: Key) -> TrainState:
        params_key, vq_key, sample_key = jax.random.split(key, 3)
        keys = {"params": params_key, "vq": vq_key, "sample": sample_key}

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
                    normal_dist = distrax.MultivariateNormalDiag(
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

            ## LATENT ACTION DECODER REGULARIZATION LOSS
            if labelled:
                action_logits = lam_output.action_output.logits
                action_pred = lam_output.action_output.action

                if self.cfg.idm.encoder_cfg.name in ["transformer", "vit"]:
                    gt_actions = batch.actions[:, :-1]
                    gt_actions = einops.rearrange(gt_actions, "b t -> (b t)")
                    action_logits = einops.rearrange(action_logits, "b t a -> (b t) a")
                    action_pred = einops.rearrange(action_pred, "b t -> (b t)")
                else:
                    # we want to predict action going from o_t to o_t+1
                    # so we need to predict the action at t-1
                    gt_actions = batch.actions[:, -2]

                if batch.actions.shape[-1] == 1:
                    gt_actions = gt_actions.squeeze(axis=-1)

                if self.continuous_actions:
                    if self.gaussian_policy:
                        dist = lam_output.action_output.dist
                        action_loss = -dist.log_prob(gt_actions)
                    else:
                        action_loss = optax.squared_error(action_pred, gt_actions)
                else:
                    # discrete actions
                    action_loss = optax.softmax_cross_entropy_with_integer_labels(
                        action_logits, gt_actions.astype(jnp.int32)
                    )

                latent_action_loss = jnp.mean(action_loss)
                latent_action_loss = (
                    self.cfg.action_pred_loss_weight * latent_action_loss
                )

                if not self.continuous_actions:
                    import ipdb

                    ipdb.set_trace()
                    # for discrete actions
                    acc = action_pred == gt_actions
            else:
                latent_action_loss = 0.0

            loss += latent_action_loss

            metrics = {
                "recon_loss": recon_loss,
                "vq_loss": vq_loss,
                "kl_loss": kl_div,
                "latent_action_loss": latent_action_loss,
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


class LAMAndDecoderNetwork(nn.Module):
    cfg: DictConfig
    action_dim: int
    state_dim: int
    init_kwargs: Dict

    is_continuous: bool
    gaussian_policy: bool
    observation_shape: Tuple[int, ...]

    def setup(self):
        if "mlp" in self.cfg.idm.encoder_cfg.name:
            state_dim = self.observation_shape[-1]
            if self.cfg.hierarchical_vq:
                self.lam = HierarchicalStateBasedLAM(
                    self.cfg, state_dim, self.init_kwargs
                )
            elif self.cfg.multistep_prediction_rnn:
                self.lam = RNNStateBasedLAM(self.cfg, state_dim, self.init_kwargs)
            else:
                self.lam = StateBasedLAM(self.cfg, state_dim, self.init_kwargs)
        elif "vit" in self.cfg.idm.encoder_cfg.name:
            self.lam = ViTLAM(self.cfg, self.init_kwargs)
        elif "cnn_encoder" in self.cfg.idm.encoder_cfg.name:
            self.lam = ConvLAM(self.cfg, self.init_kwargs)
        else:
            raise ValueError(f"Invalid encoder name: {self.cfg.idm.encoder_cfg.name}")

        self.decoder = ActionDecoderNetwork(
            cfg=self.cfg,
            action_dim=self.action_dim,
            init_kwargs=self.init_kwargs,
            is_continuous=self.is_continuous,
            gaussian_policy=self.gaussian_policy,
        )

    def __call__(
        self,
        x: jnp.ndarray,
        actions: jnp.ndarray = None,
        timesteps: jnp.ndarray = None,
        is_training: bool = True,
    ) -> Tuple[LAMOutput, Any]:
        lam_output = self.lam(x, is_training=is_training)

        if not self.cfg.idm.apply_quantization:
            latent_actions = lam_output.idm.latent_actions
        else:
            latent_actions = lam_output.idm.vq.quantize

        action_output = (
            None
            if actions is None
            else self.decoder(latent_actions, is_training=is_training)
        )

        return LAMOutput(
            next_obs_pred=lam_output.next_obs_pred,
            action_output=action_output,
            idm=lam_output.idm,
        )

import copy
from typing import Dict

import distrax
import flax.linen as nn
import jax.numpy as jnp
from einops import rearrange, repeat
from jaxtyping import Array, Float
from loguru import logger
from omegaconf import DictConfig

from prompt_dtla.models.mlp import MLP, GaussianMLP
from prompt_dtla.models.vq import NAME_TO_VQ_CLS
from prompt_dtla.stage.lam.utils import IDMOutput, LAMOutput
from prompt_dtla.utils.data_utils import BTD, get_latent_action_dim
from prompt_dtla.utils.logger import log


class StateBasedLAM(nn.Module):
    cfg: DictConfig
    state_dim: int
    init_kwargs: Dict

    def setup(self):
        self.idm = StateBasedIDM(self.cfg.idm, self.init_kwargs)
        self.fdm = StateBasedFDM(self.cfg.fdm, self.state_dim, self.init_kwargs)

    def __call__(
        self,
        x: BTD,
        is_training: bool = True,
        latent_actions: jnp.ndarray = None,  # for debugging
        **kwargs,
    ) -> LAMOutput:
        """
        Input:
            x: (B, T, D)

        Output:
            next_state_pred: (B, T, D)
        """
        # IDM predicts the latent action (z_t) given o_t-k, ..., o_t and o_t+1
        idm_output = self.idm(x, is_training=is_training)

        if latent_actions is None:
            if self.cfg.idm.apply_quantization:
                latent_actions = idm_output.vq.quantize
            else:
                latent_actions = idm_output.latent_actions

        # FDM predicts o_t+1 given o_t-k, o_t and z_t
        context = x[:, :-1]

        # log(f"context: {context.shape}, latent_actions: {latent_actions.shape}")

        next_state_pred = self.fdm(
            context,
            latent_actions,
            is_training=is_training,
        )

        return LAMOutput(
            next_obs_pred=next_state_pred,
            action_output=None,
            idm=idm_output,
        )


class StateBasedIDM(nn.Module):
    """Inverse Dynamics Model for Latent Action Model with VectorQuantization

    IDM - p(z_t | o_t-k,..., o_t, o_t+1)

    Args:
        pass
    """

    cfg: DictConfig
    init_kwargs: Dict

    def setup(self) -> None:
        self.state_encoder = MLP(
            hidden_dims=self.cfg.encoder_cfg.hidden_dims,
            init_kwargs=self.init_kwargs,
            activate_final=False,
        )

        la_mlp_dim = copy.deepcopy(self.cfg.la_mlp_dims)
        latent_action_dim = get_latent_action_dim(self.cfg)
        la_mlp_dim[-1] = latent_action_dim

        log(f"latent action dim: {latent_action_dim}, la_mlp_dim: {la_mlp_dim}")

        if self.cfg.distributional_la:
            self.latent_action_head = GaussianMLP(
                hidden_dims=la_mlp_dim,
                init_kwargs=self.init_kwargs,
                activate_final=False,
            )
        else:
            self.latent_action_head = MLP(
                hidden_dims=la_mlp_dim,
                init_kwargs=self.init_kwargs,
                activate_final=False,
            )

        self.vq = NAME_TO_VQ_CLS[self.cfg.vq.name](
            self.cfg.vq, init_kwargs=self.init_kwargs
        )

    def __call__(self, x: BTD, is_training: bool = True) -> IDMOutput:
        """
        IDM takes the state and next state and predicts the action
        IDM predicts the latent action (z_t) given o_t-k,..., o_t and o_t+1

        Input:
            states: (B, T, D)

        Output:
            vq_outputs: Dict
        """

        if self.cfg.use_state_diff:
            # take the difference between the current and next state
            x = x[:, 1:] - x[:, :-1]

        # [B, T, state_dim]
        # flatten the last two dimensions
        x = rearrange(x, "b t d -> b (t d)")
        x_embeds = self.state_encoder(x)

        # predict latent actions
        latent_actions = self.latent_action_head(nn.gelu(x_embeds))

        if self.cfg.distributional_la:
            # latent action is parameterized as a Gaussian distribution
            mu, log_std = latent_actions

            # clamp the log_std, TODO check this
            log_std = jnp.clip(log_std, -20, 2)

            dist = distrax.MultivariateNormalDiag(loc=mu, scale_diag=jnp.exp(log_std))
            latent_actions = dist.sample(seed=self.make_rng("sample"))
        else:
            dist = None

        vq_outputs = None

        # maybe split into a categorical and continuous latent action
        if (
            hasattr(self.cfg, "separate_categorical_la")
            and self.cfg.separate_categorical_la
        ):
            categorical_la, continuous_la = jnp.split(
                latent_actions, indices_or_sections=[self.cfg.vq.code_dim], axis=-1
            )

            # we will quantize the categorical part and treat the
            # continuous part as as the offset*
            vq_outputs = self.vq(categorical_la, is_training=is_training)

            # latent action = [quantize, continuous]
            latent_actions = jnp.concatenate(
                [vq_outputs.quantize, continuous_la], axis=-1
            )

        # compute quantized latent actions
        if self.cfg.apply_quantization:
            vq_outputs = self.vq(latent_actions, is_training=is_training)

        idm_output = IDMOutput(
            vq=vq_outputs, latent_actions=latent_actions, latent_action_dist=dist
        )
        return idm_output


class StateBasedFDM(nn.Module):
    """
    Forward Dynamics Model - p(o_t+1 | o_t-k,..., o_t, z_t)
    FDM predicts the next observation o_t+1

    Args:
        pass
    """

    cfg: DictConfig
    state_dim: int
    init_kwargs: Dict

    def setup(self) -> None:
        self.encoder = MLP(
            hidden_dims=self.cfg.encoder_cfg.hidden_dims,
            init_kwargs=self.init_kwargs,
            activate_final=False,
        )

        self.decoder = MLP(
            hidden_dims=self.cfg.decoder_cfg.hidden_dims + [self.state_dim],
            init_kwargs=self.init_kwargs,
            activate_final=False,
        )

    def __call__(
        self,
        context: BTD,
        actions: Float[Array, "B D"],
        is_training: bool = True,
    ) -> BTD:
        """
        FDM takes the prev states and latent action and predicts the next state
        T is the length of the context provided.

        For LAPO, T=2, just o_t-1 and o_t
        If we use a Transformer, T is the full sequence or a larger context window

        Input:
            context: (B, T, D)
            actions: (B, D_L)

        Output:
            next_state_pred: (B, T, D)
        """
        # repeat the actions across the spatial dimensions
        # we will inject the actions into the middle of the U-net
        action_expand = repeat(actions, "b dl -> b t dl", t=context.shape[1])

        # (B, T, D + D_L)
        x = jnp.concatenate([context, action_expand], axis=-1)

        x = rearrange(x, "b t d -> b (t d)")

        embeddings = self.encoder(x, is_training=is_training)
        next_state_pred = self.decoder(embeddings, is_training=is_training)

        return next_state_pred

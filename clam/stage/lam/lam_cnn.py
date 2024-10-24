from typing import Dict

import einops
import flax.linen as nn
import jax.numpy as jnp
from jaxtyping import Array, Float
from loguru import logger
from omegaconf import DictConfig

from prompt_dtla.models.cnn import ConvDecoder, ConvEncoder
from prompt_dtla.models.mlp import MLP
from prompt_dtla.models.vq import NAME_TO_VQ_CLS
from prompt_dtla.stage.lam.utils import IDMOutput, LAMOutput
from prompt_dtla.utils.data_utils import BTHWC
from prompt_dtla.utils.logger import log


class ConvLAM(nn.Module):
    cfg: DictConfig
    init_kwargs: Dict

    def setup(self):
        self.idm = ConvIDM(self.cfg.idm, self.init_kwargs)
        self.fdm = ConvFDM(self.cfg.fdm, self.init_kwargs)

    def __call__(
        self,
        x: BTHWC,
        is_training: bool = True,
        latent_actions: jnp.ndarray = None,  # for debugging
        **kwargs,
    ) -> LAMOutput:
        """
        Input:
            x: (B, T, H, W, C) image observations

        Output:
            next_state_pred: (B, T, H, W, C)
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

        log(f"context: {context.shape}, latent_actions: {latent_actions.shape}")

        next_state_pred = self.fdm(
            context,
            latent_actions,
            is_training=is_training,
        )

        if self.cfg.normalize_obs_pred:  # this is necessary for procgen
            next_state_pred = jnp.tanh(next_state_pred) / 2

        return LAMOutput(
            next_obs_pred=next_state_pred,
            action_output=None,
            idm=idm_output,
        )


class ConvIDM(nn.Module):
    """Inverse Dynamics Model for Latent Action Model with VectorQuantization

    IDM - p(z_t | o_t-k,..., o_t, o_t+1)

    Args:
        pass
    """

    cfg: DictConfig
    init_kwargs: Dict

    def setup(self) -> None:
        self.encoder = ConvEncoder(
            self.cfg.encoder_cfg,
            init_kwargs=self.init_kwargs,
        )

        # Predict latent action before inputting to VQ
        self.latent_action_head = MLP(
            hidden_dims=self.cfg.la_mlp_dims,
            init_kwargs=self.init_kwargs,
            activate_final=False,
        )

        self.vq = NAME_TO_VQ_CLS[self.cfg.vq.name](
            self.cfg.vq, init_kwargs=self.init_kwargs
        )

    def __call__(self, x: BTHWC, is_training: bool = True) -> IDMOutput:
        """
        IDM takes the state and next state and predicts the action
        IDM predicts the latent action (z_t) given o_t-k,..., o_t and o_t+1

        Input:
            states: (B, T, H, W, C) image observations

        Output:
            vq_outputs: Dict
        """

        if self.cfg.use_state_diff:
            # take the difference between the current and next state
            x = x[:, 1:] - x[:, :-1]

        # combine T and C dimension so that will be channel input to the CNN
        x = einops.rearrange(x, "b t h w c -> b h w (t c)")
        x_embeds = self.encoder(x, is_training=is_training)

        # predict latent actions
        latent_actions = self.latent_action_head(nn.gelu(x_embeds))

        vq_outputs = None

        # compute quantized latent actions
        if self.cfg.apply_quantization:
            vq_outputs = self.vq(latent_actions, is_training=is_training)

        idm_output = IDMOutput(vq=vq_outputs, latent_actions=latent_actions)

        return idm_output


class ConvFDM(nn.Module):
    """
    Forward Dynamics Model - p(o_t+1 | o_t-k,..., o_t, z_t)
    FDM predicts the next observation o_t+1

    Use U-net style architecture following: https://github.com/schmidtdominik/LAPO/blob/main/lapo/models.py

    Args:
        pass
    """

    cfg: DictConfig
    init_kwargs: Dict

    def setup(self) -> None:
        # https://asiltureli.github.io/Convolution-Layer-Calculator/
        self.encoder = ConvEncoder(
            self.cfg.encoder_cfg,
            init_kwargs=self.init_kwargs,
        )
        self.decoder = ConvDecoder(
            self.cfg.decoder_cfg,
            init_kwargs=self.init_kwargs,
        )

    def __call__(
        self,
        context: BTHWC,
        actions: Float[Array, "B D"],
        is_training: bool = True,
    ) -> BTHWC:
        """
        FDM takes the prev states and latent action and predicts the next state
        T is the length of the context provided.

        For LAPO, T=2, just o_t-1 and o_t
        If we use a Transformer, T is the full sequence or a larger context window

        Input:
            context: (B, T, C, H, W) for image inputs
            actions: (B, D_L)

        Output:
            next_state_pred: (B, T, C, H, W)
        """

        B, T, H, W, C = context.shape

        # combine the timestep dimension into the channel dimension
        context = einops.rearrange(context, "b t h w c -> b h w (t c)")
        actions = einops.rearrange(actions, "b dl -> b 1 1 dl")

        # repeat the actions across the spatial dimensions
        # we will inject the actions into the middle of the U-net
        action_expand = einops.repeat(actions, "b 1 1 dl -> b h w dl", h=H, w=W)

        x = jnp.concatenate([context, action_expand], axis=-1)
        # log(f"shape after concat: {x.shape}")

        # run through U-net encoder
        # intermediates are the outputs of each layer in the U-net
        _, intermediates = self.encoder(
            x, is_training=is_training, return_intermediate=True
        )
        embeddings = intermediates[-1]

        # inject actions into the middle of the U-net
        intermediates[-1] = actions

        # decode the next observation
        # [B, T, C, H, W]
        next_state_pred = self.decoder(
            embeddings,
            intermediates=intermediates,
            context=context,
            is_training=is_training,
        )
        return next_state_pred

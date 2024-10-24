from typing import Dict

import flax.linen as nn
import jax.numpy as jnp
from einops import rearrange, repeat
from jaxtyping import Array, Float
from loguru import logger
from omegaconf import DictConfig

from prompt_dtla.models.mlp import MLP
from prompt_dtla.models.vq import NAME_TO_VQ_CLS
from prompt_dtla.stage.lam.lam_state import StateBasedIDM
from prompt_dtla.stage.lam.utils import IDMOutput, LAMOutput
from prompt_dtla.utils.data_utils import BTD
from prompt_dtla.utils.logger import log


class HierarchicalStateBasedLAM(nn.Module):
    """
    Latent Action Model for continuous action spaces.
    The Inverse Dynamics Model (IDM) predicts latents actions z1 and z2
    where z1 represents the high-level skill and z2 represents the behavior conditioned on z1.

    The Forward Dynamics Model (FDM) predicts k-step future observations.
    Given z1, the FDM predicts the future observation.
    Given z1 and z2, the FDM predicts the sequence of future observations.
    """

    cfg: DictConfig
    state_dim: int
    init_kwargs: Dict

    def setup(self):
        # makes sure we are using residual VQ
        assert (
            self.cfg.idm.vq.name == "residual_vq"
        ), "Use residual VQ for hierarchical LAM"

        self.idm = StateBasedIDM(self.cfg.idm, self.init_kwargs)
        self.fdm = HierarchicalStateBasedFDM(
            self.cfg.fdm, self.state_dim, self.init_kwargs
        )

    def __call__(self, x: BTD, is_training: bool = True, **kwargs) -> LAMOutput:
        """
        Input:
            x: (B, T, D)

        Output:
            next_state_pred_k: (B, T, D)
        """

        # IDM predicts the latent actions (z1, z2) given o_t-k, ..., o_t-1, o_t
        log("inside HierarchicalStateBasedLAM")
        log(f"x: {x.shape}")

        idm_output = self.idm(x, is_training=is_training)
        quantized_latent_actions = idm_output.vq.all_quantized

        # skill latent action and behavior latent action
        z1, z2 = quantized_latent_actions[0], quantized_latent_actions[1]

        context = x[:, :-1]

        log(
            f"context: {context.shape}, latent_actions: {quantized_latent_actions.shape}"
        )

        """
            FDM does two things:
            1. Predict k-step future observation given z1
            FDM predicts o_t+k given o_t-k, ..., o_t and z1

            2. Predict next k-step future observations given z1 and z2
            FDM predicts o_t+1, o_t+2, ... o_t+k given the context, z1 and z2
        """
        k_step_pred, next_k_step_pred = self.fdm(
            context,
            z1,
            z2,
            is_training=is_training,
        )
        return LAMOutput(
            next_obs_pred=k_step_pred,
            next_k_step_pred=next_k_step_pred,
            action_output=None,
            idm=idm_output,
        )


class HierarchicalStateBasedFDM(nn.Module):
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
        self.encoder = nn.RNN(nn.LSTMCell(64), time_major=False, return_carry=False)

        # takes the context, z1 and predicts k-steps ahead
        self.decode_k_step = MLP(
            hidden_dims=self.cfg.decoder_cfg.hidden_dims + [self.state_dim],
            init_kwargs=self.init_kwargs,
            activate_final=False,
        )

        # takes the context, z1, z2 and predicts the next k-steps of observations,
        self.decode_k_steps_rnn = nn.RNN(
            nn.LSTMCell(64), time_major=False, return_carry=False
        )
        self.decode_k_steps_mlp = MLP(
            hidden_dims=self.cfg.decoder_cfg.hidden_dims + [self.state_dim],
            init_kwargs=self.init_kwargs,
            activate_final=False,
        )

    def __call__(
        self,
        context: BTD,
        z1: Float[Array, "B D"],
        z2: Float[Array, "B D"],
        is_training: bool = True,
    ) -> BTD:
        """
        FDM takes the prev states and latent action and predicts the next state
        T is the length of the context provided.

        For LAPO, T=2, just o_t-1 and o_t
        If we use a Transformer, T is the full sequence or a larger context window

        Input:
            context: (B, T-1, D)
            actions: (B, D_L)

        Output:
            k_step_pred: (B, D)
            next_k_step_pred: (B, k_step_preds, D)
        """
        # combine the timestep dimension into the channel dimension

        log("inside HierarchicalStateBasedFDM")
        log(f"context: {context.shape}, z1: {z1.shape}, z2: {z2.shape}")

        z1 = rearrange(z1, "b dl -> b 1 dl")

        # repeat the actions across the spatial dimensions
        z1 = repeat(z1, "b 1 dl -> b t dl", t=context.shape[1])

        z2 = rearrange(z2, "b dl -> b 1 dl")
        z2 = repeat(z2, "b 1 dl -> b t dl", t=context.shape[1])

        # first predict k-steps ahead with z1

        # (B, T, D + D_L)
        context_and_z1 = jnp.concatenate([context, z1], axis=-1)

        context_and_z1 = rearrange(context_and_z1, "b t d -> b (t d)")

        # this is an RNN and we use the last hidden state to predict the next observation
        embeddings = self.encoder(context_and_z1)
        k_step_pred = self.decode_k_step(embeddings, is_training=is_training)

        # predict the next couple steps with z1 and z2
        context_z1_z2 = jnp.concatenate([context, z1, z2], axis=-1)

        # TODO: make this autoregressive
        next_k_step_out = self.decode_k_steps_rnn(context_z1_z2)
        next_k_step_pred = self.decode_k_steps_mlp(
            next_k_step_out, is_training=is_training
        )
        return k_step_pred, next_k_step_pred

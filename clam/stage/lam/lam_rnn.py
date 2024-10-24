from typing import Dict

import flax.linen as nn
import jax.numpy as jnp
from einops import rearrange, repeat
from jaxtyping import Array, Float
from loguru import logger
from omegaconf import DictConfig

from prompt_dtla.models.mlp import MLP
from prompt_dtla.models.vq import NAME_TO_VQ_CLS
from prompt_dtla.stage.lam.utils import IDMOutput, LAMOutput
from prompt_dtla.utils.data_utils import BTD


class LSTM(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x, rng_key=None):
        ScanLSTM = nn.scan(
            nn.LSTMCell,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=1,
            out_axes=1,
        )

        lstm = ScanLSTM(self.features)
        input_shape = x[:, 0].shape
        carry = lstm.initialize_carry(rng_key, input_shape)
        carry, x = lstm(carry, x)
        return x


class RNNStateBasedLAM(nn.Module):
    cfg: DictConfig
    state_dim: int
    init_kwargs: Dict

    def setup(self):
        self.idm = RNNStateBasedIDM(self.cfg.idm, self.init_kwargs)
        self.fdm = RNNStateBasedFDM(self.cfg.fdm, self.state_dim, self.init_kwargs)

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
            next_state_pred: (B, num_step_preds, D)
        """
        idm_context = x[:, : self.cfg.context_len + 1]
        fdm_context = x[:, : self.cfg.context_len]

        next_state_preds = []

        for indx in range(self.cfg.k_step_preds):
            idm_output = self.idm(idm_context, is_training=is_training)

            if latent_actions is None:
                if self.cfg.idm.apply_quantization:
                    latent_actions = idm_output.vq.quantize
                else:
                    latent_actions = idm_output.latent_actions

            # FDM predicts the next state given the context
            next_state_pred = self.fdm(
                context=fdm_context,
                actions=latent_actions,
                is_training=is_training,
            )
            # add an extra time dimension
            next_state_pred = next_state_pred[:, None]

            # add the ground truth next state to the context
            gt_next_state = x[:, self.cfg.context_len + 1 + indx][:, None]
            idm_context = jnp.concatenate([idm_context, gt_next_state], axis=1)

            # add the predicted next state to the context for the FDM
            fdm_context = jnp.concatenate([fdm_context, next_state_pred], axis=1)

            next_state_preds.append(next_state_pred)

        # concatenate the predictions along the time dimension
        next_state_pred = jnp.concatenate(next_state_preds, axis=1)

        return LAMOutput(
            next_obs_pred=next_state_pred,
            action_output=None,
            idm=idm_output,
        )


class RNNStateBasedIDM(nn.Module):
    """Inverse Dynamics Model for Latent Action Model with VectorQuantization

    IDM - p(z_t | o_t-k,..., o_t, o_t+1, o_t+k) where K is the number of steps
    to predict into the future

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

        self.lstm = LSTM(
            features=self.cfg.lstm_hidden_dim,
        )

        # Predict latent action before inputting to VQ
        self.latent_action_head = MLP(
            hidden_dims=self.cfg.la_mlp_dims,
            init_kwargs=self.init_kwargs,
            activate_final=False,
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
        # x = rearrange(x, "b t d -> b (t d)")
        x_embeds = self.state_encoder(x)

        # apply LSTM on the context
        x_embeds = self.lstm(x_embeds, rng_key=self.make_rng("sample"))

        # take the last timestep
        x_embeds = x_embeds[:, -1]

        # predict latent actions
        latent_action = self.latent_action_head(nn.gelu(x_embeds))

        # compute quantized latent actions
        idm_output = IDMOutput(vq=None, latent_actions=latent_action)
        return idm_output


class RNNStateBasedFDM(nn.Module):
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

        self.lstm = LSTM(
            features=self.cfg.lstm_hidden_dim,
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
        """ """

        # combine the timestep dimension into the channel dimension
        actions = rearrange(actions, "b dl -> b 1 dl")

        # repeat the actions across the spatial dimensions
        # we will inject the actions into the middle of the U-net
        action_expand = repeat(actions, "b 1 dl -> b t dl", t=context.shape[1])

        # (B, T, D + D_L)
        x = jnp.concatenate([context, action_expand], axis=-1)
        embeddings = self.encoder(x, is_training=is_training)

        # apply LSTM on the context
        embeddings = self.lstm(embeddings, rng_key=self.make_rng("sample"))

        # take the last timestep
        embeddings = embeddings[:, -1]

        next_state_pred = self.decoder(embeddings, is_training=is_training)

        return next_state_pred

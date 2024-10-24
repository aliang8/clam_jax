from typing import Dict

import einops
import flax.linen as nn
import jax.numpy as jnp
from omegaconf import DictConfig

from prompt_dtla.models.cnn import ConvEncoder
from prompt_dtla.models.mlp import MLP
from prompt_dtla.models.policy import ActionHead
from prompt_dtla.models.transformer import TransformerEncoder
from prompt_dtla.models.vit import ViT
from prompt_dtla.utils.logger import log


class IDM(nn.Module):
    cfg: DictConfig
    is_continuous: bool
    action_dim: int
    init_kwargs: Dict

    def setup(self):
        """
        Inverse Dynamics Model for VPT.
        An Inverse Dynamics Model that predicts the ground truth action
        between two consecutive observations.
        """

        if self.cfg.encoder_cfg.name == "mlp":
            self.state_embed = MLP(
                hidden_dims=self.cfg.encoder_cfg.hidden_dims,
                init_kwargs=self.init_kwargs,
                activate_final=False,
            )
        elif self.cfg.encoder_cfg.name == "vit":
            # Use a ViT transformer to encode sequence of images into patches
            assert self.cfg.image_obs, "ViT only works with image observations"
            self.state_embed = ViT(
                cfg=self.cfg.encoder_cfg, init_kwargs=self.init_kwargs
            )
        elif self.cfg.encoder_cfg.name == "transformer":
            self.state_embed = TransformerEncoder(
                cfg=self.cfg.encoder_cfg, **self.init_kwargs
            )
        elif "cnn_encoder" in self.cfg.encoder_cfg.name:
            if self.cfg.image_obs:
                self.state_embed = ConvEncoder(
                    cfg=self.cfg.encoder_cfg, init_kwargs=self.init_kwargs
                )
            else:
                raise NotImplementedError()

        # output head to predict ground truth action
        self.action_head = ActionHead(
            gaussian_policy=self.cfg.gaussian_policy,
            is_continuous=self.is_continuous,
            action_dim=self.action_dim,
            init_kwargs=self.init_kwargs,
        )

    def __call__(
        self,
        x: jnp.ndarray,
        timestep: jnp.ndarray = None,
        is_training: bool = True,
    ):
        """IDM takes the state and next state and predicts the action

        Args:
            x: (B, T, D) or (B, T, H, W, C) if state or image observations

        Return:
            action_output
        """
        log("inside VPT IDM")

        if self.cfg.use_state_diff:
            x = x[:, 1:] - x[:, :-1]

        if self.cfg.encoder_cfg.name == "vit":
            state_embeds = self.state_embed(
                x, timestep=timestep, is_training=is_training
            )
            cls_token_emb = state_embeds[:, :-1, 0]

            if self.cfg.use_cls_embedding:
                state_embeds = cls_token_emb
            else:
                raise NotImplementedError()

        elif self.cfg.encoder_cfg.name == "transformer":
            # first embed the x
            if self.cfg.image_obs:
                b = x.shape[0]
                x = einops.rearrange(x, "b t h w c -> (b t) c h w")
                state_embeds = self.state_embed(x, is_training=is_training)
                state_embeds = einops.rearrange(state_embeds, "(b t) d -> b t d", b=b)
            else:
                raise NotImplementedError()

            state_embeds = nn.gelu(state_embeds)
            state_embeds = self.transformer(state_embeds, is_training=is_training)
            state_embeds = state_embeds[:, 1:]
            # ignore the first timestep embedding
            # the resulting VQ actions should have T-1 outputs because we are
            # predicting the action that took us from t to t+1
            log(f"shape after transformer enc: {state_embeds.shape}")
        else:
            if self.cfg.image_obs:
                # combine T and C dimension so that will be channel input to the CNN
                x = einops.rearrange(x, "b t h w c -> b h w (t c)")
            else:
                x = einops.rearrange(x, "b t d -> b (t d)")

            state_embeds = self.state_embed(x, is_training=is_training)

        log(f"states shape: {x.shape}")
        log(f"state_embeds shape: {state_embeds.shape}")

        # predict action logits
        actions = self.action_head(nn.gelu(state_embeds), is_training=is_training)
        return actions

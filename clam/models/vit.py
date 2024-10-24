from typing import Dict

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from einops import rearrange, repeat
from omegaconf import DictConfig

from prompt_dtla.models.transformer import TransformerEncoder
from prompt_dtla.utils.logger import log


class ViT(nn.Module):
    """
    ViT architecture for implementing Image-based IDM
    """

    cfg: DictConfig
    init_kwargs: Dict

    def setup(self):
        self.image_height, self.image_width = self.cfg.image_shape
        self.patch_height, self.patch_width = self.cfg.patch_size

        assert self.image_height % self.patch_height == 0

        num_patches = (self.image_height // self.patch_height) * (
            self.image_width // self.patch_width
        )
        self.num_patches = num_patches

        self.to_patch_embedding = nn.Sequential(
            [
                nn.LayerNorm(),
                nn.Dense(self.cfg.embedding_dim, **self.init_kwargs),
                nn.LayerNorm(),
            ]
        )

        self.timestep_embed = nn.Embed(
            num_embeddings=3000,
            features=self.cfg.embedding_dim,
            embedding_init=nn.initializers.normal(stddev=0.02),
        )

        self.transformer = TransformerEncoder(
            **self.cfg.model_cfg, causal=False, init_kwargs=self.init_kwargs
        )

        # these are learnable parameters
        self.cls_token = self.param(
            "cls_token",
            nn.initializers.normal(stddev=0.02),
            (self.cfg.max_timesteps, self.cfg.embedding_dim),
        )

        self.img_pos_enc = self.param(
            "img_pos_enc",
            nn.initializers.normal(stddev=0.02),
            (self.num_patches, self.cfg.embedding_dim),
        )

        self.cls_token_pos = self.param(
            "cls_token_pos",
            nn.initializers.normal(stddev=0.02),
            (self.cfg.max_timesteps, self.cfg.embedding_dim),
        )

    @nn.compact
    def __call__(
        self,
        states: jnp.ndarray,
        timestep: jnp.ndarray = None,
        is_training: bool = True,
    ):
        """
          Predicts latent action from sequence of images. We patchify the images
          and pass them through a linear layer and then a bidirectional transformer encoder.

          Add patch embed and timestep embed for each image in the sequence.

          o_1, o_2, .... o_T -> [p_1, p_2, ...]


        Full self-attention, every patch can attend to every other patch

            z1                   z2                       zT
            |                    |                        |
          [CLS - - ... -]      [CLS - - ... -]     ...  [CLS - - ... -]       patch embeddings
                   |                   |                         |
           --------------------------------------------------------
          |                                                        |
          |                                                        |
          |                        ViT Encoder                     |
          |                                                        |
          |                                                        |
           --------------------------------------------------------
                   |                   |                         |
                  o1                   o2                       oT
          [CLS p1 p2 ... pM]   [CLS p1 p2 ... pM]  ...  [CLS p1 p2 ... pM]    patches
        + [1   1  1  ... 1]    [2   2  2  ... 2]   ...  [T   T  T  ... T]     timestep embedding
        + [*   1  2  ... M]    [*   1  2  ... M]   ...  [*   1  2  ... M]     patch position embedding

          Args:
              states: (B, T, H, W, C)

          Return:
              embedding: jnp.ndarray: (B, T-1, D)
        """
        cls_token = self.variables["params"]["cls_token"]
        img_pos_enc = self.variables["params"]["img_pos_enc"]
        cls_token_pos = self.variables["params"]["cls_token_pos"]

        b, t = states.shape[:2]

        patches = rearrange(
            states,
            "b t (h p1) (w p2) c -> b t (h w) (p1 p2 c)",
            p1=self.patch_height,
            p2=self.patch_width,
            t=t,
        )
        # [B, T, # patches, D]
        patch_emb = self.to_patch_embedding(patches)

        # add timestep embedding
        if timestep is None:
            timestep = jnp.arange(t)
            timestep = repeat(timestep, "t -> b t", b=b)

        # add extra dimension
        timestep = timestep[..., None]
        timestep_embed = self.timestep_embed(timestep.astype(jnp.int32).squeeze(-1))
        timestep_embed = repeat(timestep_embed, "b t d -> b t n d", n=self.num_patches)
        patch_emb += timestep_embed

        # one CLS token per timestep
        cls_token = repeat(cls_token[:t], "t d -> b t 1 d", b=b, t=t)

        # [B, T, 1+num_patches, D]
        patch_emb = jnp.concatenate([cls_token, patch_emb], axis=2)

        # ADD POSITION ENCODINGS
        # repeat img_pos_enc for each timestep
        img_pos_enc = repeat(img_pos_enc, "n d -> b t n d", b=b, t=t)
        cls_token_pos = repeat(cls_token_pos[:t], "t d -> b t 1 d", b=b, t=t)
        pos_enc = jnp.concatenate([cls_token_pos, img_pos_enc], axis=2)

        patch_emb += pos_enc

        patch_emb = nn.Dropout(
            rate=self.cfg.dropout_rate, deterministic=not is_training
        )(patch_emb)

        # reshape to [B, T * (num_patches + 1), D]
        patch_emb = rearrange(patch_emb, "b t n d -> b (t n) d")

        log(f"patch_emb shape: {patch_emb.shape}")
        # should be non-causal
        x = self.transformer(patch_emb)

        embeddings = rearrange(
            x, "b (t n) d -> b t n d", b=b, t=t, n=self.num_patches + 1
        )
        return embeddings


if __name__ == "__main__":
    config = DictConfig(
        {
            "image_shape": [64, 64],
            "patch_size": [8, 8],
            "embedding_dim": 512,
            "max_timesteps": 100,
            "dropout_rate": 0.1,
            "num_patches": 64,
            "model_cfg": {
                "embedding_dim": 512,
                "num_heads": 8,
                "num_layers": 6,
                "attn_size": 64,
                "dropout_rate": 0.1,
            },
        }
    )

    init_kwargs = {"kernel_init": nn.initializers.xavier_uniform()}
    vit = ViT(config, init_kwargs)

    states = np.random.randn(2, 100, 64, 64, 3)
    timestep = np.random.randint(0, 100, (2, 100))

    keys = {"params": jax.random.key(0), "vit": jax.random.key(1)}
    params = vit.init(jax.random.PRNGKey(0), states, timestep)
    out = vit.apply(params, states, timestep, is_training=True, rngs=keys)

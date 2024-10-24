from typing import Dict

import jax
import jax.numpy as jnp
import numpy as np
from einops import rearrange, repeat
from flax import linen as nn
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from prompt_dtla.models.transformer import TransformerDecoder
from prompt_dtla.models.vit import ViT
from prompt_dtla.models.vq import NAME_TO_VQ_CLS
from prompt_dtla.stage.lam.utils import IDMOutput, LAMOutput
from prompt_dtla.utils.data_utils import BTX


def create_lower_triangular_block_matrix(B, n_blocks):
    """
    Create a lower triangular block matrix using a given block matrix B.

    Parameters:
    B (np.ndarray): The block matrix to be repeated.
    n_blocks (int): The number of blocks along the diagonal.

    Returns:
    np.ndarray: The lower triangular block matrix.
    """
    # Create a lower triangular matrix of size n_blocks
    L = np.tril(np.ones((n_blocks, n_blocks)))

    # Generate the block matrix using the Kronecker product
    lower_triangular_block_matrix = np.kron(L, B)
    return lower_triangular_block_matrix


class ViTLAM(nn.Module):
    cfg: DictConfig
    init_kwargs: Dict

    def setup(self):
        self.idm = ViTIDM(self.cfg.idm, self.init_kwargs)
        self.fdm = ViTFDM(self.cfg.fdm, self.init_kwargs)

    def __call__(self, x: BTX, is_training: bool = True, **kwargs) -> LAMOutput:
        """
        Input:
            x: (B, T, D) or (B, T, H, W, C) if state or image observations

        Output:
            next_state_pred: (B, T, D) or (B, T, H, W, C)
        """

        # IDM predicts the latent action (z_t) given o_t-k, ..., o_t and o_t+1
        # video embedding - B, T, num_patches+1, D
        video_embedding, idm_output = self.idm(x, is_training=is_training)
        quantized_latent_actions = idm_output.vq.quantize

        log(
            f"video_embedding: {video_embedding.shape}, latent_actions: {quantized_latent_actions.shape}"
        )

        next_state_pred = self.fdm(
            video_embedding,
            quantized_latent_actions,
            is_training=is_training,
        )

        if self.cfg.normalize_obs_pred:  # this is necessary for procgen
            next_state_pred = jnp.tanh(next_state_pred) / 2

        # log(f"next_state_pred: {next_state_pred.shape}")
        return LAMOutput(
            next_obs_pred=next_state_pred,
            action_output=None,
            idm=idm_output,
        )


class ViTIDM(nn.Module):
    cfg: DictConfig
    init_kwargs: Dict

    def setup(self):
        """
        ViT architecture for implementing Image-based IDM
        """
        # TODO: fix this hacky way to add model_cfg to the encoder_cfg
        model_cfg = DictConfig(
            {
                "num_heads": self.cfg.encoder_cfg.num_heads,
                "num_layers": self.cfg.encoder_cfg.num_layers,
                "attn_size": self.cfg.encoder_cfg.attn_size,
                "dropout_rate": self.cfg.encoder_cfg.dropout_rate,
            }
        )
        image_shape = self.cfg.encoder_cfg.image_shape
        encoder_cfg = OmegaConf.to_container(self.cfg.encoder_cfg)
        encoder_cfg.update(model_cfg=model_cfg, image_shape=image_shape)
        encoder_cfg = DictConfig(encoder_cfg)

        self.vit = ViT(encoder_cfg, init_kwargs=self.init_kwargs)
        self.to_act = nn.Dense(self.cfg.vq.code_dim, **self.init_kwargs)
        vq_cls = NAME_TO_VQ_CLS[self.cfg.vq.name]
        self.vq = vq_cls(self.cfg.vq, init_kwargs=self.init_kwargs)

    def __call__(
        self,
        states: jnp.ndarray,
        timestep: jnp.ndarray = None,
        is_training: bool = True,
    ):
        # output of this is # [B, T, N+1, D] because of the CLS token
        log(f"states: {states.shape}")

        patch_embeddings = self.vit(
            states=states, timestep=timestep, is_training=is_training
        )

        if self.cfg.encoder_cfg.use_cls_embedding:
            # remove the CLS token
            embeddings = patch_embeddings[:, :, 0]
        else:
            # average all the tokens
            embeddings = jnp.mean(patch_embeddings[:, :, 1:], axis=2)

        # [B, T, D]
        b, t = embeddings.shape[:2]

        # [BT, D]
        latent_actions = rearrange(embeddings, "b t d -> (b t) d")

        # apply a linear layer to the latent actions
        latent_actions = self.to_act(latent_actions)

        # apply VQ to latent actions to discretize them
        vq_outputs = self.vq(jax.nn.relu(latent_actions), is_training=is_training)

        # reshape back to [B, T, ...]
        def reshape_obs(x):
            if x is not None and isinstance(x, jnp.ndarray) and len(x.shape) > 0:
                return rearrange(x, "(b t) ... -> b t ...", b=b, t=t)
            return x

        vq_outputs = jax.tree_util.tree_map(reshape_obs, vq_outputs)
        return patch_embeddings, IDMOutput(latent_actions=latent_actions, vq=vq_outputs)


class ViTFDM(nn.Module):
    cfg: DictConfig
    init_kwargs: Dict

    def setup(self):
        """
        Patch based transformer FDM. This is just a Transformer decoder.
        """
        self.patch_height, self.patch_width = self.cfg.decoder_cfg.patch_size

        self.image_height, self.image_width = self.cfg.decoder_cfg.image_shape
        self.num_patches = (self.image_height // self.patch_height) * (
            self.image_width // self.patch_width
        )
        model_cfg = DictConfig(
            {
                "num_heads": self.cfg.decoder_cfg.num_heads,
                "num_layers": self.cfg.decoder_cfg.num_layers,
                "attn_size": self.cfg.decoder_cfg.attn_size,
                "dropout_rate": self.cfg.decoder_cfg.dropout_rate,
            }
        )
        self.decoder = TransformerDecoder(**model_cfg, init_kwargs=self.init_kwargs)
        self.to_vid = nn.Dense(
            features=self.patch_height * self.patch_width * 3,
            **self.init_kwargs,
        )

        self.to_proj = nn.Dense(
            features=self.cfg.decoder_cfg.attn_size, **self.init_kwargs
        )

    def __call__(
        self,
        video_embedding: jnp.ndarray,  # (B, T, N, D)
        latent_actions: jnp.ndarray,  # (B, T, D)
        timestep: jnp.ndarray = None,
        src_mask: jnp.ndarray = None,
        tgt_mask: jnp.ndarray = None,
        is_training: bool = True,
    ):
        """
        Transformer decoder to predict future states from latent action

        Args:
            video_embedding: (B, T, N+1, D)
            latent_actions: (B, T, D)
            is_training: bool
        Out:
            Reconstructed patches: (B, T, C, H, W)
        """
        # make sure the timesteps for states and latent_actions are the same
        assert video_embedding.shape[1] == latent_actions.shape[1]

        # we can ignore the CLS token for each timestep
        video_embedding = video_embedding[:, :, 1:]
        b, t, _, _ = video_embedding.shape
        video_embedding = rearrange(video_embedding, "b t n d -> b (t n) d")

        target_len = video_embedding.shape[1]
        source_len = latent_actions.shape[1]

        # should be lower triangular block matrix
        causal_mask = create_lower_triangular_block_matrix(
            np.ones((self.num_patches, self.num_patches)), t
        )

        # controls attention weights to encoder output
        # 1 means attend to this position, 0 means do not attend
        src_mask = np.ones((target_len, source_len))

        # source is coming from the encoder outputs
        src_mask = repeat(src_mask, "tl sl -> b t tl sl", b=b, t=1)

        # first project the latent actions to the same dimension as video embedding
        latent_actions = self.to_proj(latent_actions)

        # [B, T * num_patches, D]
        patch_emb = self.decoder(
            embeddings=video_embedding,
            cond=latent_actions,
            src_mask=src_mask,
            # tgt_mask=tgt_mask,
            causal_mask=causal_mask,
            is_training=is_training,
        )

        reconstructed_patches = rearrange(
            patch_emb, "b (t n) d -> b t n d", b=b, t=t, n=self.num_patches
        )

        # apply linear layer to decode
        # reconstruct patches of the next image
        reconstructed_patches = self.to_vid(reconstructed_patches)

        # reshape back to image size
        reconstructed_patches = rearrange(
            reconstructed_patches,
            "b t (h w) (p1 p2 c) -> b t (h p1) (w p2) c",
            h=int(self.image_height // self.patch_height),
            w=int(self.image_width // self.patch_width),
            p1=self.patch_height,
            p2=self.patch_width,
            c=3,
        )
        log(f"reconstructed_patches: {reconstructed_patches.shape}")
        return reconstructed_patches


if __name__ == "__main__":
    import jax
    from omegaconf import DictConfig

    codebook_config = DictConfig(
        dict(
            code_dim=64,
            num_codes=512,
            ema_decay=0.8,
            eps=1e-5,
            ema_update=True,
            num_codebooks=2,
            threshold_ema_dead_code=2.0,
            codebook_dim=32,
            sample_codebook_temp=1.0,
            stochastic_sample_codes=False,
            straight_through=False,
            affine_params=False,
            affine_param_batch_decay=0.99,
            affine_param_codebook_decay=0.9,
            sync_affine_param=True,
            learnable_codebook=False,
        )
    )

    vq_config = DictConfig(
        dict(
            name="vq",
            code_dim=64,
            codebook_cls="euclidean",
            learnable_codebook=False,
            separate_codebook_per_head=True,
            codebook_diversity_loss_weight=1.0,
            codebook_diversity_temp=1.0,
            beta=1.0,
            ema_update=True,
            codebook_dim=32,
            heads=1,
            codebook_cfg=codebook_config,
            num_codes=512,
            sync_update_v=0,
        )
    )
    config = DictConfig(
        dict(
            encoder_cfg=dict(
                image_shape=(64, 64),
                patch_size=(8, 8),
                num_channels=3,
                hidden_dim=128,
                dropout_rate=0.1,
                max_timesteps=100,
                num_patches=64,
                model_cfg=dict(
                    hidden_dim=128,
                    num_heads=8,
                    num_layers=4,
                    attn_size=128,
                    dropout_rate=0.1,
                ),
                use_cls_embedding=True,
            ),
            decoder_cfg=dict(
                image_shape=(64, 64),
                patch_size=(8, 8),
                num_channels=3,
                hidden_dim=128,
                dropout_rate=0.1,
                max_timesteps=100,
                num_patches=64,
                model_cfg=dict(
                    hidden_dim=128,
                    num_heads=8,
                    num_layers=4,
                    attn_size=128,
                    dropout_rate=0.1,
                ),
                use_cls_embedding=True,
            ),
            epsilon=1e-5,
            num_codebooks=2,
            num_codes=6,
            num_discrete_latents=4,
            ema_decay=0.99,
            beta=0.25,
            image_shape=[64, 64],
            vq=vq_config,
        )
    )
    init_kwargs = {"kernel_init": nn.initializers.xavier_uniform()}
    video = jnp.ones((1, 10, 64, 64, 3))

    # model = ViTIDM(config, init_kwargs=init_kwargs)
    # params = model.init(
    #     jax.random.PRNGKey(42), states=video, timestep=jnp.ones((1, 10))
    # )

    # model = ViTFDM(config, init_kwargs=init_kwargs)
    # params = model.init(
    #     jax.random.PRNGKey(42),
    #     video_embedding=jnp.ones((1, 10, 65, 128)),
    #     latent_actions=jnp.ones((1, 10, 128)),
    # )

    # model = ViTLAM(config, state_dim=128, init_kwargs=init_kwargs)
    # params = model.init(
    #     jax.random.PRNGKey(42),
    #     x=video,
    # )

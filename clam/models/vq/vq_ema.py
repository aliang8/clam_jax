from typing import Dict

import einops
import flax.linen as nn
import jax
import jax.core
import jax.numpy as jnp
from jaxtyping import Array, Float
from loguru import logger
from omegaconf import DictConfig

from prompt_dtla.models.vq.utils import VQOutput, bounded_uniform


class VQEmbeddingEMA(nn.Module):
    cfg: DictConfig
    init_kwargs: Dict = None

    def setup(self):
        cfg = self.cfg
        # todo also need to handle init_embedding (see Anthony's haiku version)

        self.embedding = self.variable(
            "vq",
            "embedding",
            bounded_uniform(-1.0 / cfg.num_embs * 5, 1.0 / cfg.num_embs * 5),
            self.make_rng("vq"),
            (cfg.num_codebooks, cfg.num_embs, cfg.emb_dim),
            jnp.float32,
        )

        self.ema_count = self.variable(
            "vq",
            "ema_count",
            nn.initializers.zeros,
            self.make_rng("vq"),
            (cfg.num_codebooks, cfg.num_embs),
            jnp.int32,
        )

        self.ema_weight = self.variable(
            "vq",
            "ema_weight",
            bounded_uniform(-1.0 / cfg.num_embs * 5, 1.0 / cfg.num_embs * 5),
            self.make_rng("vq"),
            (cfg.num_codebooks, cfg.num_embs, cfg.emb_dim),
            jnp.float32,
        )

        log(f"embedding shape: {self.embedding.value.shape}")
        log(f"ema_weight shape: {self.ema_weight.value.shape}")
        log(f"ema_count shape: {self.ema_count.value.shape}")

    def forward_2d(self, x, is_training=True):
        """
        einops notation:

        b ... batch size
        n ... num_codebooks
        d ... codebook dimension
        m ... number of codes per codebook
        """
        cfg = self.cfg

        embedding = self.embedding.value
        ema_count = self.ema_count.value
        ema_weight = self.ema_weight.value

        B, C, H, W = x.shape  # at some point.. B T H W C -> B (T C) H W
        N, M, D = embedding.shape
        assert C == N * D, f"{C} != {N} * {D}"

        log(f"batch size: {B}, num codebooks: {N}, codebook dim: {D}, num codes: {M}")
        log(f"in shape: {x.shape}")

        x = einops.rearrange(x, "b (n d) h w -> n b h w d", n=N, d=D)
        x_flat = einops.rearrange(x, "n b h w d -> n (b h w) d")
        x_flat = jax.lax.stop_gradient(x_flat)

        log(f"x_flat shape: {x_flat.shape}")

        to_add = jnp.expand_dims(jnp.sum(embedding**2, axis=2), axis=1)
        to_add += jnp.sum(x_flat**2, axis=2, keepdims=True)

        embedding_t = jnp.swapaxes(embedding, 1, 2)
        b1b2 = jax.lax.batch_matmul(x_flat, embedding_t)
        alpha = -2.0
        beta = 1.0

        # [N, B, D]
        distances = beta * to_add + alpha * b1b2
        log(f"distances shape: {distances.shape}")

        # [N, B]
        indices = jnp.argmin(distances, axis=-1)
        log(f"indices shape: {indices.shape}")

        # [N, B, M]
        encodings = jax.nn.one_hot(indices, M).astype(jnp.float32)
        log(f"encodings shape: {encodings.shape}")

        indices_exp = einops.repeat(indices, "n b -> n b d", d=D)

        quantized = jnp.take_along_axis(embedding, indices_exp, axis=1)
        log(f"quantized shape: {quantized.shape}")
        quantized = quantized.reshape(x.shape)

        if is_training:
            new_ema_count = cfg.ema_decay * ema_count + (1 - cfg.ema_decay) * jnp.sum(
                encodings, axis=1
            )

            n = jnp.sum(new_ema_count, axis=-1, keepdims=True)

            new_ema_count = (new_ema_count + cfg.eps) / (n + M * cfg.eps) * n

            encodings_t = jnp.swapaxes(encodings, 1, 2)
            dw = jax.lax.batch_matmul(encodings_t, x_flat)

            new_ema_weight = cfg.ema_decay * ema_weight + (1 - cfg.ema_decay) * dw

            new_embeddings = new_ema_weight / jnp.expand_dims(new_ema_count, axis=-1)

            if not self.is_initializing():
                self.embedding.value = new_embeddings
                self.ema_count.value = new_ema_count
                self.ema_weight.value = new_ema_weight

        e_latent_loss = jnp.mean((x - jax.lax.stop_gradient(quantized)) ** 2)
        loss = cfg.beta * e_latent_loss

        quantized = jax.lax.stop_gradient(quantized) + (x - jax.lax.stop_gradient(x))
        avg_probs = jnp.mean(encodings, axis=1)
        perplexity = jnp.exp(-jnp.sum(avg_probs * jnp.log(avg_probs + 1e-10), axis=-1))
        quantized = einops.rearrange(quantized, "n b h w d -> b (n d) h w", n=N, d=D)

        indices = einops.rearrange(indices, "n (b h w) -> b n h w", n=N, b=B, h=H, w=W)
        return quantized, loss, perplexity.sum(), indices

    def __call__(
        self,
        x: Float[Array, "batch emb_dim"],
        is_training: bool = True,
    ):
        cfg = self.cfg
        bs = x.shape[0]
        x = einops.rearrange(
            x,
            "b (n d l) -> b (n d) l 1",
            b=bs,
            n=cfg.num_codebooks,
            d=cfg.emb_dim,
            l=cfg.num_discrete_latents,
        )
        z_q, loss, perplexity, indices = self.forward_2d(x, is_training=is_training)
        z_q = z_q.reshape(
            bs,
            cfg.num_codebooks * cfg.num_discrete_latents * cfg.emb_dim,
        )

        encodings = self.inds_to_z_q(indices)

        return VQOutput(
            quantize=z_q,
            loss=loss,
            perplexity=perplexity,
            encoding_indices=indices,
            encodings=encodings,
        )

    def inds_to_z_q(self, indices):
        """look up quantization inds in embedding"""

        # todo other embedding needs to be made in setup. confusion on Normal vs. Uniform

        N, M, D = self.embedding.value.shape
        B, N_, H, W = indices.shape
        assert N == N_

        # N ... num_codebooks
        # M ... num_embs
        # D ... emb_dim
        # H ... num_discrete_latents (kinda)
        inds_flat = einops.rearrange(indices, "b n h w -> n (b h w)")
        inds_flat = einops.repeat(inds_flat, "n b -> n b d", d=D)
        quantized = jnp.take_along_axis(self.embedding.value, inds_flat, axis=1)
        quantized = quantized.reshape(N, B, H, W, D)
        quantized = einops.rearrange(quantized, "n b h w d -> b (n d) h w", n=N, d=D)

        return (
            quantized  # shape is (B, num_codebooks * emb_dim, num_discrete_latents, 1)
        )

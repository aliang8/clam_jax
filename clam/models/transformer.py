from typing import Dict

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from prompt_dtla.utils.logger import log


class TransformerEncoderLayer(nn.Module):
    num_heads: int
    attn_size: int
    dropout_rate: float
    widening_factor: int = 4
    init_kwargs: Dict = None

    def setup(self):
        self.attn_block = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.attn_size,
            dropout_rate=self.dropout_rate,
            **self.init_kwargs,
        )

        self.dense_block = nn.Sequential(
            [
                nn.Dense(self.attn_size * self.widening_factor, **self.init_kwargs),
                nn.gelu,
                nn.Dense(self.attn_size, **self.init_kwargs),
            ]
        )

        self.ln = nn.LayerNorm()

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        mask: jnp.ndarray,
        is_training: bool = True,
    ) -> jnp.ndarray:
        h_norm = self.ln(x)
        h_attn = self.attn_block(
            h_norm, h_norm, h_norm, mask=mask, deterministic=not is_training
        )
        h = x + h_attn

        h_norm = self.ln(h)
        h_dense = self.dense_block(h_norm)
        h_dense = nn.Dropout(rate=self.dropout_rate, deterministic=not is_training)(
            h_dense
        )
        h = h + h_dense
        return h


class TransformerEncoder(nn.Module):
    num_heads: int
    num_layers: int
    attn_size: int
    dropout_rate: float
    widening_factor: int = 4
    causal: bool = False
    init_kwargs: Dict = None

    def setup(self):
        self.layers = [
            TransformerEncoderLayer(
                num_heads=self.num_heads,
                attn_size=self.attn_size,
                dropout_rate=self.dropout_rate,
                widening_factor=self.widening_factor,
                init_kwargs=self.init_kwargs,
            )
            for _ in range(self.num_layers)
        ]

        self.ln = nn.LayerNorm()

    def __call__(
        self,
        embeddings: jnp.ndarray,  # [B, T, D]
        mask: jnp.ndarray = None,  # [B, T]
        causal_mask: jnp.ndarray = None,  # [B, T, T]
        is_training: bool = True,
    ) -> jnp.ndarray:  # [B, T, D]
        """Transforms input embedding sequences to output embedding sequences."""
        log("=" * 50)
        log("inside transformer encoder")
        log(f"input embeddings shape: {embeddings.shape}")
        B, seq_len, D = embeddings.shape

        if mask is None:  # don't mask anything
            mask = jnp.ones((B, seq_len))

        # Compute causal mask for autoregressive sequence modelling.
        mask = mask[:, None, None, :]  # [B, H=1, T'=1, T]

        if self.causal:
            if causal_mask is None:
                causal_mask = np.tril(
                    np.ones((1, 1, seq_len, seq_len))
                )  # [B=1, H=1, T, T]

            mask = mask * causal_mask  # [B, H=1, T, T]

        h = embeddings
        for i, layer in enumerate(self.layers):
            log(f"layer {i}, h shape: {h.shape}")
            h = layer(h, mask, is_training=is_training)

        return self.ln(h)


class TransformerDecoderLayer(nn.Module):
    num_heads: int
    attn_size: int
    dropout_rate: float
    widening_factor: int = 4
    init_kwargs: Dict = None

    def setup(self):
        self.attn_block_src = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.attn_size,
            dropout_rate=self.dropout_rate,
            **self.init_kwargs,
        )

        self.attn_block_tgt = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.attn_size,
            dropout_rate=self.dropout_rate,
            **self.init_kwargs,
        )

        self.dense_block = nn.Sequential(
            [
                nn.Dense(self.attn_size * self.widening_factor, **self.init_kwargs),
                nn.gelu,
                nn.Dense(self.attn_size, **self.init_kwargs),
            ]
        )

        self.ln = nn.LayerNorm()

    @nn.compact
    def __call__(self, x, key, value, src_mask, tgt_mask, is_training: bool = True):
        # layer norm -> attention -> dropout -> residual connection
        h_norm = self.ln(x)
        h_attn = self.attn_block_src(
            h_norm, h_norm, h_norm, mask=tgt_mask, deterministic=not is_training
        )
        h = x + h_attn

        # cross attention with encoder output -> dropout -> residual connection
        h_norm = self.ln(h)
        h_cross_attn = self.attn_block_tgt(
            h_norm, key, value, mask=src_mask, deterministic=not is_training
        )
        h = h + h_cross_attn

        h_norm = self.ln(h)
        h_dense = self.dense_block(h_norm)
        h_dense = nn.Dropout(rate=self.dropout_rate, deterministic=not is_training)(
            h_dense
        )
        h = h + h_dense
        return h


class TransformerDecoder(nn.Module):
    num_heads: int
    num_layers: int
    attn_size: int
    dropout_rate: float
    widening_factor: int = 4
    init_kwargs: Dict = None

    def setup(self):
        self.layers = [
            TransformerDecoderLayer(
                num_heads=self.num_heads,
                attn_size=self.attn_size,
                dropout_rate=self.dropout_rate,
                widening_factor=self.widening_factor,
                init_kwargs=self.init_kwargs,
            )
            for _ in range(self.num_layers)
        ]

        self.ln = nn.LayerNorm()

    def __call__(
        self,
        embeddings: jnp.ndarray,  # [B, T, D]
        cond: jnp.ndarray,  # [B, T, D]
        src_mask: jnp.ndarray = None,  # [B, T]
        tgt_mask: jnp.ndarray = None,  # [B, T]
        causal_mask: jnp.ndarray = None,  # [B, T, T]
        is_training: bool = True,
    ) -> jnp.ndarray:
        log("=" * 50)
        log("inside transformer decoder")
        log(f"input embeddings shape: {embeddings.shape}")
        B, seq_len, D = embeddings.shape

        if tgt_mask is None:
            tgt_mask = jnp.ones((B, seq_len)).astype(jnp.int32)

        # Compute causal mask for autoregressive sequence modelling.
        tgt_mask = tgt_mask[:, None, None, :]  # [B, H=1, T'=1, T]

        if causal_mask is None:
            causal_mask = np.tril(np.ones((1, 1, seq_len, seq_len)))  # [B=1, H=1, T, T]

        # apply mask to the target sequence
        tgt_mask = tgt_mask * causal_mask  # [B, H=1, T, T]

        # source mask is for the encoder output
        h = embeddings

        for i, layer in enumerate(self.layers):
            log(f"layer {i}, h shape: {h.shape}")
            h = layer(
                x=h,
                key=cond,
                value=cond,
                src_mask=src_mask,
                tgt_mask=tgt_mask,
                is_training=is_training,
            )

        return self.ln(h)


if __name__ == "__main__":
    # test encoder
    B = 2
    T = 5
    D = 64
    num_heads = 4
    num_layers = 2
    attn_size = 64
    dropout_rate = 0.1

    encoder = TransformerEncoder(
        num_heads=num_heads,
        num_layers=num_layers,
        attn_size=attn_size,
        dropout_rate=dropout_rate,
        init_kwargs={"kernel_init": nn.initializers.xavier_uniform()},
    )

    states = jax.random.normal(jax.random.PRNGKey(0), (B, T, D))
    mask = jnp.ones((B, T))

    params = encoder.init(jax.random.PRNGKey(0), states, mask)
    keys = {
        "params": jax.random.key(0),
    }
    out = encoder.apply(params, states, mask, rngs=keys)
    print(out)

    log(f"output shape: {out.shape}")

    decoder = TransformerDecoder(
        num_heads=num_heads,
        num_layers=num_layers,
        attn_size=attn_size,
        dropout_rate=dropout_rate,
        init_kwargs={"kernel_init": nn.initializers.xavier_uniform()},
    )

    states = jnp.ones((B, T, D))
    mask = jnp.ones((B, T))
    enc_out = jnp.ones((B, T, D))
    src_mask = jnp.ones((B, 1, T, T))
    tgt_mask = jnp.ones((B, T))

    params = decoder.init(jax.random.PRNGKey(0), states, enc_out, src_mask, tgt_mask)

    out = decoder.apply(params, states, enc_out, src_mask, tgt_mask, rngs=keys)
    log(f"output shape: {out.shape}")

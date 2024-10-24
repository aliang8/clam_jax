from typing import Dict

import einops
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from loguru import logger


def round_ste(z):
    """Round with straight through gradients."""
    zhat = jnp.round(z)
    return z + jax.lax.stop_gradient(zhat - z)


class FSQ(hk.Module):
    """
    Finite State Quantization (https://arxiv.org/pdf/2309.15505) from GDM.

    Quantize to nearest point of a d-dimensional hypercube.
    Should be a drop-in replacement for VQ-VAE.

    Benefits over VQ-VAE:
        - removes the auxiliary loss
        - better codebook utilization
        - does not need EMA on codebook
    """

    def __init__(
        self,
        levels: list[int],
        dim: int,
        eps: float = 1e-3,
        num_codebooks: int = 1,
        init_kwargs: Dict = None,
    ):
        super().__init__("FSQ")
        self._levels = levels
        self._eps = eps
        self._levels_np = np.asarray(levels)
        self._basis = np.concatenate(([1], np.cumprod(self._levels_np[:-1]))).astype(
            np.uint32
        )

        self._implicit_codebook = self.indexes_to_codes(np.arange(self.codebook_size))

        self.dim = dim * num_codebooks
        self.num_codebooks = num_codebooks
        effective_codebook_dim = self.num_dimensions * num_codebooks
        has_projections = self.dim != effective_codebook_dim

        if has_projections:
            self.project_in = hk.Linear(effective_codebook_dim, **init_kwargs)
            self.project_out = hk.Linear(self.dim, **init_kwargs)
        else:
            self.project_in = lambda x: x
            self.project_out = lambda x: x

    @property
    def num_dimensions(self) -> int:
        """Number of dimensions expected from inputs."""
        return len(self._levels)

    @property
    def codebook_size(self) -> int:
        """Size of the codebook."""
        return np.prod(self._levels)

    @property
    def codebook(self):
        """Returns the implicit codebook. Shape (prod(levels), num_dimensions)."""
        return self._implicit_codebook

    def bound(self, z: jax.Array) -> jax.Array:
        """Bound `z`, an array of shape (..., d)."""
        half_l = (self._levels_np - 1) * (1 - self._eps) / 2
        offset = jnp.where(self._levels_np % 2 == 1, 0.0, 0.5)
        shift = jnp.tan(offset / half_l)
        return jnp.tanh(z + shift) * half_l - offset

    def quantize(self, z: jax.Array) -> jnp.ndarray:
        """Quanitzes z, returns quantized zhat, same shape as z."""
        quantized = round_ste(self.bound(z))

        # Renormalize to [-1, 1].
        half_width = self._levels_np // 2
        return quantized / half_width

    def _scale_and_shift(self, zhat_normalized):
        # Scale and shift to range [0, ..., L-1]
        half_width = self._levels_np // 2
        return (zhat_normalized * half_width) + half_width

    def _scale_and_shift_inverse(self, zhat):
        half_width = self._levels_np // 2
        return (zhat - half_width) / half_width

    def codes_to_indexes(self, zhat: jnp.ndarray) -> jnp.ndarray:
        """Converts a `code` to an index in the codebook."""
        assert zhat.shape[-1] == self.num_dimensions
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis).sum(axis=-1).astype(jnp.uint32)

    def indexes_to_codes(self, indices: jnp.ndarray) -> jnp.ndarray:
        """Inverse of `indexes_to_codes`."""
        indices = indices[..., jnp.newaxis]
        codes_non_centered = jnp.mod(
            jnp.floor_divide(indices, self._basis), self._levels_np
        )
        return self._scale_and_shift_inverse(codes_non_centered)

    def __call__(self, z: jax.Array, is_training: bool = True) -> dict:
        z = self.project_in(z)

        z = einops.rearrange(z, "b (c d) -> b c d", c=self.num_codebooks)

        z = jnp.float32(z)

        code = self.quantize(z)

        quantize = einops.rearrange(code, "b c d -> b (c d)", c=self.num_codebooks)

        quantize = self.project_out(quantize)
        encoding_indices = self.codes_to_indexes(code)
        encodings = self.indexes_to_codes(encoding_indices)

        log(f"quantize shape: {quantize.shape}")

        return dict(
            quantize=quantize,
            encoding_indices=encoding_indices,
            encodings=encodings,
            loss=0.0,
        )


if __name__ == "__main__":
    # fsq = FSQ(levels=[10, 10, 10])

    # z = np.asarray([0.25, 0.6, -7])
    # zhat = fsq.quantize(z)
    # print(f"Quantized {z} -> {zhat}")

    # print(fsq.codebook)
    # print(fsq.codebook.shape)

    import haiku as hk

    @hk.transform_with_state
    def forward_fsq(z):
        init_kwargs = dict(
            w_init=hk.initializers.RandomNormal(stddev=0.01),
            b_init=hk.initializers.Constant(0.0),
        )
        fsq = FSQ(
            levels=[8, 6, 5],
            dim=20,
            init_kwargs=init_kwargs,
            num_codebooks=1,
        )
        return fsq(z)

    z = np.random.randn(3, 20)

    params, state = forward_fsq.init(jax.random.PRNGKey(42), z)

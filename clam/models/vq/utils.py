from functools import partial
from typing import Sequence

import chex
import einops
import flax.linen as nn
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

VQInput = Float[Array, "B ... D"]


@chex.dataclass
class VQOutput:
    quantize: jnp.ndarray
    encoding_indices: jnp.ndarray
    encodings: jnp.ndarray
    loss: jnp.ndarray
    perplexity: jnp.ndarray = None
    distances: jnp.ndarray = None
    num_expired_codes: jnp.ndarray = None
    all_quantized: jnp.ndarray = None  # only for residual VQ


def bounded_uniform(minval, maxval, dtype=jnp.float_) -> nn.initializers.Initializer:
    """Initializer that generates arrays with values uniformly distributed in [minval, maxval).
    Args:
        minval: the lower bound of the random distribution.
        maxval: the upper bound of the random distribution.
        dtype: the dtype of the output array.
    Returns:
        An initializer that generates arrays with values uniformly distributed in [minval, maxval).
    """

    def init(
        key: PRNGKeyArray,
        shape: Sequence[int],
        dtype=dtype,
    ) -> Array:
        dtype = jax.dtypes.canonicalize_dtype(dtype)
        return jax.random.uniform(key, shape, dtype, minval, maxval)

    return init


def pack_one(t, pattern):
    return einops.pack([t], pattern)


def unpack_one(t, ps, pattern):
    return einops.unpack(t, ps, pattern)[0]


def laplace_smoothing(x, n_categories, eps=1e-5, axis=-1):
    """
    Smoothing technique to prevent zero probabilities for categorical distributions.
    Additive smoothing is a technique where you add a small value to the probability of each event.
    """
    denom = jnp.sum(x, axis=axis, keepdims=True)
    return (x + eps) / (denom + n_categories * eps)


def log(t, eps=1e-20):
    return jnp.log(jnp.clip(t, a_min=eps))


def gumbel_noise(key, t):
    noise = jax.random.uniform(key, shape=t.shape, minval=0, maxval=1)
    return -log(-log(noise))


def normalize(x, axis=-1, epsilon=1e-12):
    norm = jnp.linalg.norm(x, ord=2, axis=axis, keepdims=True)
    return x / (norm + epsilon)


def sample_indices_with_permutation(key, num_samples, num):
    permuted_indices = jax.random.permutation(key, num_samples)
    return jax.lax.dynamic_slice(permuted_indices, (0,), (num,))


def sample_indices_with_randint(key, num_samples, num):
    return jax.random.randint(key, minval=0, maxval=num_samples, shape=(num,))


def get_indices(key, num_samples, num):
    # do the same thing for now...
    return jax.lax.cond(
        num_samples >= num,
        lambda _: sample_indices_with_randint(key, num_samples, num),
        lambda _: sample_indices_with_randint(key, num_samples, num),
        operand=None,  # No operand needed here
    )


def sample_vectors(key, samples, num):
    num_samples = samples.shape[0]
    indices = get_indices(key, num_samples, num)
    return samples[indices]


def jax_unstack(x, axis=0):
    return [
        jax.lax.index_in_dim(x, i, axis, keepdims=False) for i in range(x.shape[axis])
    ]


@partial(jax.jit, static_argnames=("num"))
def batched_sample_vectors(key, samples, num):
    _, *keys = jax.random.split(key, samples.shape[0] + 1)
    return jnp.stack(
        [
            sample_vectors(keys[i], sample, num)
            for i, sample in enumerate(jax_unstack(samples, axis=0))
        ],
        axis=0,
    )


def gumbel_sample(
    key,
    logits,
    temperature=1.0,
    stochastic=False,
    straight_through=False,
    axis=-1,
    is_training=True,
):
    """
    Sample from a Gumbel-Softmax distribution.
    """
    dtype, size = logits.dtype, logits.shape[axis]

    if is_training and stochastic and temperature > 0:
        sampling_logits = (logits / temperature) + gumbel_noise(key, logits)
    else:
        sampling_logits = logits

    ind = sampling_logits.argmax(axis=axis)
    one_hot = jax.nn.one_hot(ind, size).astype(dtype)

    if not straight_through or temperature <= 0.0 or not is_training:
        return ind, one_hot

    π1 = (logits / temperature).softmax(axis=axis)
    one_hot = one_hot + π1 - jax.lax.stop_gradient(π1)

    return ind, one_hot


def ema(old, new, decay):
    return old * decay + new * (1 - decay)


def euclidean_distances(x, y):
    """
    Compute the pairwise Euclidean distances between two sets of vectors.

    dist(x, y) = sqrt(dot(x, x) - 2 * dot(x, y) + dot(y, y))
    """
    x2 = einops.reduce(x**2, "b n d -> b n", reduction="sum")
    y2 = einops.reduce(y**2, "b n d -> b n", reduction="sum")
    xy = einops.einsum(x, y, "b i d, b j d -> b i j") * -2
    sum_ = (
        einops.rearrange(x2, "b i -> b i 1") + xy + einops.rearrange(y2, "b j -> b 1 j")
    )
    sum_ = jnp.clip(sum_, 0.0)
    return jnp.sqrt(sum_)


def uniform_init(key, shape, dtype=jnp.float32):
    # kaiming uniform init
    initializer = jax.nn.initializers.he_uniform()
    return initializer(key, shape, dtype)


def batched_embedding(indices, embeds):
    batch, dim = indices.shape[1], embeds.shape[-1]
    indices = einops.repeat(indices, "h b n -> h b n d", d=dim)
    embeds = einops.repeat(embeds, "h c d -> h b c d", b=batch)
    return jnp.take_along_axis(embeds, indices, axis=2)


def get_codebook_util(indices: jnp.ndarray, num_codes: int):
    # TODO: modify this to handle multiple codebooks

    # need to use size argument to make this jittable
    uniques = jnp.unique(indices, size=indices.shape[0], fill_value=-1)
    num_uniques = jnp.sum(uniques > 0)
    return (num_uniques / num_codes) * 100

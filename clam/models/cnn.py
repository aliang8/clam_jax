from typing import Dict, List, Tuple, Union

import einops
import flax.linen as nn
import jax.numpy as jnp
from jaxtyping import Array, Float
from loguru import logger
from omegaconf import DictConfig

from prompt_dtla.utils.logger import log


class DownsamplingBlock(nn.Module):
    spec: Dict[str, int]
    add_bn: bool
    add_residual: bool
    add_max_pool: bool
    mp_kernel_size: int
    init_kwargs: Dict

    @nn.compact
    def __call__(self, x: Array, is_training: bool = True) -> Array:
        x = nn.Conv(**self.spec, **self.init_kwargs)(x)

        if self.add_bn:
            x = nn.BatchNorm(
                use_running_average=not is_training,
                momentum=0.9,
            )(x)

        if self.add_residual:
            x = ResidualBlock(self.spec["features"], self.init_kwargs)(x)

        if self.add_max_pool:
            x = nn.max_pool(
                x,
                window_shape=(self.mp_kernel_size, self.mp_kernel_size),
                strides=(2, 2),
                padding="SAME",
            )

        x = nn.gelu(x)
        return x


class UpsamplingBlock(nn.Module):
    spec: Dict[str, int]
    add_bn: bool
    add_residual: bool
    init_kwargs: Dict

    @nn.compact
    def __call__(self, x: Array, is_training: bool = True) -> Array:
        x = nn.ConvTranspose(**self.spec, **self.init_kwargs)(x)

        if self.add_bn:
            x = nn.BatchNorm(
                use_running_average=not is_training,
                momentum=0.9,
            )(x)

        if self.add_residual:
            x = ResidualBlock(self.spec["features"], self.init_kwargs)(x)

        x = nn.gelu(x)
        return x


class ResidualBlock(nn.Module):
    """
    flax.conv uses BHWC format, whereas hk.Conv2D uses BCHW format
    """

    num_channels: int
    init_kwargs: Dict

    @nn.compact
    def __call__(self, x):
        main_branch = nn.Sequential(
            [
                nn.gelu,
                nn.Conv(
                    features=self.num_channels // 2,
                    kernel_size=[3, 3],
                    strides=[1, 1],
                    padding="SAME",
                    **self.init_kwargs,
                ),
                nn.gelu,
                nn.Conv(
                    features=self.num_channels,
                    kernel_size=[3, 3],
                    strides=[1, 1],
                    padding="SAME",
                    **self.init_kwargs,
                ),
            ]
        )

        return main_branch(x) + x


class ConvEncoder(nn.Module):
    cfg: DictConfig
    init_kwargs: Dict

    @nn.compact
    def __call__(
        self,
        x: Float[Array, "B H W cxt"],
        is_training: bool = True,
        return_intermediate: bool = False,
    ) -> Union[Float[Array, "B d"], Tuple[Float[Array, "B d"], List[Array]]]:
        log("=" * 50)
        log("inside CNN encoder")
        log(f"x shape: {x.shape}")

        # reshape if there are leading dimensions
        reshape = x.ndim > 4
        if reshape:
            lead_dims = x.shape[:-3]  # last three are h w c
            x = einops.rearrange(x, "... h w c -> (...) h w c")

        log(f"x shape, reshape lead dims: {x.shape}")
        intermediates = []

        for i, spec in enumerate(self.cfg.arch):
            # for the last layer, just apply conv
            if i == len(self.cfg.arch) - 1:
                x = nn.Conv(**spec, **self.init_kwargs)(x)
            else:
                x = DownsamplingBlock(
                    spec,
                    add_bn=self.cfg.add_bn,
                    add_residual=self.cfg.add_residual,
                    add_max_pool=self.cfg.add_max_pool,
                    mp_kernel_size=self.cfg.mp_kernel_size,
                    init_kwargs=self.init_kwargs,
                )(x, is_training=is_training)

            log(f"layer {i}, x shape: {x.shape}")
            intermediates.append(x)

        # flatten and re-embed
        x = jnp.reshape(x, (x.shape[0], -1))
        x = nn.Dense(self.cfg.output_embedding_dim, **self.init_kwargs)(x)

        log(f"x shape, flatten: {x.shape}")

        if reshape:
            # restore leading dimensions
            x = x.reshape(lead_dims + (x.shape[-1],))

        log(f"x shape, final: {x.shape}")

        if not return_intermediate:
            return x
        return x, intermediates


class ConvDecoder(nn.Module):
    cfg: DictConfig
    init_kwargs: Dict

    @nn.compact
    def __call__(
        self,
        x,
        intermediates: List[Array],
        context: Array,
        is_training: bool = True,
    ) -> Array:
        reshape = x.ndim > 4

        if reshape:
            lead_dims = x.shape[:-3]
            x = einops.rearrange(x, "... h w c -> (...) h w c")

        for i, spec in enumerate(self.cfg.arch):
            # combine with intermediate outputs from U-net encoder
            if intermediates is not None:
                x = jnp.concatenate([x, intermediates[-i - 1]], axis=-1)

            x = UpsamplingBlock(
                spec,
                add_bn=self.cfg.add_bn,
                add_residual=self.cfg.add_residual,
                init_kwargs=self.init_kwargs,
            )(x, is_training)

        # last layer
        # this is just to get the right number of channels, doesn't change the spatial dimensions
        # also remember no activation after this
        final_conv_inp = x
        if context is not None:
            final_conv_inp = jnp.concatenate([x, context], axis=-1)

        final_conv_inp = nn.Conv(
            features=self.cfg.arch[-1]["features"],
            kernel_size=[3, 3],
            strides=[1, 1],
            padding="SAME",
            **self.init_kwargs,
        )(final_conv_inp)

        # residual -> activation -> conv
        final_conv_inp = ResidualBlock(
            self.cfg.arch[-1]["features"],
            self.init_kwargs,
        )(final_conv_inp)

        x = nn.Conv(
            features=self.cfg.num_output_channels,
            kernel_size=[1, 1],
            strides=[1, 1],
            padding="SAME",
            **self.init_kwargs,
        )(final_conv_inp)

        if reshape:
            x = x.reshape(lead_dims + (x.shape[-1],))

        return x

from typing import Dict

import flax.linen as nn
import jax
import jax.numpy as jnp
from omegaconf import DictConfig

from prompt_dtla.models.vq.utils import (
    VQInput,
    VQOutput,
    gumbel_sample,
)
from prompt_dtla.models.vq.vq import VectorQuantize
from prompt_dtla.utils.logger import log


class ResidualVQ(nn.Module):
    """
    Based on: https://arxiv.org/pdf/2107.03312
    """

    cfg: DictConfig
    init_kwargs: Dict = None

    def setup(self):
        assert self.cfg.heads == 1, "Residual VQ only supports 1 head"
        self.dim = self.cfg.code_dim

        if self.cfg.codebook_dim is None:
            codebook_dim = self.dim

        codebook_input_dim = codebook_dim * self.cfg.heads
        self.requires_projection = codebook_input_dim != self.dim

        # first project the input to the codebook dim
        if self.requires_projection:
            self.project_in = nn.Sequential(
                [
                    nn.Linear(codebook_input_dim, **self.init_kwargs),
                    nn.gelu,
                ]
            )
            self.project_out = nn.Linear(self.dim, **self.init_kwargs)
        else:
            self.project_in = lambda x: x
            self.project_out = lambda x: x

        self.layers = [
            VectorQuantize(
                config=self.cfg,
                init_kwargs=self.init_kwargs,
            )
            for _ in range(self.cfg.num_quantizers)
        ]

        assert all([not vq.requires_projection for vq in self.layers])

        if not self.cfg.shared_codebook:
            return

        first_vq, *rest_vq = self.layers
        codebook = first_vq._codebook

        for vq in rest_vq:
            # vq._codebook = codebook
            # set the codebook to be the same
            object.__setattr__(vq, "_codebook", codebook)

    def __call__(
        self,
        x: VQInput,
        freeze_codebook: bool = False,
        is_training: bool = True,
    ):
        log(f"in ResidualVQ: x.shape: {x.shape}")
        x = self.project_in(x)

        quantized_out = 0.0
        residual = x

        all_losses = []
        all_indices = []
        all_quantized = []

        perplexity = 0.0

        for quantizer_index, layer in enumerate(self.layers):
            log(
                f"quantizer_index: {quantizer_index}, x.shape: {x.shape}, residual.shape: {residual.shape}"
            )
            vq_output = layer(
                x, freeze_codebook=freeze_codebook, is_training=is_training
            )
            perplexity += vq_output.perplexity
            quantized = vq_output.quantize
            residual = residual - jax.lax.stop_gradient(quantized)
            quantized_out = quantized_out + quantized

            all_indices.append(vq_output.encoding_indices)
            all_losses.append(vq_output.loss)
            all_quantized.append(quantized)

        # project out
        quantized_out = self.project_out(quantized_out)

        all_losses = jnp.stack(all_losses, axis=0)
        all_indices = jnp.stack(all_indices, axis=0)
        all_quantized = jnp.stack(all_quantized, axis=0)

        # compute perplexity just for the codebook in level 1
        # perplexity = get_codebook_util(all_indices[:1], self.cfg.num_codes)

        loss = jnp.sum(all_losses)
        return VQOutput(
            quantize=quantized_out,
            loss=loss,
            perplexity=perplexity,
            encoding_indices=all_indices,
            encodings=None,
            num_expired_codes=None,
            all_quantized=all_quantized,
        )


if __name__ == "__main__":
    logits = jnp.array([[0.1, 0.2, 0.3, 0.4]])
    key = jax.random.PRNGKey(4)
    ind, one_hot = gumbel_sample(key, logits, temperature=1.0, stochastic=True)

    print(ind, one_hot)

    codebook_config = DictConfig(
        dict(
            code_dim=128,
            num_codes=512,
            ema_decay=0.8,
            eps=1e-5,
            ema_update=True,
            num_codebooks=1,
            threshold_ema_dead_code=2.0,
            codebook_dim=None,
            sample_codebook_temp=1.0,
            stochastic_sample_codes=False,
            straight_through=False,
            affine_params=False,
            learnable_codebook=False,
            sync_update_v=0.0,
        )
    )

    vq_config = DictConfig(
        dict(
            code_dim=128,
            codebook_cls="euclidean",
            learnable_codebook=False,
            separate_codebook_per_head=False,
            codebook_diversity_loss_weight=1.0,
            codebook_diversity_temp=1.0,
            beta=1.0,
            ema_update=True,
            codebook_dim=None,
            heads=1,
            codebook_cfg=codebook_config,
            num_quantizers=2,
            shared_codebook=False,
            num_codes=512,
            sync_update_v=0.0,
        )
    )

    vq = ResidualVQ(config=vq_config)
    # randn
    x = jax.random.normal(key, (128, 128))

    params = vq.init(key, x)

    vq_output, state = vq.apply(
        params,
        x,
        rngs={"sample": key, "vq": key},
        mutable=["batch_stats", "vq"],
    )

    print(vq_output.quantize.shape)

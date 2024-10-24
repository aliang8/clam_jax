from typing import Dict

import einops
import flax.linen as nn
import jax
import jax.numpy as jnp
from omegaconf import DictConfig

from prompt_dtla.models.vq.utils import (
    VQInput,
    VQOutput,
    batched_embedding,
    batched_sample_vectors,
    ema,
    euclidean_distances,
    get_codebook_util,
    gumbel_sample,
    laplace_smoothing,
    log,
    normalize,
    pack_one,
    uniform_init,
    unpack_one,
)
from prompt_dtla.utils.logger import log


class EuclideanCodebook(nn.Module):
    cfg: DictConfig
    init_kwargs: Dict = None

    def setup(self):
        num_codebooks = self.cfg.num_codebooks
        num_codes = self.cfg.num_codes
        dim = self.cfg.code_dim

        embeddings = uniform_init(
            self.make_rng("vq"),
            (num_codebooks, num_codes, dim),
        )

        # initialize codebook
        self.embed = self.variable(
            "vq",
            "embed",
            lambda *_: embeddings,
            self.make_rng("vq"),
            (num_codebooks, num_codes, dim),
            jnp.float32,
        )

        self.embed_avg = self.variable(
            "vq",
            "embed_avg",
            lambda *_: embeddings,
            self.make_rng("vq"),
            (num_codebooks, num_codes, dim),
            jnp.float32,
        )

        self.cluster_size = self.variable(
            "vq",
            "cluster_size",
            nn.initializers.ones,
            self.make_rng("vq"),
            (num_codebooks, num_codes),
            jnp.float32,
        )

        log(f"embed shape: {self.embed.value.shape}")
        log(f"embed_avg shape: {self.embed_avg.value.shape}")
        log(f"cluster_size shape: {self.cluster_size.value.shape}")

        if self.cfg.affine_params:
            # following: https://arxiv.org/abs/2305.08842
            # proposes to address issues of codebook collapse via affine reparametrization of code vectors
            self.codebook_mean = self.variable(
                "vq",
                "codebook_mean",
                nn.initializers.zeros,
                self.make_rng("vq"),
                (num_codebooks, 1, dim),
                jnp.float32,
            )
            self.codebook_variance = self.variable(
                "vq",
                "codebook_variance",
                nn.initializers.ones,
                self.make_rng("vq"),
                (num_codebooks, 1, dim),
                jnp.float32,
            )
            self.batch_mean = self.variable(
                "vq",
                "batch_mean",
                nn.initializers.zeros,
                self.make_rng("vq"),
                (num_codebooks, 1, dim),
                jnp.float32,
            )
            self.batch_variance = self.variable(
                "vq",
                "batch_variance",
                nn.initializers.ones,
                self.make_rng("vq"),
                (num_codebooks, 1, dim),
                jnp.float32,
            )

            log("using affine params")
            log(
                f"codebook_mean shape: {self.codebook_mean.value.shape}, codebook_variance shape: {self.codebook_variance.value.shape}"
            )
            log(
                f"batch_mean shape: {self.batch_mean.value.shape}, batch_variance shape: {self.batch_variance.value.shape}"
            )

    def expire_code_(self, batch_samples):
        """
        Replace expired codes with new ones from the current batch.
        """
        if self.cfg.threshold_ema_dead_code == 0:
            return None

        cluster_size = self.cluster_size.value
        expired_codes = cluster_size < self.cfg.threshold_ema_dead_code

        # replace the expired codes
        batch_samples = normalize(batch_samples)

        # squeeze (batch, number of codebooks)
        batch_samples = einops.rearrange(batch_samples, "h ... d -> h (...) d")

        # pick random vectors from batch to replace the codebook
        embed = self.embed.value
        embed_avg = self.embed_avg.value

        # this is iterating over each head
        for ind, (samples, mask) in enumerate(zip(batch_samples, expired_codes)):
            # current samples random indices
            sampled = batched_sample_vectors(
                key=self.make_rng("vq"),
                samples=einops.rearrange(samples, "... -> 1 ..."),
                # num=mask.sum(),
                num=self.cfg.num_codes,  # we need this to be static
            )
            sampled = einops.rearrange(sampled, "1 ... -> ...")

            new_embed = jnp.where(mask[:, None], sampled, embed[ind])
            embed = embed.at[ind].set(new_embed)

            new_embed_avg = jnp.where(
                mask[:, None],
                sampled * self.cfg.threshold_ema_dead_code,
                embed_avg[ind],
            )
            embed_avg = embed_avg.at[ind].set(new_embed_avg)

            new_cluster_size = jnp.where(
                mask,
                self.cfg.threshold_ema_dead_code,
                cluster_size[ind],
            )

            cluster_size = cluster_size.at[ind].set(new_cluster_size)

        if not self.is_initializing():
            self.embed.value = embed
            self.embed_avg.value = embed_avg
            self.cluster_size.value = cluster_size
        return jnp.sum(expired_codes)

    def __call__(
        self,
        x: jnp.ndarray,
        freeze_codebook: bool = False,
        is_training: bool = True,
    ):
        log("=" * 80)
        log("inside EuclideanCodebook")
        needs_head_dim = x.ndim < 4

        x = x.astype(jnp.float32)

        if needs_head_dim:
            x = einops.rearrange(x, "... -> 1 ...")
            log(f"adding head dim, x shape: {x.shape}")

        log(f"number of input dims: {x.ndim}")

        # x shape is (1, b, n, d) where n is the number of codebooks
        dtype = x.dtype
        flatten, ps = pack_one(x, "h * d")

        log(f"flatten dims: {flatten.shape}")

        embed = self.embed.value

        log(f"embed shape: {embed.shape}")

        if not self.cfg.learnable_codebook:
            embed = jax.lax.stop_gradient(embed)

        if self.cfg.affine_params:
            log("updating codebook and batch affine params")
            embed_h = einops.rearrange(embed, "h ... d -> h (...) d")

            codebook_mean = self.codebook_mean.value
            codebook_variance = self.codebook_variance.value
            batch_mean = self.batch_mean.value
            batch_variance = self.batch_variance.value

            if is_training:
                codebook_mean = ema(
                    codebook_mean,
                    einops.reduce(embed_h, "h n d -> h 1 d", "mean"),
                    self.cfg.affine_param_codebook_decay,
                )

                codebook_variance = ema(
                    codebook_variance,
                    einops.reduce(embed_h, "h n d -> h 1 d", jnp.var),
                    self.cfg.affine_param_codebook_decay,
                )

                # prepare batch data
                batch_data = einops.rearrange(flatten, "h ... d -> h (...) d")
                if not self.cfg.sync_affine_param:
                    batch_mean = ema(
                        batch_mean,
                        einops.reduce(batch_data, "h n d -> h 1 d", "mean"),
                        self.cfg.affine_param_batch_decay,
                    )
                    batch_variance = ema(
                        batch_variance,
                        einops.reduce(batch_data, "h n d -> h 1 d", jnp.var),
                        self.cfg.affine_param_batch_decay,
                    )
                else:
                    pass

                if not self.is_initializing():
                    self.codebook_mean.value = codebook_mean
                    self.codebook_variance.value = codebook_variance
                    self.batch_mean.value = batch_mean
                    self.batch_variance.value = batch_variance

            codebook_std = jnp.sqrt(codebook_variance.clip(min=1e-5))
            batch_std = jnp.sqrt(batch_variance.clip(min=1e-5))
            embed = (embed - codebook_mean) * (batch_std / codebook_std) + batch_mean

        # compute pairwise distances between input and codebook
        # higher distance means less similar, smaller logit
        dist = -euclidean_distances(flatten, embed)

        log(f"dist shape: {dist.shape}")

        # sample based on distances
        embed_ind, embed_onehot = gumbel_sample(
            self.make_rng("vq"),
            dist,
            axis=-1,
            temperature=self.cfg.sample_codebook_temp,
            stochastic=self.cfg.stochastic_sample_codes,
            straight_through=self.cfg.straight_through,
            is_training=is_training,
        )

        embed_ind = unpack_one(embed_ind, ps, "h *")

        num_expired_codes = None
        if is_training:
            unpacked_onehot = unpack_one(embed_onehot, ps, "h * c")
            # einops notation
            # c is codebook size
            # d is codebook dimension
            # h is 1 for now, is the number of heads
            # b is batch size
            # n is number of codebooks

            # TODO: be careful there are some numerical rounding things with float32

            # use onehot as selector for codes in codebook
            quantize = einops.einsum(
                unpacked_onehot, embed, "h b n c, h c d -> h b n d"
            )
        else:
            quantize = batched_embedding(embed_ind, embed)

        if is_training and self.cfg.ema_update and not freeze_codebook:
            # EMA UPDATE RULE

            if self.cfg.affine_params:
                flatten = (flatten - batch_mean) * (
                    codebook_std / batch_std
                ) + codebook_mean

            cluster_size_old = self.cluster_size.value

            # cluster size is the number of times a code is used
            cluster_size = embed_onehot.sum(axis=1)

            cluster_size = ema(cluster_size_old, cluster_size, self.cfg.ema_decay)

            if not self.is_initializing():
                self.cluster_size.value = cluster_size

            embed_sum = einops.einsum(flatten, embed_onehot, "h n d, h n c -> h c d")

            embed_avg = self.embed_avg.value
            embed_avg = ema(embed_avg, embed_sum, self.cfg.ema_decay)

            n = jnp.sum(cluster_size, axis=-1, keepdims=True)
            cluster_size = laplace_smoothing(
                x=cluster_size,
                n_categories=self.cfg.num_codes,
                eps=self.cfg.eps,
                axis=-1,
            )
            cluster_size *= n
            embed_norm = embed_avg / einops.rearrange(cluster_size, "... -> ... 1")
            # hk.set_state("cluster_size", cluster_size)

            if not self.is_initializing():
                self.embed_avg.value = embed_avg
                self.embed.value = embed_norm

            # replace codes that are not used
            num_expired_codes = self.expire_code_(x)

        if needs_head_dim:
            # removing codebook dim from output
            quantize = einops.rearrange(quantize, "1 ... -> ...")
            embed_ind = einops.rearrange(embed_ind, "1 ... -> ...")

        dist = unpack_one(dist, ps, "h * d")
        log("=" * 80)
        return quantize, embed_ind, embed_onehot, dist, num_expired_codes


class VectorQuantize(nn.Module):
    cfg: DictConfig
    init_kwargs: Dict = None

    def setup(self):
        assert not (
            self.cfg.ema_update and self.cfg.learnable_codebook
        ), "learnable codebook not compatible with EMA update"

        if self.cfg.separate_codebook_per_head:
            self.cfg.codebook_cfg.num_codebooks = self.cfg.heads
        else:
            self.cfg.codebook_cfg.num_codebooks = 1

        if self.cfg.codebook_dim is None:
            codebook_dim = self.cfg.code_dim
        else:
            codebook_dim = self.cfg.codebook_dim

        self.cfg.codebook_cfg.code_dim = codebook_dim
        codebook_input_dim = codebook_dim * self.cfg.heads
        log(f"codebook input dim: {codebook_input_dim}")
        self.requires_projection = codebook_input_dim != self.cfg.code_dim

        # first project the input to the codebook dim
        if self.requires_projection:
            log(f"projecting input to codebook dim: {codebook_input_dim}")
            self.project_in = nn.Sequential(
                [
                    nn.Dense(codebook_input_dim, **self.init_kwargs),
                    jax.nn.gelu,
                ]
            )
            self.project_out = nn.Dense(self.cfg.code_dim, **self.init_kwargs)
        else:
            self.project_in = lambda x: x
            self.project_out = lambda x: x

        if self.cfg.codebook_cls == "euclidean":
            codebook_class = EuclideanCodebook
        else:
            raise NotImplementedError(
                f"codebook class {self.cfg.codebook_cls} not implemented"
            )

        self._codebook = codebook_class(self.cfg.codebook_cfg, self.init_kwargs)

    def __call__(
        self,
        x: VQInput,
        freeze_codebook: bool = False,
        is_training: bool = True,
    ):
        log("*" * 80)
        log("inside VQ")

        orig_input = x
        is_multiheaded = self.cfg.heads > 1
        log(f"number of input dims: {x.ndim}")

        only_one = x.ndim == 2

        # add dimension for number of codebook
        if only_one:
            x = einops.rearrange(x, "b d -> b 1 d")

        log(f"only_one: {only_one}")
        x = self.project_in(x)
        log(f"after projection: {x.shape}")

        if is_multiheaded:
            ein_rhs_eq = (
                "h b n d" if self.cfg.separate_codebook_per_head else "1 (b h) n d"
            )
            x = einops.rearrange(x, f"b n (h d) -> {ein_rhs_eq}", h=self.cfg.heads)

        # quantize
        quantize, embed_ind, embed_onehot, distances, num_expired_codes = (
            self._codebook(x, is_training=is_training)
        )

        if is_training:
            if not self.cfg.learnable_codebook or freeze_codebook:
                log("codebook is not learnable, stopping gradients")
                commit_quantize = jax.lax.stop_gradient(quantize)
            else:
                commit_quantize = quantize

            # straight through
            quantize = x + jax.lax.stop_gradient(quantize - x)

            if self.cfg.sync_update_v > 0.0:
                # (21) in https://minyoungg.github.io/vqtorch/assets/draft_050523.pdf
                # ν is a scalar to decide whether we want a pessimistic
                # or an optimistic update
                quantize = quantize + self.cfg.sync_update_v * (
                    quantize - quantize.detach()
                )

        # if return_loss:
        #     return quantize

        loss = 0.0
        commit_loss = 0.0
        codebook_diversity_loss = 0.0

        if is_multiheaded:
            if self.cfg.separate_codebook_per_head:
                embed_ind = einops.rearrange(
                    embed_ind, "h b n -> b n h", h=self.cfg.heads
                )
            else:
                embed_ind = einops.rearrange(
                    embed_ind, "1 (b h) n -> b n h", h=self.cfg.heads
                )

        if only_one:
            embed_ind = einops.rearrange(embed_ind, "b 1 ... -> b ...")

        if is_training:
            # diversity loss
            if self.cfg.codebook_diversity_loss_weight > 0.0:
                log("diversity loss")
                # TODO: debug this
                prob = -distances * self.cfg.codebook_diversity_temp
                prob = jax.nn.softmax(prob, axis=-1)

                avg_prob = einops.reduce(prob, "... n l -> n l", reduction="mean")
                codebook_diversity_loss = -(avg_prob * log(avg_prob)).sum()
                loss = (
                    loss
                    + self.cfg.codebook_diversity_loss_weight * codebook_diversity_loss
                )

            # commitment loss
            if self.cfg.beta > 0.0:
                log("commitment loss")
                commit_loss = jnp.mean((x - commit_quantize) ** 2)
                loss = loss + self.cfg.beta * commit_loss

        if is_multiheaded:
            if self.cfg.separate_codebook_per_head:
                quantize = einops.rearrange(
                    quantize, "h b n d -> b n (h d)", h=self.cfg.heads
                )
            else:
                quantize = einops.rearrange(
                    quantize, "1 (b h) n d -> b n (h d)", h=self.cfg.heads
                )

        # project out
        quantize = self.project_out(quantize)

        if only_one:
            quantize = einops.rearrange(quantize, "b 1 d -> b d")

        # loss_breakdown = {
        #     "commitment_loss": commit_loss,
        #     "codebook_diversity_loss": codebook_diversity_loss,
        # }

        log(f"quantize shape: {quantize.shape}")
        perplexity = get_codebook_util(embed_ind, self.cfg.num_codes * self.cfg.heads)

        log(f"embed_onehot shape: {embed_onehot.shape}")
        log(f"embed ind shape: {embed_ind.shape}")

        # compute exp(entropy) = perplexity which measures codebook utilization
        avg_probs = jnp.mean(embed_onehot, axis=1)
        log(f"avg probs shape: {avg_probs.shape}")

        perplexity = jnp.exp(-jnp.sum(avg_probs * jnp.log(avg_probs + 1e-10), axis=-1))
        # averaging over the number of heads (TODO: check this)
        perplexity = jnp.mean(perplexity)
        return VQOutput(
            quantize=quantize,
            loss=loss,
            perplexity=perplexity,
            encoding_indices=embed_ind,
            encodings=None,
            num_expired_codes=num_expired_codes,
        )


if __name__ == "__main__":
    # test function
    X = jnp.array([[0, 1], [0, 2]])
    X = X.reshape(1, 2, 2)
    Y = jnp.array([[0, 1], [0, 2]])
    Y = Y.reshape(1, 2, 2)
    print(euclidean_distances(X, Y))
    dist = -euclidean_distances(X, Y)

    print(f"dist shape: {dist.shape}")

    # logits = jnp.array([[0.1, 0.2, 0.3, 0.4]])
    key = jax.random.PRNGKey(5)
    ind, one_hot = gumbel_sample(key, dist, temperature=1.0, stochastic=False)

    print(ind.shape, one_hot.shape)
    print(ind, one_hot)

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

    init_kwargs = dict(
        kernel_init=nn.initializers.xavier_uniform(),
    )

    vq = VectorQuantize(vq_config, init_kwargs=init_kwargs)
    # randn
    x = jax.random.normal(key, (64, 64))

    # jit_init = jax.jit(vq.init)
    # jit_apply = jax.jit(vq.apply)
    params = vq.init(key, x)

    vq_output, _ = vq.apply(params, x, rngs={"vq": key}, mutable=["batch_stats", "vq"])

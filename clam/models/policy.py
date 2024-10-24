from typing import Dict, Optional

import distrax
import flax
import flax.linen as nn
import jax.numpy as jnp
from loguru import logger
from omegaconf import DictConfig

from prompt_dtla.models.cnn import ConvEncoder
from prompt_dtla.models.mlp import MLP
from prompt_dtla.utils.logger import log


@flax.struct.dataclass
class PolicyOutput:
    action: jnp.ndarray
    entropy: Optional[jnp.ndarray] = None
    dist: Optional[distrax.Distribution] = None
    log_prob: Optional[jnp.ndarray] = None
    logits: Optional[jnp.ndarray] = None
    latent_action: Optional[jnp.ndarray] = None


class ActionHead(nn.Module):
    """
    Input: input embeddings
    Output: PolicyOutput.{action, entropy, dist, log_prob, logits}

    Works on both continuous and discrete action spaces.
    """

    is_continuous: bool
    gaussian_policy: bool
    action_dim: int
    init_kwargs: Dict

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        is_training: bool = True,
        rng=None,
    ) -> PolicyOutput:
        if self.is_continuous:
            mean = nn.Dense(self.action_dim, **self.init_kwargs)(x)
            logvar = nn.Dense(self.action_dim, **self.init_kwargs)(x)

            if self.gaussian_policy:
                # clamp logvar
                logvar = jnp.clip(logvar, -5.0, 5.0)

                action_dist = distrax.MultivariateNormalDiag(
                    loc=mean, scale_diag=jnp.exp(0.5 * logvar)
                )
                action = action_dist.sample(
                    seed=self.make_rng("sample") if rng is None else rng
                )
                logits = jnp.stack([mean, logvar], axis=-1)
                entropy = action_dist.entropy()
                log_prob = action_dist.log_prob(action)
            else:
                action_dist = None
                action = mean
                logits = None
                entropy = None
                log_prob = None

            # apply tanh to action and rescale to action limits (assume [-1, 1])
            action = jnp.tanh(action)
        else:
            logits = nn.Dense(self.action_dim, **self.init_kwargs)(x)
            action_dist = distrax.Categorical(logits=logits)
            if is_training:
                action = action_dist.sample(
                    seed=self.make_rng("sample") if rng is None else rng
                )
            else:
                action = action_dist.mode()
            entropy = action_dist.entropy()
            log_prob = action_dist.log_prob(action)

        return PolicyOutput(
            action=action,
            entropy=entropy,
            dist=action_dist,
            logits=logits,
            log_prob=log_prob,
        )


class Policy(nn.Module):
    """
    Encoder -> MLP -> ActionHead
    Encoder can handle both image and state observations.
    ActionHead can handle both continuous (Gaussian sample or just mean)
        and discrete action spaces.
    rng: only used in ActionHead for Gaussian policy sampling
    """

    cfg: DictConfig
    is_continuous: bool
    action_dim: int
    init_kwargs: Dict

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, is_training: bool = True, rng=None
    ) -> PolicyOutput:
        log("=" * 50)
        log("inside MLP Policy")
        log(f"state shape: {x.shape}")

        if self.cfg.image_obs:
            x = ConvEncoder(self.cfg.encoder_cfg, self.init_kwargs)(
                x, is_training=is_training
            )
        else:
            x = nn.Dense(self.cfg.embedding_dim, **self.init_kwargs)(x)
        x = nn.gelu(x)
        log(f"embed shape: {x.shape}")

        x = MLP(
            list(self.cfg.mlp_layer_sizes),
            activation=nn.gelu,
            activate_final=True,
            init_kwargs=self.init_kwargs,
        )(x)

        policy_output = ActionHead(
            gaussian_policy=self.cfg.gaussian_policy,
            is_continuous=self.is_continuous,
            action_dim=self.action_dim,
            init_kwargs=self.init_kwargs,
        )(x, is_training=is_training, rng=rng)

        return policy_output

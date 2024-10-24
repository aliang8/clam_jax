from typing import Dict

import einops
import flax.linen as nn
import jax
import jax.numpy as jnp
from loguru import logger
from omegaconf import DictConfig

from prompt_dtla.models.cnn import ConvEncoder
from prompt_dtla.models.transformer import TransformerEncoder
from prompt_dtla.utils.logger import log


class SAREncoder(nn.Module):
    image_obs: bool
    embedding_dim: int
    model_cfg: DictConfig
    encoder_cfg: DictConfig
    encode_separate: bool
    batch_first: bool

    task_conditioning: bool = False

    max_timesteps: int = 1000
    use_rtg: bool = False
    init_kwargs: Dict = None

    def setup(self):
        self.model = TransformerEncoder(**self.model_cfg, init_kwargs=self.init_kwargs)

        if self.image_obs:
            # first encode the input images with a cnn
            self.state_embed = ConvEncoder(self.encoder_cfg, self.init_kwargs)
        else:
            self.state_embed = nn.Dense(self.embedding_dim, **self.init_kwargs)

        self.action_embed = nn.Dense(self.embedding_dim, **self.init_kwargs)
        self.reward_embed = nn.Dense(self.embedding_dim, **self.init_kwargs)
        self.timestep_embed = nn.Embed(5000, self.embedding_dim)

        self.prompt_embed = nn.Dense(self.embedding_dim, **self.init_kwargs)
        self.traj_index_embed = nn.Embed(5, self.embedding_dim)

        self.embed = nn.Dense(self.embedding_dim, **self.init_kwargs)

    def __call__(
        self,
        states: jnp.ndarray,  # [T, B, D]
        actions: jnp.ndarray,  # [T, B, D]
        rewards: jnp.ndarray = None,  # [T, B, 1]
        mask: jnp.ndarray = None,  # [T, B]
        prompt: jnp.ndarray = None,
        timestep: jnp.ndarray = None,  # [T, B]
        traj_index: jnp.ndarray = None,
        is_training: bool = True,
        **kwargs,
    ) -> jnp.ndarray:
        """
        Implement causal transformer similar to the Decision Transformer paper.
        Each of state, action, and reward are either embedded
        as separate tokens or treated as a single timestep.

        Reward can be optional. If not provided, then we will just have (s, a) sequences.

        Return:
            embeddings: [3*T, B, D] tensor of embeddings
        """

        log("=" * 50)
        log("inside SAR encoder")

        # convert to batch first
        def batch_first(x):
            if x is not None:
                return einops.rearrange(x, "t b ... -> b t ...")

        if self.batch_first:
            B, T, *_ = states.shape
        else:
            T, B, *_ = states.shape
            states, actions, rewards, mask = map(
                batch_first, [states, actions, rewards, mask]
            )

        if mask is None:
            mask = jnp.ones((B, T))

        log(f"states shape: {states.shape}, actions shape: {actions.shape}")
        if self.use_rtg and rewards is not None:
            log(f"rewards shape: {rewards.shape}")
        if mask is not None:
            log(f"mask shape: {mask.shape}")

        if self.image_obs:
            log("embedding image observations")
            state_embed = self.state_embed(states, is_training=is_training)
        else:
            state_embed = self.state_embed(states)
        action_embed = self.action_embed(actions)

        if prompt is not None:
            prompt_embed = self.prompt_embed(prompt)

        if timestep is None:
            timestep = jnp.arange(T)
            timestep = einops.repeat(timestep, "t -> b t", b=B)

        timestep_embed = self.timestep_embed(timestep.astype(jnp.int32))

        if self.use_rtg and rewards is not None:
            reward_embed = self.reward_embed(rewards)
            reward_embed = reward_embed + timestep_embed

        # jax.debug.breakpoint()

        state_embed = state_embed + timestep_embed  # [B, T, D]
        action_embed = action_embed + timestep_embed

        # for ICL training, we add a trajectory index embedding
        if traj_index is not None:
            traj_index_embed = self.traj_index_embed(traj_index.astype(jnp.int32))
            state_embed = state_embed + traj_index_embed
            action_embed = action_embed + traj_index_embed

        if self.encode_separate:
            # stack all the tokens together, [B, T, D] -> [B, 3, T, D]
            # reward should be RTG if available

            if not self.use_rtg:
                embeddings = jnp.stack([state_embed, action_embed], axis=1)
            else:
                embeddings = jnp.stack(
                    [reward_embed, state_embed, action_embed], axis=1
                )

            # make one long sequence
            embeddings = einops.rearrange(embeddings, "b c t d -> b (t c) d")

            # if we have a prompt, add it to the beginning
            if prompt is not None:
                embeddings = jnp.concatenate([prompt_embed, embeddings], axis=1)

            # make mask
            num_tokens = 3 if self.use_rtg else 2
            mask = einops.repeat(mask, "b t -> b c t", c=num_tokens)
            mask = einops.rearrange(mask, "b c t -> b (t c)")

            # add mask for the prompt
            if prompt is not None:
                mask = jnp.concatenate([jnp.ones((B, 1)), mask], axis=1)
        else:
            embeddings = jnp.concatenate(
                [state_embed, action_embed, reward_embed], axis=-1
            )
            embeddings = nn.gelu(embeddings)
            embeddings = self.embed(embeddings)

        embeddings = nn.gelu(embeddings)
        embeddings = self.model(embeddings, mask=mask, is_training=is_training)

        if not self.batch_first:
            embeddings = einops.rearrange(embeddings, "b t d -> t b d")

        return embeddings


if __name__ == "__main__":
    # test encoder
    B, T, D = 2, 5, 64
    hidden_dim = 64

    transformer_cfg = DictConfig(
        {
            "num_heads": 4,
            "num_layers": 2,
            "hidden_dim": 64,
            "attn_size": 32,
            "dropout_rate": 0.1,
        }
    )

    sar_encoder = SAREncoder(
        image_obs=True,
        embedding_dim=hidden_dim,
        model_cfg=transformer_cfg,
        batch_first=True,
        encode_separate=True,
        encoder_cfg=DictConfig(
            {
                "output_embedding_dim": 64,
                "arch": [
                    {
                        "features": 32,
                        "kernel_size": [3, 3],
                        "strides": 2,
                        "padding": "SAME",
                    },
                    {
                        "features": 32,
                        "kernel_size": [3, 3],
                        "strides": 2,
                        "padding": "SAME",
                    },
                ],
                "add_bn": True,
                "add_residual": True,
                "add_max_pool": True,
                "mp_kernel_size": 2,
            }
        ),
        init_kwargs={"kernel_init": nn.initializers.xavier_uniform()},
    )

    states = jnp.ones((B, T, 64, 64, 3))
    actions = jnp.ones((B, T, D))
    rewards = jnp.ones((B, T, 1))
    mask = jnp.ones((B, T))

    params = sar_encoder.init(jax.random.PRNGKey(0), states, actions, rewards, mask)

    keys = {
        "params": jax.random.key(0),
        "other": jax.random.key(0),
    }
    log(f"mutable params: {list(params['batch_stats'].keys())}")
    out, mparams = sar_encoder.apply(
        params,
        states=states,
        actions=actions,
        rewards=rewards,
        mask=mask,
        mutable=["batch_stats"],
        rngs=keys,
    )
    log(f"output shape: {out.shape}")

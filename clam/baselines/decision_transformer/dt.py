from typing import Dict, Tuple

import einops
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Key
from omegaconf import DictConfig

from prompt_dtla.models.base import Base
from prompt_dtla.models.policy import Policy
from prompt_dtla.models.sar_encoder import SAREncoder
from prompt_dtla.utils.data_utils import BTX, Batch, PRNGKeyDict
from prompt_dtla.utils.training import TrainState, get_AdamW_optimizer


class DecisionTransformer(nn.Module):
    cfg: DictConfig
    init_kwargs: Dict
    continuous_actions: bool
    action_dim: int

    def setup(self):
        self.model = SAREncoder(**self.cfg.model, init_kwargs=self.init_kwargs)

        self.action_head = Policy(
            self.cfg.policy,
            is_continuous=self.continuous_actions,
            action_dim=self.action_dim,
            init_kwargs=self.init_kwargs,
        )

    def __call__(
        self,
        states: BTX,
        actions: BTX,
        rewards: BTX = None,
        mask: BTX = None,
        prompt: BTX = None,
        timestep: BTX = None,
        traj_index: BTX = None,
        is_training: bool = True,
    ):
        embeddings = self.model(
            states=states,
            actions=actions,
            rewards=rewards,
            mask=mask,
            prompt=prompt,
            timestep=timestep,
            traj_index=traj_index,
            is_training=is_training,
        )

        if prompt is not None:
            # the first token is the prompt
            prompt_embedding = embeddings[:, 0]
            embeddings = embeddings[:, 1:]

        num_tokens = 3 if self.cfg.model.use_rtg else 2
        embeddings = einops.rearrange(embeddings, "b (t c) d -> b c t d", c=num_tokens)

        # predict actions from the state embedding
        if not self.cfg.model.use_rtg:  # SA
            embeddings = embeddings[:, 0]
        else:  # RSA
            embeddings = embeddings[:, 1]

        action_output = self.action_head(embeddings, is_training=is_training)
        return action_output


class DecisionTransformerAgent(Base):
    def _init_model(self, keys: PRNGKeyDict) -> Tuple[dict, nn.Module]:
        B, T = 2, 2
        dummy_x = jnp.zeros((B, T, *self.observation_shape), dtype=jnp.float32)
        dummy_actions = jnp.zeros((B, T, self.input_action_dim))
        dummy_timesteps = jnp.zeros((B, T))

        if self.cfg.model.use_rtg:
            dummy_rewards = jnp.zeros((B, T, 1))
        else:
            dummy_rewards = None

        dummy_mask = jnp.ones((B, T))

        if self.cfg.model.task_conditioning:
            dummy_prompt = jnp.zeros((B, 1, self.task_dim))
        else:
            dummy_prompt = None

        dummy_traj_index = jnp.zeros((B, T))

        model_def = DecisionTransformer(
            self.cfg,
            continuous_actions=self.continuous_actions,
            action_dim=self.action_dim,
            init_kwargs=self.init_kwargs,
        )

        params = model_def.init(
            keys,
            states=dummy_x,
            actions=dummy_actions,
            rewards=dummy_rewards,
            mask=dummy_mask,
            prompt=dummy_prompt,
            timestep=dummy_timesteps,
            traj_index=dummy_traj_index,
            is_training=True,
        )
        return params, model_def

    def create_train_state(self, key: Key) -> TrainState:
        key, sample_key, dropout_key = jax.random.split(key, 3)
        keys = {"params": key, "sample": sample_key, "dropout": dropout_key}

        variables, model_def = self._init_model(keys)
        tx = get_AdamW_optimizer(self.cfg)

        ts = TrainState.create(
            apply_fn=model_def.apply,
            params=variables["params"],
            tx=tx,
            mparams={
                "batch_stats": variables["batch_stats"],
            },
            keys=keys,
        )

        return ts

    def update(
        self,
        key: Key,
        ts: TrainState,
        batch: Batch,
        is_training: bool,
        **kwargs: Dict,
    ) -> Tuple[TrainState, Dict, Dict]:
        key, sample_key, dropout_key = jax.random.split(key, 3)

        def loss_fn(params, batch: Batch, is_training: bool = True):
            observations = batch.observations
            actions = batch.actions
            rewards = batch.rewards

            if self.cfg.model.task_conditioning:
                prompt = batch.tasks[:, 0:1]
            else:
                prompt = None

            # [B, T, 1]
            if len(actions.shape) == 2:
                actions = jnp.expand_dims(actions, axis=-1)

            # [B, T]
            mask = jnp.ones_like(rewards)

            # [B, T, 1]
            if self.cfg.model.use_rtg:
                if len(rewards.shape) == 2:
                    rewards = jnp.expand_dims(rewards, axis=-1)
            else:
                rewards = None

            # predict action
            policy_output = ts.apply_fn(
                {"params": params, **ts.mparams},
                states=observations.astype(jnp.float32),
                actions=actions,
                rewards=rewards,
                mask=mask,
                prompt=prompt,
                timestep=batch.timestep,
                traj_index=batch.traj_index,
                is_training=is_training,
                mutable=list(ts.mparams.keys()) if is_training else False,
                rngs={
                    "sample": sample_key,
                    "dropout": dropout_key,
                },
            )

            if is_training:
                policy_output, mparams = policy_output
            else:
                policy_output, mparams = policy_output, None

            if policy_output.entropy is not None:
                entropy = policy_output.entropy
                entropy = jnp.mean(entropy)
            else:
                entropy = 0.0

            if self.continuous_actions:
                action_preds = policy_output.action
            else:
                action_preds = policy_output.logits

            if self.continuous_actions:
                # compute MSE loss
                loss = optax.squared_error(action_preds, actions.squeeze())
                acc = 0.0
                loss *= batch.mask[..., jnp.newaxis]
                loss = loss.sum() / batch.mask.sum()
            else:
                # compute cross entropy with logits
                loss = optax.softmax_cross_entropy_with_integer_labels(
                    action_preds, actions.squeeze(axis=-1).astype(jnp.int32)
                )
                loss *= batch.mask
                loss = loss.sum() / batch.mask.sum()

                acc = policy_output.action == actions.squeeze(axis=-1)
                acc *= batch.mask
                acc = acc.sum() / batch.mask.sum()

            metrics = {"bc_loss": loss, "entropy": entropy, "decoded_acc": acc}

            return loss, (metrics, {}, mparams)

        value_grad_fn = jax.value_and_grad(
            lambda params: loss_fn(params, batch, is_training=is_training),
            has_aux=True,
        )
        (loss, (metrics, extra, mparams)), grads = value_grad_fn(ts.params)
        ts = ts.apply_gradients(grads=grads)
        ts = ts.replace(mparams=mparams)

        return ts, metrics, extra

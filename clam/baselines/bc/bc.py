from typing import Dict, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Key
from loguru import logger

from prompt_dtla.models.base import Base
from prompt_dtla.models.policy import Policy
from prompt_dtla.utils.data_utils import Batch, PRNGKeyDict
from prompt_dtla.utils.logger import log
from prompt_dtla.utils.training import (
    TrainState,
    default_weight_init,
    get_AdamW_optimizer,
)


class BCAgent(Base):
    weight_init_kwargs = default_weight_init

    def _init_model(self, keys: PRNGKeyDict) -> Tuple[dict, nn.Module]:
        T = 2
        B = 2

        dummy_x = jnp.zeros((B, T, *self.observation_shape), dtype=jnp.float32)
        model_def = Policy(
            self.cfg.policy,
            is_continuous=self.continuous_actions,
            action_dim=self.action_dim,
            init_kwargs=default_weight_init,
        )

        params = model_def.init(keys, dummy_x, is_training=True)

        return params, model_def

    def create_train_state(self, key: Key) -> TrainState:
        params_key, sample_key = jax.random.split(key, 2)
        keys = {"params": params_key, "sample": sample_key}

        variables, model_def = self._init_model(keys)
        tx = get_AdamW_optimizer(self.cfg)

        mparams = {}
        if "batch_stats" in variables:
            mparams["batch_stats"] = variables["batch_stats"]

        ts = TrainState.create(
            apply_fn=model_def.apply,
            params=variables["params"],
            tx=tx,
            mparams=mparams,
            keys={"sample": sample_key},
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
        key, sample_key = jax.random.split(key, 2)

        def loss_fn(params, batch: Batch, is_training: bool = True):
            log(f"bc loss function, observations: {batch.observations.shape}")

            # predict action
            apply_out = ts.apply_fn(
                {"params": params, **ts.mparams},
                x=batch.observations,
                is_training=is_training,
                mutable=list(ts.mparams.keys()) if is_training else False,
                rngs={"sample": sample_key},
            )

            if is_training:
                action_output, mparams = apply_out
            else:
                action_output, mparams = apply_out, None

            gt_actions = batch.actions

            if self.continuous_actions:
                if self.cfg.policy.gaussian_policy:
                    dist = action_output.dist
                    loss = -dist.log_prob(gt_actions).mean()
                else:
                    loss = optax.squared_error(action_output.action, gt_actions)
            else:
                loss = optax.softmax_cross_entropy_with_integer_labels(
                    action_output.logits, gt_actions.squeeze().astype(jnp.int32)
                )
                acc = action_output.action == gt_actions.squeeze()
                acc = jnp.mean(acc)

            loss = jnp.mean(loss)
            metrics = {"bc_loss": loss}

            if not self.continuous_actions:
                metrics["acc"] = acc

            if action_output.entropy is not None:
                metrics["entropy"] = action_output.entropy.mean()

            return loss, (metrics, {}, mparams)

        value_grad_fn = jax.value_and_grad(
            lambda params: loss_fn(params, batch, is_training=is_training),
            has_aux=True,
        )
        (loss, (metrics, extra, mparams)), grads = value_grad_fn(ts.params)
        ts = ts.apply_gradients(grads=grads)
        ts = ts.replace(mparams=mparams)
        return ts, metrics, extra

from typing import Dict, Tuple

import einops
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Key
from omegaconf import DictConfig

from prompt_dtla.models.base import Base
from prompt_dtla.models.mlp import MLP
from prompt_dtla.models.policy import ActionHead, PolicyOutput
from prompt_dtla.utils.data_utils import Batch, PRNGKeyDict
from prompt_dtla.utils.training import (
    TrainState,
    default_weight_init,
    get_AdamW_optimizer,
)


class LatentActionDecoder(Base):
    gaussian_policy: bool = False

    def _init_model(self, keys: PRNGKeyDict) -> Tuple[dict, nn.Module]:
        B = 2
        dummy_x = jnp.zeros((B, self.input_action_dim), dtype=jnp.float32)

        model_def = ActionDecoderNetwork(
            self.cfg,
            action_dim=self.action_dim,
            init_kwargs=default_weight_init,
            is_continuous=self.continuous_actions,
            gaussian_policy=self.gaussian_policy,
        )

        if self.load_from_ckpt:
            ts = self.load_model_from_ckpt()

            if "decoder" in ts["params"]:
                params = ts["params"]["decoder"]
            else:
                params = ts["params"]["params"]

        else:
            params = model_def.init(keys, x=dummy_x, is_training=True)

        return params, model_def

    def create_train_state(self, key: Key) -> TrainState:
        params_key, sample_key = jax.random.split(key, 2)
        keys = {"params": params_key, "sample": sample_key}

        params, model_def = self._init_model(keys)
        tx = get_AdamW_optimizer(self.cfg)

        ts = TrainState.create(
            apply_fn=model_def.apply,
            params=params,
            tx=tx,
            mparams=None,
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

        def loss_fn(params, batch: Batch, is_training: bool):
            latent_actions = batch.latent_actions
            gt_actions = batch.actions

            output = ts.apply_fn(
                params,
                x=latent_actions,
                is_training=is_training,
                rngs={"sample": sample_key},
            )

            if gt_actions.shape[-1] == 1:
                gt_actions = gt_actions.squeeze(axis=-1)

            if self.continuous_actions:
                if self.gaussian_policy:
                    dist = output.dist
                    loss = -dist.log_prob(gt_actions)
                else:
                    loss = optax.squared_error(output.action, gt_actions)
            else:
                # discrete actions
                loss = optax.softmax_cross_entropy_with_integer_labels(
                    output.logits, gt_actions.astype(jnp.int32)
                )

            loss = jnp.mean(loss)
            metrics = {"action_loss": loss}

            if not self.continuous_actions:
                # for discrete actions
                acc = output.action == gt_actions
                acc = jnp.mean(acc)

            extras = None
            mparams = None
            return loss, (metrics, extras, mparams)

        value_grad_fn = jax.value_and_grad(
            lambda params: loss_fn(params, batch, is_training),
            has_aux=True,
        )

        (loss, (metrics, _, _)), grad = value_grad_fn(ts.params)

        ts = ts.apply_gradients(grads=grad)
        return ts, metrics, {}


class ActionDecoderNetwork(nn.Module):
    cfg: DictConfig
    action_dim: int
    init_kwargs: Dict
    is_continuous: bool
    gaussian_policy: bool = False

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        is_training: bool = True,
        rng=None,
    ) -> PolicyOutput:
        x = einops.rearrange(x, "b ... -> b (...)")

        # for LAMDecoder joint training, versus decoder only training
        if "la_decoder" in self.cfg:
            hidden_dims = self.cfg.la_decoder.mlp_layer_sizes
        else:
            hidden_dims = self.cfg.mlp_layer_sizes

        x = MLP(
            hidden_dims=hidden_dims,
            init_kwargs=self.init_kwargs,
            activation=nn.gelu,
            activate_final=True,
        )(x)

        output = ActionHead(
            is_continuous=self.is_continuous,
            gaussian_policy=self.gaussian_policy,
            action_dim=self.action_dim,
            init_kwargs=self.init_kwargs,
        )(x, is_training=is_training, rng=rng)

        return output

from typing import Dict, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxtyping import Key

from prompt_dtla.baselines.vpt.model import IDM
from prompt_dtla.models.base import Base
from prompt_dtla.utils.data_utils import Batch, PRNGKeyDict
from prompt_dtla.utils.logger import log
from prompt_dtla.utils.training import (
    TrainState,
    default_weight_init,
    get_AdamW_optimizer,
)


class VPT(Base):
    """
    Video Pretraining Model. An Inverse Dynamics Model that predicts the ground truth action
    between two consecutive observations. VPT can only be learned on labelled action dataset.
    """

    weight_init_kwargs = default_weight_init

    # @partial(jax.pmap, axis_name="device", static_broadcasted_argnums=(0,))
    def _init_model(self, keys: PRNGKeyDict) -> Tuple[dict, nn.Module]:
        T, B = self.cfg.seq_len, 2
        dummy_x = np.zeros((B, T, *self.observation_shape), dtype=np.float32)
        dummy_ts = np.zeros((B, T), dtype=np.float32)

        model_def = IDM(
            self.cfg.idm,
            is_continuous=self.continuous_actions,
            action_dim=self.action_dim,
            init_kwargs=self.weight_init_kwargs,
        )

        if self.load_from_ckpt:
            ts = self.load_model_from_ckpt()
            params = {"params": ts["params"], **ts["mparams"]}
        else:
            params = model_def.init(
                keys,
                x=dummy_x,
                is_training=True,
            )
        return params, model_def

    def create_train_state(self, key: Key) -> TrainState:
        # todo is this the right way to split for vq?
        key, sample_key, dropout_key, model_key = jax.random.split(key, 4)
        keys = {
            "params": key,
            "sample": sample_key,
            "dropout": dropout_key,
            "model": model_key,
        }

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
        key, sample_key, dropout_key, model_key = jax.random.split(key, 4)

        def loss_fn(params, batch: Batch, is_training: bool = True):
            log(f"vpt loss function, observations: {batch.observations.shape}")

            apply_out = ts.apply_fn(
                {"params": params, **ts.mparams},
                x=batch.observations.astype(jnp.float32),
                # timestep=batch.timestep.astype(jnp.float32),
                is_training=is_training,
                mutable=list(ts.mparams.keys()) if is_training else False,
                rngs={"sample": sample_key},
            )

            if is_training:
                action_pred, mparams = apply_out
            else:
                action_pred, mparams = apply_out, None

            if self.cfg.idm.encoder_cfg.name in ["vit", "transformer"]:
                gt_actions = batch.actions[:, :-1]
            else:
                gt_actions = batch.actions[:, -2]

            # action prediction, if discrete use cross entropy, if continuous use squared error
            if self.continuous_actions:
                loss = optax.squared_error(action_pred.action, gt_actions)
            else:
                loss = optax.softmax_cross_entropy_with_integer_labels(
                    action_pred.logits.squeeze(),
                    gt_actions.squeeze().astype(jnp.int32),
                )

            loss = jnp.mean(loss)
            metrics = {"action_pred_loss": loss}

            if not self.continuous_actions:
                # for discrete actions
                acc = action_pred.action.squeeze() == gt_actions.squeeze()
                acc = jnp.mean(acc)

            return loss, (metrics, {}, mparams)

        # todo rng?
        value_grad_fn = jax.value_and_grad(
            lambda params: loss_fn(params, batch, is_training=is_training),
            has_aux=True,
        )
        (loss, (metrics, extra, mparams)), grads = value_grad_fn(ts.params)

        ts = ts.apply_gradients(grads=grads)
        ts = ts.replace(mparams=mparams)
        return ts, metrics, extra

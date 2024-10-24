from typing import Any, Callable, Dict

import flax.linen as nn
import jax
import optax
from flax.training.train_state import TrainState as TrainStateFlax
from omegaconf import DictConfig

from prompt_dtla.utils.data_utils import PRNGKeyDict


class TrainState(TrainStateFlax):
    mparams: Any  # state variables for EMA & batch stats
    keys: PRNGKeyDict  # for flax dropout and vq
    # does this need lr or is lr captured in opt_state? (think latter)


kernel_init = nn.initializers.variance_scaling(
    scale=2.0,
    mode="fan_in",
    distribution="truncated_normal",
)
bias_init = nn.initializers.constant(0.0)

default_weight_init: Dict[str, Callable] = {
    "kernel_init": kernel_init,
    "bias_init": bias_init,
}


def get_AdamW_optimizer(config: Dict) -> optax.GradientTransformation:
    """
    expects config to contain the following:
    - lr
    - use_lr_scheduler
    - warmup_steps
    - eps
    """
    if config.use_lr_scheduler:
        warmup_schedule = optax.linear_schedule(
            init_value=0,
            end_value=config.lr,
            transition_steps=config.warmup_steps,
        )

        if config.decay_lr is None:
            lr = warmup_schedule
        else:
            if config.decay_lr == "cosine":
                decay_schedule = optax.cosine_decay_schedule(
                    init_value=config.lr,
                    decay_steps=config.num_updates - config.warmup_steps,
                    alpha=config.min_lr,
                )
            elif config.decay_lr == "linear":
                decay_schedule = optax.linear_schedule(
                    init_value=config.lr,
                    end_value=config.min_lr,
                    transition_steps=config.num_updates,
                )
            else:
                raise ValueError(f"Invalid lr decay type: {config.decay_lr}")

            lr = optax.join_schedules(
                [warmup_schedule, decay_schedule],
                [config.warmup_steps],
            )
    else:
        lr = config.lr

    tx = optax.inject_hyperparams(
        lambda lr: optax.chain(
            optax.clip_by_global_norm(2.0),
            optax.adamw(lr, eps=config.eps),
        )
    )(lr=lr)

    return tx


def save_dict(ts: TrainStateFlax, cfg: DictConfig, num_devices: int = 1) -> Dict:
    """
    Defines a dictionary of objects to save into a checkpoint file
    """
    # use orbax?
    params = ts.params
    mparams = ts.mparams

    if num_devices > 1:
        params = jax.tree_util.tree_map(lambda x: x[0], params)
        mparams = jax.tree_util.tree_map(lambda x: x[0], mparams)

    out = {
        "params": params,
        "mparams": mparams,
    }

    return out

"""
Script to test whether the LAM is ignoring the latent actions
Plug in random actions and see if the LAM still predicts the next observation...
"""

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import tensorflow as tf
from loguru import logger

from prompt_dtla.utils.logger import *

tf_config = tf.compat.v1.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf.config.experimental.set_visible_devices([], "GPU")

from functools import partial
from pathlib import Path

import einops
import haiku as hk
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm
from omegaconf import DictConfig, OmegaConf

import prompt_dtla.resolvers
from prompt_dtla.baselines import NAME_TO_BASELINE_CLS
from prompt_dtla.envs.utils import make_envs
from prompt_dtla.stage import NAME_TO_STAGE_CLS
from prompt_dtla.utils.general_utils import omegaconf_to_dict, print_dict
from prompt_dtla.utils.logger import log
from prompt_dtla.utils.tfds_data_utils import load_data as load_data_tfds
from prompt_dtla.utils.training import TrainState, default_weight_init


def run_sanity_check(
    cfg: DictConfig,
    ts_lam: TrainState,
    rng_seq: hk.PRNGSequence,
    # rng_seq: gutil.PRNGSequence,
    data: tf.data.Dataset,
    split="train",
):
    log(f"relabeling dataset {split}")

    # todo: running into error if i don't feed in the mutable stuff here
    jit_apply = jax.jit(
        partial(ts_lam.apply_fn, mutable=["batch_stats", "ema"]),
        static_argnames=("is_training"),
    )

    for batch in tqdm.tqdm(data.as_numpy_iterator()):
        key = next(rng_seq)
        key, vq_key, sample_key = jax.random.split(key, 3)
        observations = batch["observations"]

        # forward pass through the LAM normally
        lam_output, _ = jit_apply(
            ts_lam.params,
            x=observations.astype(jnp.float32),
            is_training=False,
            rngs={"params": key, "vq": vq_key, "sample": sample_key},
        )
        la = lam_output.idm.latent_actions
        next_obs_pred = lam_output.next_obs_pred

        # compute loss between the predicted and actual observations
        gt_next_obs = observations[:, -1]

        loss = optax.squared_error(next_obs_pred, gt_next_obs)

        # forward pass with some random latent actions
        random_latent_actions = jax.random.uniform(
            key, la.shape, minval=-1.0, maxval=1.0
        )
        lam_output_rand, _ = jit_apply(
            ts_lam.params,
            x=observations.astype(jnp.float32),
            latent_actions=random_latent_actions,
            is_training=False,
            rngs={"params": key, "vq": vq_key, "sample": sample_key},
        )
        next_obs_pred_rand = lam_output_rand.next_obs_pred

        rand_loss = optax.squared_error(next_obs_pred_rand, gt_next_obs)

        log(f"loss: {loss.mean()}, random loss: {rand_loss.mean()}")
        diff = jnp.abs(loss.mean() - rand_loss.mean())
        log(f"diff: {diff}")

        break


@hydra.main(version_base=None, config_name="config", config_path="../cfg")
def main(cfg: DictConfig):
    tf.random.set_seed(0)

    cfg_path = Path(cfg.stage.idm_ckpt) / "config.yaml"
    stage_cfg = OmegaConf.load(str(cfg_path))
    stage_cfg_dict = omegaconf_to_dict(stage_cfg)
    print_dict(stage_cfg_dict)

    # load pretrained IDM/FDM for predicting the latent actions from observations
    log("loading pretrained IDM from checkpoint")
    envs = make_envs(
        env_name=stage_cfg.env.env_name,
        num_envs=1,
        env_id=stage_cfg.env.env_id,
        gamma=1,
    )

    if stage_cfg.env.env_name == "procgen":
        obs_shape = envs.single_observation_space.shape
        continuous_actions = False
        input_action_dim = 1
        action_dim = envs.single_action_space.n
    elif stage_cfg.env.env_name == "mujoco":
        obs_shape = envs.single_observation_space.shape
        continuous_actions = True
        action_dim = input_action_dim = envs.single_action_space.shape[0]
    else:
        raise NotImplementedError

    log(
        f"env id: {stage_cfg.env.env_id}, env name: {stage_cfg.env.env_name}, action dim: {action_dim}, obs shape: {obs_shape}"
    )
    rng_seq = hk.PRNGSequence(stage_cfg.seed)

    NAME_TO_STAGE_CLS.update(NAME_TO_BASELINE_CLS)

    idm = NAME_TO_STAGE_CLS[cfg.stage.name](
        stage_cfg.stage,
        key=next(rng_seq),
        load_from_ckpt=True,
        ckpt_dir=Path(cfg.stage.idm_ckpt),
        ckpt_file=Path(cfg.stage.idm_ckpt_file) if cfg.stage.idm_ckpt_file else None,
        observation_shape=obs_shape,
        action_dim=action_dim,
        input_action_dim=input_action_dim,
        continuous_actions=continuous_actions,
        init_kwargs=default_weight_init,
    )
    ts_lam = idm.create_train_state(next(rng_seq))

    log("loading original observation-only dataset")
    # make sure to not shuffle the data here
    stage_cfg.data.train_frac = 1.0

    # update data directory
    stage_cfg.env.env_id = cfg.env.env_id
    stage_cfg.data.dataset_name = cfg.data.dataset_name

    train_data, eval_data, _ = load_data_tfds(
        stage_cfg, shuffle=False, drop_remainder=False
    )

    run_sanity_check(stage_cfg, ts_lam, rng_seq, train_data, split="")


if __name__ == "__main__":
    main()

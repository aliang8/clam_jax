"""
Script for relabelling observation-only dataset with latent actions
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


def relabel_data_with_la(
    cfg: DictConfig,
    ts_idm: TrainState,
    rng_seq: hk.PRNGSequence,
    # rng_seq: gutil.PRNGSequence,
    data: tf.data.Dataset,
    split="train",
):
    log(f"relabeling dataset {split}")

    # todo: running into error if i don't feed in the mutable stuff here
    jit_apply = jax.jit(
        partial(ts_idm.apply_fn, mutable=["batch_stats", "ema"]),
        static_argnames=("is_training"),
    )

    all_la_quantized = []
    all_la_prequantized = []
    dones = []  # need the dones to figure out where to split

    # params = jax.tree_util.tree_map(lambda x: x[0], idm._ts.params)

    action_dim = cfg.stage.idm.vq.code_dim if "lam" in cfg.stage.name else 1

    for batch in tqdm.tqdm(data.as_numpy_iterator()):
        key = next(rng_seq)
        key, vq_key, sample_key = jax.random.split(key, 3)
        observations = batch["observations"]
        done = batch["is_terminal"]
        dones.extend(done[:, -1])

        # forward pass through the latent action model
        idm_output, _ = jit_apply(
            {"params": ts_idm.params},
            x=observations.astype(jnp.float32),
            is_training=False,
            rngs={"params": key, "vq": vq_key, "sample": sample_key},
        )

        if "lam" in cfg.stage.name:
            la = idm_output.idm.latent_actions
            if cfg.stage.idm.apply_quantization:
                quantized_la = idm_output.idm.vq.quantize
            else:
                quantized_la = la
        elif "vpt" in cfg.stage.name:
            quantized_la = idm_output.action
            la = idm_output.action

            quantized_la = einops.repeat(quantized_la, "b -> b 1")
            la = einops.repeat(la, "b -> b 1")

        all_la_quantized.append(quantized_la)
        all_la_prequantized.append(la)

    log("done relabeling")
    log(f"num total batches: {len(all_la_quantized)}")
    all_la_quantized = np.concatenate(all_la_quantized, axis=0)
    all_la_prequantized = np.concatenate(all_la_prequantized, axis=0)
    dones = np.array(dones)

    # split latent actions based on dones using np.split
    log("splitting latent actions into trajectories")
    quantized_la = np.split(all_la_quantized, np.where(dones)[0] + 1)[:-1]
    prequantized_la = np.split(all_la_prequantized, np.where(dones)[0] + 1)[:-1]
    log(f"num trajs: {len(quantized_la)}")

    for i, la in enumerate(quantized_la):
        b, d = la.shape

        # add zeros at the end to make it the same length as the original
        zeros = np.zeros((1, d))
        pre = np.zeros((cfg.data.context_len, d))
        quantized_la[i] = np.concatenate([pre, la, zeros], axis=0)
        quantized_la[i] = tf.convert_to_tensor(quantized_la[i])

        prequantized_la[i] = np.concatenate([pre, prequantized_la[i], zeros], axis=0)

    # log stats for latent actions
    log("latent action stats")
    log(
        f"la min: {np.min(all_la_quantized)}, la max: {np.max(all_la_quantized)}, la mean: {np.mean(all_la_quantized)}"
    )

    def generator():
        for quantize, la in zip(quantized_la, prequantized_la):
            yield {
                "quantize": quantize,
                "latent_action": la,
            }

    tfds_la = tf.data.Dataset.from_generator(
        generator,
        output_signature={
            "quantize": tf.TensorSpec(shape=(None, action_dim), dtype=tf.float32),
            "latent_action": tf.TensorSpec(shape=(None, action_dim), dtype=tf.float32),
        },
    )

    save_dir = (
        Path(cfg.data.data_dir)
        / "tensorflow_datasets"
        / cfg.data.dataset_name
        / cfg.env.env_id
    )

    save_dir.mkdir(parents=True, exist_ok=True)

    if "vpt" in cfg.stage.name:
        file = f"la-{split}_vpt_nt-{cfg.data.num_trajs}"
    else:
        file = f"la-{split}"

    if Path(file).exists():
        log(f"the latent action dataset {file} already exists")
        # ask for input to overwrite
        overwrite = input("overwrite? (y/n): ")
        if overwrite != "y":
            return

    save_file = save_dir / file
    log(f"saving latent actions to {save_file}")
    tf.data.experimental.save(tfds_la, str(save_file))

    OmegaConf.save(cfg, save_dir / "config.yaml")


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
    ts_idm = idm.create_train_state(next(rng_seq))

    log("loading original observation-only dataset")
    # make sure to not shuffle the data here
    stage_cfg.data.train_frac = 1.0

    # update data directory
    stage_cfg.env.env_id = cfg.env.env_id
    stage_cfg.data.dataset_name = cfg.data.dataset_name

    train_data, eval_data, _ = load_data_tfds(
        stage_cfg, shuffle=False, drop_remainder=False
    )

    if stage_cfg.env.env_name == "procgen":
        relabel_data_with_la(stage_cfg, ts_idm, rng_seq, train_data, split="train")
        relabel_data_with_la(
            stage_cfg, ts_idm, rng_seq, eval_data[stage_cfg.env.env_id], split="val"
        )
    else:
        relabel_data_with_la(stage_cfg, ts_idm, rng_seq, train_data, split="")


if __name__ == "__main__":
    main()

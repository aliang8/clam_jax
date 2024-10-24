import os
import sys

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import time
from pathlib import Path

import einops
import hydra
import jax.tree_util as jtu
import numpy as np
import rlds
import tensorflow as tf
from loguru import logger

logger_format = (
    "<level>{level: <2}</level> <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan>"
    "- <level>{message}</level>"
)
logger.remove()
logger.add(sys.stderr, format=logger_format)

from omegaconf import DictConfig

from prompt_dtla.envs.utils import make_procgen_envs
from prompt_dtla.utils.data_utils import Transition, normalize_obs
from prompt_dtla.utils.logger import log


@hydra.main(version_base=None, config_name="generate_rand_data", config_path="../cfg")
def main(cfg: DictConfig):
    save_file = (
        Path(cfg.data.data_dir)
        / "tensorflow_datasets"
        / cfg.data.dataset_name
        / cfg.env.env_id
        / "random_policy"
    )
    log(f"save_file: {save_file}")

    num_envs = cfg.num_parallel_envs

    if cfg.generate:
        envs = make_procgen_envs(num_envs=num_envs, env_id=cfg.env.env_id, gamma=1.0)

        transitions = []
        obs = envs.reset()
        dones = np.bool_(np.zeros(num_envs))

        timesteps = 0

        start = time.time()

        while not np.all(dones):
            action = np.asarray([envs.action_space.sample() for _ in range(num_envs)])
            obs, reward, dones, info = envs.step(action)
            obs = normalize_obs(obs)

            transitions.append(
                Transition(
                    observation=obs,
                    action=action,
                    reward=reward,
                    done=dones,
                )
            )
            timesteps += 1

            if timesteps >= cfg.max_timesteps_per_env:
                break

        envs.close()

        log(f"total time taken: {time.time() - start}")

        transitions = jtu.tree_map(lambda *x: np.stack(x), *transitions)
        transitions = jtu.tree_map(
            lambda x: einops.rearrange(x, "t n ... -> (n t) ..."), transitions
        )

        num_transitions = transitions.observation.shape[0]

        for k, v in transitions.items():
            log(f"{k}, {v.shape}")

        log(f"num_transitions: {num_transitions}")
        obs_shape = transitions.observation.shape[1:]

        def generator():
            for obs, action, reward in zip(
                transitions.observation, transitions.action, transitions.reward
            ):
                yield {
                    "observations": obs,
                    "actions": action,
                    "rewards": reward,
                }

        # save to tfds data loader
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature={
                "observations": tf.TensorSpec(shape=obs_shape, dtype=tf.float32),
                "actions": tf.TensorSpec(shape=(), dtype=tf.int32),
                "rewards": tf.TensorSpec(shape=(), dtype=tf.float32),
            },
        )

        log(f"saving dataset to file: {save_file}")
        tf.data.experimental.save(dataset, str(save_file))
    else:
        # test load
        log(f"loading dataset from file: {save_file}")
        dataset = tf.data.experimental.load(str(save_file))

        # generate a video of the dataset to check if it's working
        import cv2

        frames = dataset.skip(4000).take(500)
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter("output.mp4", fourcc, 20.0, (64, 64))

        for indx, frame in enumerate(frames):
            frame = frame["observations"].numpy().astype(np.uint8)
            out.write(frame)

        out.release()

        # make vdeo into a gif but make it compressed
        os.system("ffmpeg -i output.mp4 -vf scale=320:-1 -r 10 -y random_policy.gif")
        import ipdb

        ipdb.set_trace()
        dataset = rlds.transformations.batch(dataset, size=2, shift=1)
        dataset = dataset.shuffle(10000000)
        dataset_size = len(dataset)

        train_size = int(0.8 * dataset_size)
        val_size = dataset_size - train_size

        train_ds = dataset.take(train_size)
        val_ds = dataset.skip(train_size)

        log(f"num train: {len(train_ds)}, num val: {len(val_ds)}")

        iterator = train_ds.batch(32).as_numpy_iterator()

        for batch in iterator:
            for key, value in batch.items():
                log(f"{key}: {value.shape}")
            break


if __name__ == "__main__":
    main()

"""
Script for visualizing trajectories from the d4rl dataset
"""

from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
import tqdm

from prompt_dtla.envs.utils import make_mujoco_envs
from prompt_dtla.utils.logger import log
from prompt_dtla.utils.rollouts import annotate_single_video


def main():
    log("visualizing trajectories")
    env_id = "halfcheetah-expert-v2"
    ds_name = "v2-expert"

    # loading from the raw dataset
    data_dir = Path(
        "/scr/shared/prompt_dtla/tensorflow_datasets/d4rl_mujoco_halfcheetah"
    )

    log(f"loading dataset for {env_id}")
    save_file = data_dir / ds_name
    ds = tf.data.experimental.load(str(save_file))

    dataloader = ds.take(5).as_numpy_iterator()

    env = make_mujoco_envs(1, env_id)
    env = env.envs[0]
    qpos_dim = env.sim.data.qpos.size

    fps = 10

    # video_folder = save_file / "videos"
    video_folder = Path("videos")
    video_folder.mkdir(parents=True, exist_ok=True)
    log(f"Saving videos to {video_folder}")

    for traj_indx, traj in tqdm.tqdm(
        enumerate(dataloader), desc="trajectories", total=2
    ):
        env.reset()
        states = traj["env_state"]
        actions = traj["actions"]
        dones = traj["is_terminal"]
        rewards = traj["rewards"]
        returns = np.cumsum(rewards)

        log(f"actions shape: {actions.shape}, done shape: {dones.shape}")

        frames = []
        for state in tqdm.tqdm(states, desc="timesteps"):
            qpos, qvel = np.split(state, [qpos_dim])
            env.set_state(qpos, qvel)

            frame = env.render(mode="rgb_array")
            frames.append(frame)

        meta = {
            "S": np.arange(len(states)),
            "R": rewards,
            "Ret": returns,
            "D": dones,
            "A": actions,
        }

        # annotate video
        annotated_frames = annotate_single_video(env_id, video=frames, meta=meta)

        # make cv2 video
        video_file = video_folder / f"traj_{traj_indx}.mp4"
        log(f"saving video to {video_file}")

        video = cv2.VideoWriter(
            str(video_file),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,  # fps
            annotated_frames.shape[1:-1][::-1],
        )
        for frame in annotated_frames:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video.write(frame)

        video.release()


if __name__ == "__main__":
    main()

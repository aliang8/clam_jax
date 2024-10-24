"""
Script for visualizing trajectories from the dataset
"""

from pathlib import Path

import cv2
import numpy as np
import tqdm

from prompt_dtla.envs.utils import (
    ALL_PROCGEN_GAMES,
    procgen_action_meanings,
)
from prompt_dtla.utils.logger import log


def main():
    log("visualizing trajectories")
    # loading from the raw dataset
    data_dir = Path("/scr/shared/prompt_dtla/datasets/procgen/expert_data")

    fps = 10
    img_size = (256, 256)

    for game in tqdm.tqdm(ALL_PROCGEN_GAMES, desc="games"):
        game_dir = data_dir / game

        for split in ["train", "test"]:
            # each file in this folder is a npz file for a single trajectory
            traj_dir = game_dir / split / "trajs"

            trajs = list(
                sorted(
                    traj_dir.glob("*.npz"),
                    key=lambda x: int(str(x).split("_")[-1].replace(".npz", "")),
                )
            )

            log(f"game: {game}, split: {split}, num trajs: {len(trajs)}")

            for traj_file in tqdm.tqdm(trajs[:10], desc="trajs"):
                traj_data = np.load(traj_file)
                obs = traj_data["observations"]
                actions = traj_data["actions"]
                dones = traj_data["dones"]
                rewards = traj_data["rewards"]
                returns = np.cumsum(rewards)

                log(
                    f"obs shape: {obs.shape}, actions shape: {actions.shape}, done shape: {dones.shape}"
                )

                # make cv2 video
                video_file = traj_file.with_suffix(".mp4")
                log(f"saving video to {video_file}")

                video = cv2.VideoWriter(
                    video_file,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,  # fps
                    img_size,
                )

                for frame_indx, (frame, action, return_, done) in enumerate(
                    zip(obs, actions, returns, dones)
                ):
                    # resize frame
                    frame = cv2.resize(frame, img_size, interpolation=cv2.INTER_AREA)
                    frame = (frame + 0.5) * 255
                    frame = frame.astype(np.uint8)

                    # add some text
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.6
                    font_color = (255, 0, 0)  # red
                    font_thickness = 1
                    action_label = procgen_action_meanings[int(action)]

                    texts = [
                        f"t: {frame_indx}, r: {return_:.2f}",
                        f"a: {action_label}, d: {done}",
                    ]

                    x, y = 5, 20
                    for text in texts:
                        cv2.putText(
                            frame,
                            text,
                            (x, y),
                            font,
                            font_scale,
                            font_color,
                            font_thickness,
                        )
                        y += 20

                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    video.write(frame)

                video.release()


if __name__ == "__main__":
    main()

from pathlib import Path

import numpy as np
import tensorflow as tf
import tqdm

from prompt_dtla.envs.utils import ALL_PROCGEN_GAMES
from prompt_dtla.utils.logger import log

if __name__ == "__main__":
    data_dir = Path("/scr/shared/prompt_dtla")

    features_dict = {
        "observations": tf.TensorSpec(shape=(None, 64, 64, 3), dtype=np.float32),
        "actions": tf.TensorSpec(shape=(None,), dtype=np.int32),
        "discount": tf.TensorSpec(shape=(None,), dtype=np.float32),
        "rewards": tf.TensorSpec(shape=(None,), dtype=np.float32),
        "is_first": tf.TensorSpec(shape=(None,), dtype=np.bool_),
        "is_last": tf.TensorSpec(shape=(None,), dtype=np.bool_),
        "is_terminal": tf.TensorSpec(shape=(None,), dtype=np.bool_),
    }

    for game in ALL_PROCGEN_GAMES:
        for split in ["train", "test"]:
            save_dir = Path(data_dir) / "tensorflow_datasets" / "procgen_dataset" / game
            save_file = save_dir / split

            if save_file.exists():
                log("Already exists, skipping game: {}, split: {}", game, split)
                continue

            trajectories = []

            split_data_dir = (
                data_dir / "datasets" / "procgen" / "expert_data" / game / split
            )
            log("Processing game: {}, split: {}", game, split)
            traj_dir = split_data_dir / "trajs"

            files = list(traj_dir.glob("*.npz"))

            for file in tqdm.tqdm(files):
                data = dict(np.load(file, allow_pickle=True))

                final_data = {
                    "observations": data["observations"],
                    "actions": data["actions"],
                    "rewards": data["rewards"],
                    "discount": np.ones_like(data["rewards"]),
                    "is_last": np.zeros_like(data["rewards"]),
                    "is_first": np.zeros_like(data["rewards"]),
                    "is_terminal": np.zeros_like(data["rewards"]),
                }

                final_data["is_last"][-1] = 1
                final_data["is_terminal"][-1] = 1
                final_data["is_first"][0] = 1

                trajectories.append(final_data)

            # create tfds from generator
            def generator():
                for trajectory in trajectories:
                    yield trajectory

            trajectory_tfds = tf.data.Dataset.from_generator(
                generator, output_signature=features_dict
            )

            log("Saving to: {}", save_file)
            tf.data.experimental.save(trajectory_tfds, str(save_file))

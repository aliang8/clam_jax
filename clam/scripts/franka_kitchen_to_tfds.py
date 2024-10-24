from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf

from prompt_dtla.utils.logger import log

if __name__ == "__main__":
    data_dir = Path("/scr/shared/prompt_dtla/datasets/franka_kitchen")

    dataset_file = "kitchen_no_test.hdf5"
    dataset_path = data_dir / dataset_file

    # trajs_dir = data_dir / "trajs"
    # trajs_dir.mkdir(exist_ok=True)

    tfds_data_dir = Path("/scr/shared/prompt_dtla/tensorflow_datasets")
    tfds_data_dir.mkdir(exist_ok=True)
    use_states = True

    if use_states:
        save_file = tfds_data_dir / dataset_file.replace(".hdf5", "_states")
    else:
        save_file = tfds_data_dir / dataset_file.replace(".hdf5", "_images")

    obs_shape = (84, 84, 3) if not use_states else (60,)

    features_dict = {
        "observations": tf.TensorSpec(shape=(None, *obs_shape), dtype=np.float32),
        "actions": tf.TensorSpec(shape=(None, 9), dtype=np.int32),
        "discount": tf.TensorSpec(shape=(None,), dtype=np.float32),
        "rewards": tf.TensorSpec(shape=(None,), dtype=np.float32),
        "is_first": tf.TensorSpec(shape=(None,), dtype=np.bool_),
        "is_last": tf.TensorSpec(shape=(None,), dtype=np.bool_),
        "is_terminal": tf.TensorSpec(shape=(None,), dtype=np.bool_),
        "images": tf.TensorSpec(shape=(None, 84, 84, 3), dtype=np.float32),
    }

    with h5py.File(dataset_path, "r") as f:
        frame = f["data"]["demo_0"]["obs"]["agentview_image"][-1]

        trajectories = []

        avg_traj_len = 0

        for traj_indx, traj in enumerate(f["data"]):
            observations = f["data"][traj]["obs"]["agentview_image"][:]
            actions = f["data"][traj]["actions"][:]
            rewards = f["data"][traj]["rewards"][:]
            dones = f["data"][traj]["dones"][:]
            states = f["data"][traj]["states"][:]

            if use_states:
                images = observations
                observations = states
            else:
                images = observations

            traj_data = {
                "observations": observations,
                "actions": actions,
                "rewards": rewards,
                # "dones": dones,
                "discount": np.ones_like(rewards),
                "is_last": np.zeros_like(rewards),
                "is_first": np.zeros_like(rewards),
                "is_terminal": np.zeros_like(rewards),
                "images": images,
            }

            traj_data["is_last"][-1] = 1
            traj_data["is_terminal"][-1] = 1
            traj_data["is_first"][0] = 1

            # write to a new file
            # npz_file_path = trajs_dir / f"traj_{traj_indx}.npz"
            # np.savez_compressed(npz_file_path, **traj_data)

            trajectories.append(traj_data)
            avg_traj_len += len(rewards)

        avg_traj_len /= len(trajectories)
        log(f"Average trajectory length: {avg_traj_len}")
        log(f"Number of trajectories: {len(trajectories)}")

        # create a tfds from generator
        def generator():
            for trajectory in trajectories:
                yield trajectory

        trajectory_tfds = tf.data.Dataset.from_generator(
            generator, output_signature=features_dict
        )

        log(f"Saving to: {save_file}")
        tf.data.experimental.save(trajectory_tfds, str(save_file))

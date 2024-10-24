from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf

from prompt_dtla.utils.logger import log

HALFCHEETAH_IMAGE_SIZE = (64, 64, 3)

HALFCHEETAH_FEATURES_DICT = {
    "observations": tf.TensorSpec(shape=(None, 17), dtype=np.float32),
    "actions": tf.TensorSpec(shape=(None, 6), dtype=np.float32),
    "discount": tf.TensorSpec(shape=(None,), dtype=np.float32),
    "rewards": tf.TensorSpec(shape=(None,), dtype=np.float32),
    "is_first": tf.TensorSpec(shape=(None,), dtype=np.bool_),
    "is_last": tf.TensorSpec(shape=(None,), dtype=np.bool_),
    "is_terminal": tf.TensorSpec(shape=(None,), dtype=np.bool_),
    "env_state": tf.TensorSpec(shape=(None, 18), dtype=np.float32),
}

METAWORLD_IMAGE_SIZE = (64, 64, 3)

METAWORLD_FEATURES_DICT = {
    "observations": tf.TensorSpec(shape=(None, 39), dtype=np.float32),
    "actions": tf.TensorSpec(shape=(None, 4), dtype=np.float32),
    "discount": tf.TensorSpec(shape=(None,), dtype=np.float32),
    "rewards": tf.TensorSpec(shape=(None,), dtype=np.float32),
    "is_first": tf.TensorSpec(shape=(None,), dtype=np.bool_),
    "is_last": tf.TensorSpec(shape=(None,), dtype=np.bool_),
    "is_terminal": tf.TensorSpec(shape=(None,), dtype=np.bool_),
}


def save_dataset(trajectories, save_file: Path, env_name: str, save_imgs: bool):
    log(f"Saving dataset to: {save_file}")
    save_file.parent.mkdir(parents=True, exist_ok=True)

    # create tfds from generator
    def generator():
        for trajectory in trajectories:
            yield trajectory

    features_dict = (
        HALFCHEETAH_FEATURES_DICT
        if env_name == "halfcheetah"
        else METAWORLD_FEATURES_DICT
    )

    if save_imgs:
        if env_name == "metaworld":
            features_dict["images"] = tf.TensorSpec(
                shape=(None, *METAWORLD_IMAGE_SIZE), dtype=np.uint8
            )
        elif env_name == "halfcheetah":
            features_dict["images"] = tf.TensorSpec(
                shape=(None, *HALFCHEETAH_IMAGE_SIZE), dtype=np.uint8
            )

    trajectory_tfds = tf.data.Dataset.from_generator(
        generator, output_signature=features_dict
    )
    tf.data.experimental.save(trajectory_tfds, str(save_file))

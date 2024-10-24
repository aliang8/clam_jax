import os
from functools import partial
from pathlib import Path

import rlds
import tensorflow as tf
import tensorflow_datasets as tfds
from loguru import logger
from ml_collections.config_dict import ConfigDict

from prompt_dtla.utils.logger import log
from prompt_dtla.utils.randaugment import randaugment

os.environ["TFDS_DATA_DIR"] = "/scr/shared/prompt_dtla/tensorflow_datasets"


def episode_to_step_custom(episode, size):
    episode = tf.data.Dataset.from_tensor_slices(episode)
    return rlds.transformations.batch(episode, size=size, shift=1, drop_remainder=True)


def apply_image_augmentations(dataset):
    """Augment dataset with a list of augmentations."""

    def augment(seeds, features):
        # observations = tf.cast(features["observations"], tf.uint8)
        # for aug_fn in augmentations:
        #     observations = aug_fn(seeds, observations)
        observations = features["observations"]

        # need to convert to int32 first
        observations = (observations + 0.5) * 255.0
        observations = tf.cast(observations, tf.uint8)

        # apply randaugment to each observation
        if observations.ndim > 3:
            observations = tf.map_fn(
                partial(randaugment, num_layers=1, magnitude=10, seeds=seeds),
                observations,
            )
        else:
            observations = randaugment(
                observations, num_layers=1, magnitude=10, seeds=seeds
            )

        # convert back to float32
        observations = tf.cast(observations, tf.float32) / 255.0 - 0.5

        features["observations"] = observations
        return features

    randds = tf.data.experimental.RandomDataset(1).batch(2).batch(4)
    dataset = tf.data.Dataset.zip((randds, dataset))
    dataset = dataset.map(augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset


# add additional fields to the dataset
def add_new_fields(x):
    x["mask"] = tf.ones_like(x["actions"])
    x["timestep"] = tf.range(tf.shape(x["actions"])[0])
    return x


def use_image_observations(x):
    if "images" in x:
        x["observations"] = x["images"]
    return x


# remove trajectories where the number of steps is less than 2
def filter_fn(traj):
    return tf.math.greater(tf.shape(traj["observations"])[0], 2)


def reshape_obs(x, channel_first=False, first_framestack=False, normalize=False):
    if normalize:
        x["observations"] = x["observations"] / 255.0
    if len(x["observations"].shape) > 2:
        x["observations"] = tf.image.resize(x["observations"], [64, 64])
    if first_framestack:
        x["observations"] = x["observations"][..., 0:1]
    if channel_first:
        x["observations"] = tf.transpose(x["observations"], [0, 3, 1, 2])
    return x


def process_tfds_trajectories(
    ds: tf.data.Dataset,
    channel_first: bool = False,
    shuffle: bool = True,
    drop_remainder: bool = True,
    env_name: str = "",
    env_id: str = "",
    data_type: str = "lapo",
    seq_len: int = 0,
    batch_size: int = 32,
    num_trajs: int = -1,
    num_examples: int = -1,
    use_image_obs: bool = False,
    apply_data_aug: bool = False,
    apply_resize: bool = False,
    first_framestack: bool = False,
    normalize: bool = False,
    num_few_shot_prompts: int = 0,
):
    """
    Applies transformations to base tfds such as batching, shuffling, etc.
    """
    ds = ds.filter(filter_fn)

    # caching the dataset makes it faster in the next iteration
    # over the entire dataset?
    ds = ds.cache()

    # shuffle the dataset first?
    ds = ds.shuffle(1000, reshuffle_each_iteration=False)

    # limit the number of trajectories that we use
    ds = ds.take(num_trajs)
    log(f"\ttaking {num_trajs} trajectories")

    ds = ds.map(add_new_fields)

    if use_image_obs:
        ds = ds.map(use_image_observations)

    if data_type == "trajectories":
        # ds = ds.padded_batch(config.data.batch_size)
        ds = ds.bucket_by_sequence_length(
            lambda x: tf.shape(x["observations"])[0],
            bucket_boundaries=[500, 1000, 3000],
            bucket_batch_sizes=[2, 2, 1, 1],
            pad_to_bucket_boundary=True,
            drop_remainder=True,
        )
        if num_examples != -1:
            ds = ds.take(num_examples)

    else:
        if data_type == "lapo" or data_type == "n_step":
            ds = ds.flat_map(partial(episode_to_step_custom, size=seq_len))
        elif data_type == "transitions":
            ds = ds.flat_map(tf.data.Dataset.from_tensor_slices)

        # shuffle the full dataset
        if shuffle:
            log("shuffling dataset")
            ds = ds.shuffle(1000, reshuffle_each_iteration=False)

        if apply_data_aug:
            ds = apply_image_augmentations(ds)

        # procgen observations are channel_last
        ds = ds.map(
            partial(
                reshape_obs,
                channel_first=channel_first,
                first_framestack=first_framestack,
                normalize=normalize,
            )
        )

        if num_examples != -1:
            log(f"\ttaking {num_examples} examples")
            ds = ds.take(num_examples)

            # recommended to do dataset.take(k).cache().repeat()
            ds = ds.cache()

        # combine two shuffles of the same dataset
        if num_few_shot_prompts > 0:
            # TODO: i should be combining trajectories from the same MDP
            prompt_ds = ds.shuffle(1000, reshuffle_each_iteration=True)
            ds = ds.batch(batch_size, drop_remainder=drop_remainder)
            prompt_ds = prompt_ds.batch(batch_size, drop_remainder=drop_remainder)
            concat_ds = tf.data.Dataset.zip((prompt_ds, ds))

            def combine_trajectories(x, y):
                new_traj = {}
                # create an index to keep track of which trajectory the step belongs to
                traj_index_1 = tf.zeros_like(x["mask"])
                traj_index_2 = tf.ones_like(y["mask"])
                new_traj["traj_index"] = tf.concat([traj_index_1, traj_index_2], axis=1)

                for k, v in x.items():
                    # concatenate along the time dimension
                    # TODO: check if this is correct
                    new_traj[k] = tf.concat([x[k], y[k]], axis=1)

                return new_traj

            icl_ds = concat_ds.map(lambda x, y: combine_trajectories(x, y))
            ds = icl_ds
        else:
            ds = ds.batch(batch_size, drop_remainder=drop_remainder)

    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds


def process_tfds_transitions(
    ds: tf.data.Dataset,
    channel_first: bool = False,
    shuffle: bool = True,
    drop_remainder: bool = True,
    data_type: str = "lapo",
    seq_len: int = 0,
    batch_size: int = 32,
):
    ds = rlds.transformations.batch(ds, size=seq_len + 2, shift=1, drop_remainder=True)

    ds = ds.map(add_new_fields)

    ds = ds.map(partial(reshape_obs, channel_first=channel_first))

    if shuffle:
        log("shuffling dataset")
        ds = ds.shuffle(10000, reshuffle_each_iteration=False)

    ds = ds.batch(batch_size, drop_remainder=drop_remainder)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds


def load_latent_actions(
    config: ConfigDict, ds: tf.data.Dataset, split: str = "", ds_name: str = ""
):
    data_dir = Path(config.data.data_dir) / "tensorflow_datasets"
    if not ds_name:
        ds_name = config.data.dataset_name

    if "vpt" in config.stage.hp_name:
        file = f"la-{split}_vpt_nt-{config.stage.idm_nt}"
    else:
        file = f"la-{split}"

    la_file = data_dir / ds_name / file
    latent_actions = tf.data.experimental.load(str(la_file))
    log(f"loaded latent actions tfds from {la_file}")

    ds = tf.data.Dataset.zip((ds, latent_actions))

    def combine_latent_actions(x, la):
        if config.data.replace_action_with_la:
            x["actions"] = la["quantize"]

        x["latent_actions"] = la["quantize"]
        return x

    ds = ds.map(combine_latent_actions)
    return ds


def load_data(
    config: ConfigDict,
    shuffle: bool = True,
    drop_remainder: bool = True,
    channel_first: bool = False,
):
    """
    Returns a dictionary containing the training, validation and prompt datasets.
    Validation dataset is a dictionary of {env_id: dataset} where dataset is a tf.data.Dataset

    Args:
        shuffle: shuffle the dataset after applying transformations (e.g. making N-step windows)
    """
    datasets = {}
    data_dir = Path(config.data.data_dir) / "tensorflow_datasets"
    log(f"loading tfds dataset from: {data_dir}")

    datasets["prompt"] = {}

    log(f"seq_len: {config.data.seq_len}")
    # procgen already has predefined train and validation splits

    kwargs = dict(
        channel_first=channel_first,
        shuffle=shuffle,
        drop_remainder=drop_remainder,
        num_trajs=config.data.num_trajs,
        num_examples=config.data.num_examples,
        env_name=config.env.env_name,
        env_id=config.env.env_id,
        data_type=config.data.data_type,
        seq_len=config.data.seq_len,
        batch_size=config.data.batch_size,
        apply_data_aug=config.data.image_augmentations,
        num_few_shot_prompts=config.data.num_few_shot_prompts,
        use_image_obs=config.data.use_image_obs,
    )

    if config.env.env_name == "procgen":
        for split in ["train", "test"]:
            datasets[split] = {}

            env_ids = (
                config.env.split.training_env_ids
                if split == "train"
                else config.env.split.eval_env_ids
            )
            for env_id in env_ids:
                ds_name = f"{config.data.dataset_name}/{env_id}"
                log(f"loading {split} dataset for {env_id}")

                save_file = data_dir / ds_name / split
                original_ds = tf.data.experimental.load(str(save_file))

                log(f"len of original_ds: {len(original_ds)}")

                ds = original_ds

                if config.data.load_latent_actions:
                    ds = load_latent_actions(
                        config, ds=ds, split=split, ds_name=ds_name
                    )

                kwargs["env_id"] = env_id
                processed_ds = process_tfds_trajectories(ds, **kwargs)
                datasets[split][env_id] = processed_ds

                if split == "test":
                    kwargs_prompt = kwargs.copy()
                    kwargs_prompt["num_few_shot_prompts"] = 0
                    kwargs_prompt["batch_size"] = config.num_eval_rollouts
                    kwargs_prompt["num_trajs"] = -1
                    kwargs_prompt["num_examples"] = -1
                    prompt_ds = process_tfds_trajectories(ds, **kwargs_prompt)
                    datasets["prompt"][env_id] = prompt_ds

    elif config.env.env_name == "mujoco" or config.env.env_name == "metaworld":
        env_id = config.env.env_id
        ds_name = config.data.dataset_name

        if config.data.use_image_obs:
            ds_name = f"{ds_name}_images"

        log(f"loading dataset for {env_id}")

        save_file = data_dir / ds_name
        ds = tf.data.experimental.load(str(save_file))
        if config.data.load_latent_actions:
            ds = load_latent_actions(config, ds=ds, ds_name=f"{ds_name}/{env_id}")

        log(f"dataset name: {ds_name}")
        log(f"len of original_ds: {len(ds)}")

        # split dataset into train and eval
        num_trajectories = len(ds)
        # assert config.data.train_frac < 1.0, "train_frac must be less than 1.0"
        num_train = int(num_trajectories * config.data.train_frac)
        num_eval = num_trajectories - num_train

        # first shuffle the dataset once
        if shuffle:
            log("shuffling dataset mujoco")
            ds = ds.shuffle(1000, reshuffle_each_iteration=False)

        train_ds = ds.take(num_train)
        eval_ds = ds.skip(num_train)
        log(
            f"num train trajs: {num_train}, num eval trajs: {num_eval}, len train: {len(train_ds)}, len eval: {len(eval_ds)}"
        )

        log("processing train dataset")
        train_ds = process_tfds_trajectories(train_ds, **kwargs)
        datasets["train"] = {config.env.env_id: train_ds}

        # use all the trajectories in the eval dataset
        kwargs_eval = kwargs.copy()
        kwargs_eval["num_trajs"] = -1
        kwargs_eval["num_examples"] = -1
        log("processing eval dataset")
        eval_ds = process_tfds_trajectories(eval_ds, **kwargs_eval)
        datasets["test"] = {config.env.env_id: eval_ds}
    elif config.env.env_name == "atari":
        ds = tf.data.Dataset.load(f"{data_dir}/{ds_name}")

        ds = process_tfds_trajectories(
            ds, apply_resize=True, first_framestack=True, normalize=True, **kwargs
        )
        datasets["train"] = {config.env.env_id: ds}
        datasets["test"] = {config.env.env_id: datasets["train"]}

    # combine the train data for all envs
    ds_l = list(datasets["train"].values())
    ds = tf.data.Dataset.from_tensor_slices(ds_l)

    # put everything into one big dataset
    concat_ds = ds.interleave(
        lambda x: x,
        cycle_length=1,
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    datasets["train"] = concat_ds

    # the last item is the prompt dataset
    return datasets["train"], datasets["test"], datasets["prompt"]


if __name__ == "__main__":
    config = ConfigDict(
        {
            # "env": {"env_name": "procgen", "env_id": "bigfish"},
            "env": {"env_name": "mujoco", "env_id": "halfcheetah"},
            "data": {
                "data_dir": "/scr/shared/prompt_dtla",
                "load_latent_actions": False,
                "num_trajs": -1,
                "num_examples": 100,
                "batch_size": 128,
                "seq_len": 1,
                "data_type": "lapo",
                "dataset_name": "procgen_dataset_v2",
                "image_augmentations": False,
                "load_random_policy_data": False,
                "train_frac": 0.9,
            },
        }
    )
    tf.random.set_seed(0)

    train_ds, val_ds, test_ds = load_data(config, channel_first=False)

    # log("Loading transitions dataset:")
    # train_ds, val_ds, test_ds = load_data(config, channel_first=False)
    # for batch in train_ds.as_numpy_iterator():
    #     for k, v in batch.items():
    #         print(k, v.shape)

    #     observations = batch["observations"]

    #     # make a grid of observations and show with plt
    #     # plt.figure(figsize=(10, 10))
    #     # for i in range(16):
    #     #     # for j in range(4):
    #     #     # plt.subplot(4, 4, i * 4 + j + 1)
    #     #     # plt.imshow(observations[i, ..., j])
    #     #     # plt.title(f"Frame {j}")
    #     #     # plt.axis("off")
    #     #     plt.subplot(4, 4, i + 1)
    #     #     plt.imshow(observations[i])
    #     #     plt.axis("off")

    #     # # tight layout
    #     # plt.tight_layout()
    #     # plt.savefig(f"{config.env.env_id}_observations.png")
    #     # plt.close()
    #     break

    log("Loading LAPO dataset:")
    config.data.data_type = "lapo"
    train_ds, val_ds, test_ds = load_data(config)
    import matplotlib.pyplot as plt

    for i, batch in enumerate(train_ds):
        for k, v in batch.items():
            print(k, v.shape)

        import ipdb

        ipdb.set_trace()
        observations = batch["observations"][0] + 0.5

        print(observations.shape)
        plt.figure(figsize=(10, 10))

        plt.imshow(observations[0])
        # for i in range(observations.shape[0]):
        #     plt.subplot(5, 5, i + 1)
        #     plt.imshow(observations[i])
        plt.axis("off")

        plt.tight_layout()
        plt.savefig(
            f"{config.env.env_id}_observations_lapo_0.png",
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close()

        plt.figure(figsize=(10, 10))

        plt.imshow(observations[-1])
        # for i in range(observations.shape[0]):
        #     plt.subplot(5, 5, i + 1)
        #     plt.imshow(observations[i])
        plt.axis("off")

        plt.tight_layout()
        plt.savefig(
            f"{config.env.env_id}_observations_lapo_20.png",
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close()
        break

    # log("Loading trajectory dataset")
    # config.data.data_type = "trajectories"
    # train_ds, val_ds, test_ds = load_data(config)
    # for i, batch in enumerate(val_ds["coinrun"]):
    #     for k, v in batch.items():
    #         print(k, v.shape)

    #     observations = batch["observations"][0]

    #     # make a video
    #     import cv2

    #     fourcc = cv2.VideoWriter_fourcc(*"XVID")
    #     out = cv2.VideoWriter(f"output_{i}.mp4", fourcc, 20.0, (64, 64))

    #     for i in range(observations.shape[0]):
    #         frame = observations[i].numpy()
    #         frame = ((frame + 0.5) * 255.0).astype("uint8")
    #         # import ipdb

    #         # ipdb.set_trace()
    #         # frame = frame.transpose(1, 2, 0)
    #         out.write(frame)

    #     out.release()

    #     import ipdb

    #     ipdb.set_trace()

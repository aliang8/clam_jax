import multiprocessing
import os
import time
from functools import partial
from pathlib import Path

import jax.tree_util as jtu
import numpy as np
import tqdm

from prompt_dtla.envs.utils import ALL_PROCGEN_GAMES
from prompt_dtla.utils.logger import log

TRAIN_CHUNK_LEN = 32_768
TEST_CHUNK_LEN = 4096


def normalize_obs(obs: np.ndarray) -> np.ndarray:
    # assert not torch.is_floating_point(obs)
    return obs.astype(np.float64) / 255 - 0.5


class DataStager:
    def __init__(
        self,
        files: list[Path],
        chunk_len: int,
        obs_depth: int = 3,
    ) -> None:
        self.np_d = None  # type: ignore
        self.obs_depth = obs_depth
        self.files = files
        self.chunk_len = chunk_len
        # sort files
        self.files = files

        num_transitions = self.chunk_len * len(self.files)
        log(f"num_transitions: {num_transitions}")
        self.np_d = {
            "observations": np.zeros(
                (num_transitions, 64, 64, self.obs_depth), dtype=np.float32
            ),
            "actions": np.zeros(num_transitions, dtype=np.int64),
            "dones": np.zeros(num_transitions, dtype=np.int64),
            "rewards": np.zeros(num_transitions),
            "next_observations": np.zeros(
                (num_transitions, 64, 64, self.obs_depth), dtype=np.float32
            ),
        }

        log("data stager loading data...")
        self._load()

    def _load(self):
        for i, path in tqdm.tqdm(enumerate(self.files)):
            self._load_chunk(path, i)

    def _load_chunk(self, path: Path, i: int):
        keys_map = {
            "observations": "obs",
            "actions": "ta",
            "dones": "done",
            "rewards": "rewards",
            "next_observations": "obs",
        }

        data = np.load(path)
        log(f"loading data from: {path}")

        assert data["obs"].shape[0] == self.chunk_len, data["obs"].shape

        for k in self.np_d.keys():
            v = data[keys_map[k]]
            if k == "observations" or k == "next_observations":
                v = normalize_obs(v)

            # print(v.shape, self.chunk_len)
            assert len(v) == self.chunk_len, v.shape
            self.np_d[k][i * self.chunk_len : (i + 1) * self.chunk_len] = v


def split_data_into_trajectories(data):
    dones = data["dones"]
    # get indices where the trajectory ends
    traj_ends = np.where(dones)[0]
    traj_ends = np.concatenate([np.array([-1]), traj_ends])
    traj_lens = traj_ends[1:] - traj_ends[:-1]
    max_len = int(traj_lens.max())
    data["mask"] = np.ones_like(data["dones"])
    # split each data by done
    data = jtu.tree_map(lambda x: np.split(x, np.where(dones)[0] + 1)[:-1], data)
    return data, max_len


def save_trajectory(index_traj_pair, trajs_dir):
    index, traj = index_traj_pair
    npz_file_path = trajs_dir / f"traj_{index}.npz"
    if not os.path.exists(npz_file_path):
        np.savez_compressed(npz_file_path, **traj)


def save_trajectories(trajs, trajs_dir):
    with multiprocessing.Pool() as pool:
        list(
            tqdm.tqdm(
                pool.imap_unordered(
                    partial(save_trajectory, trajs_dir=trajs_dir),
                    enumerate(trajs),
                    chunksize=1,
                ),
                total=len(trajs),
            )
        )


def process_games(games, data_dir):
    for split in ["train", "test"]:
        split_data_dir = data_dir / "procgen" / "expert_data" / game / split

        log(f"Processing game: {game}, split: {split}")
        log(f"loading data from: {split_data_dir}")

        data_files = list(split_data_dir.glob("*.npz"))
        log(f"num data files: {len(data_files)}")

        data_files = sorted(data_files, key=lambda x: int(x.stem.split("_")[-1]))
        data = DataStager(
            files=data_files,
            chunk_len=TRAIN_CHUNK_LEN if split == "train" else TEST_CHUNK_LEN,
        )

        trajs, max_len = split_data_into_trajectories(data.np_d)
        log(f"max_len: {max_len}")
        num_trajs = len(trajs["observations"])
        log(f"num trajs: {num_trajs}")

        # save individual trajectories as npy files
        trajs_dir = split_data_dir / "trajs"
        trajs_dir.mkdir(exist_ok=True)

        # split into lists of dicts
        log("splitting data into lists of dicts")
        trajs = [{k: v[i] for k, v in trajs.items()} for i in range(num_trajs)]

        log("saving trajectories")

        start = time.time()
        save_trajectories(trajs, trajs_dir)
        log(
            f"finished processing game: {game}, split: {split}, took: {time.time() - start} seconds"
        )


if __name__ == "__main__":
    data_dir = Path("/scr/shared/prompt_dtla/datasets")

    for game in ALL_PROCGEN_GAMES:
        process_games(game, data_dir)

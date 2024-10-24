import pickle
from pathlib import Path
from typing import Callable, Dict, Tuple

import flax
import flax.linen as nn
from jaxtyping import Key
from loguru import logger
from omegaconf import DictConfig

from prompt_dtla.utils.logger import log
from prompt_dtla.utils.training import TrainState


class Base(flax.struct.PyTreeNode):
    cfg: DictConfig
    key: Key
    observation_shape: Tuple

    action_dim: int
    input_action_dim: int
    continuous_actions: bool
    init_kwargs: Dict

    decoder: Callable = None
    task_dim: int = 0
    model_key: str = ""
    load_from_ckpt: bool = False
    ckpt_file: str = None
    ckpt_dir: str = None
    num_devices: int = 1

    def load_model_from_ckpt(self) -> Tuple[dict, nn.Module]:
        if self.ckpt_file is not None:
            pass
        elif self.ckpt_dir is not None:
            ckpt_dir = Path(self.ckpt_dir) / "model_ckpts"
            # search and sort by epoch
            ckpt_files = sorted(ckpt_dir.glob("*.pkl"), reverse=True)
            ckpt_file = ckpt_files[0]

        assert ckpt_file, "ckpt_file not provided"
        log(f"loading {self.cfg.name} from checkpoint {ckpt_file}")
        with open(ckpt_file, "rb") as f:
            ts = pickle.load(f)
        return ts

    def create_train_state(self, key: Key) -> TrainState:
        pass

    def update(self, key: Key, batch: dict) -> dict:
        pass

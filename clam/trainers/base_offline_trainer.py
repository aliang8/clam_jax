import pickle
from pathlib import Path
from typing import Dict

import haiku as hk
import jax
import numpy as np
import tensorflow as tf
import torch
import wandb
from hydra.core.hydra_config import HydraConfig
from jax import config as jax_config
from loguru import logger
from omegaconf import DictConfig, OmegaConf

import prompt_dtla.utils.general_utils as gutl
from prompt_dtla.baselines import NAME_TO_BASELINE_CLS
from prompt_dtla.envs.utils import make_envs
from prompt_dtla.stage import NAME_TO_STAGE_CLS
from prompt_dtla.utils.general_utils import omegaconf_to_dict
from prompt_dtla.utils.logger import log
from prompt_dtla.utils.tfds_data_utils import load_data as load_data_tfds
from prompt_dtla.utils.training import default_weight_init


class BaseOfflineTrainer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

        if cfg.debug:
            log("RUNNING IN DEBUG MODE", "red")
            # set some default config values
            cfg.num_updates = 10
            cfg.num_evals = 1
            cfg.num_eval_steps = 10

        self.rng_seq = hk.PRNGSequence(cfg.seed)
        hydra_cfg = HydraConfig.get()

        # determine if we are sweeping
        launcher = hydra_cfg.runtime["choices"]["hydra/launcher"]
        sweep = launcher in ["local", "slurm"]

        if sweep:
            self.exp_dir = Path(hydra_cfg.sweep.dir) / hydra_cfg.sweep.subdir
        else:
            self.exp_dir = Path(hydra_cfg.run.dir)

        if cfg.multistage:
            # build exp dir
            # self.exp_dir = (
            #     self.exp_dir
            #     / f"{cfg.stage.name}_s-{cfg.seed}_{cfg.env.hp_name}_{cfg.stage.hp_name}_{cfg.data.hp_name}"
            # )

            self.exp_dir = self.exp_dir / self.cfg.override_name

        log(f"launcher: {launcher}")
        log(f"experiment dir: {self.exp_dir}")

        # add exp_dir to config
        self.cfg.exp_dir = str(self.exp_dir)

        # set random seeds
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        tf.random.set_seed(cfg.seed)

        if not self.cfg.enable_jit:
            jax_config.update("jax_disable_jit", True)

        self.num_devices = jax.device_count()
        self.num_local_devices = jax.local_device_count()

        log(
            f"num_devices: {self.num_devices}, num_local_devices: {self.num_local_devices}"
        )

        if self.cfg.mode == "train":
            self.log_dir = self.exp_dir / "logs"
            self.ckpt_dir = self.exp_dir / "model_ckpts"
            self.video_dir = self.exp_dir / "videos"

            # create directories
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)
            self.video_dir.mkdir(parents=True, exist_ok=True)
            self.log_dir.mkdir(parents=True, exist_ok=True)

            # save config to yaml file
            OmegaConf.save(self.cfg, f=self.exp_dir / "config.yaml")

            wandb_name = self.cfg.wandb_name

            if self.cfg.multistage:
                wandb_name = f"{wandb_name}_{self.cfg.stage.name}"

            if self.cfg.use_wandb:
                self.wandb_run = wandb.init(
                    # set the wandb project where this run will be logged
                    entity=cfg.wandb_entity,
                    project=cfg.wandb_project,
                    name=wandb_name,
                    notes=self.cfg.wandb_notes,
                    tags=self.cfg.wandb_tags,
                    # track hyperparameters and run metadata
                    config=omegaconf_to_dict(self.cfg),
                    group=self.cfg.group_name,
                )
            else:
                self.wandb_run = None

        # create env
        log(f"creating {self.cfg.env.env_name} environments...")

        self.envs = make_envs(**self.cfg.env)
        self.cfg.env.num_envs = self.cfg.num_eval_rollouts

        if self.cfg.env.env_name == "procgen":
            # they should share the same observation space and action space
            self.obs_shape = self.envs.single_observation_space.shape
            self.continuous_actions = False
            self.input_action_dim = 1
            self.action_dim = self.envs.single_action_space.n

            try:
                self.task_dim = self.env_params.task_dim
            except:
                self.task_dim = self.cfg.env.task_dim

            log(f"task_dim: {self.task_dim}")
        elif self.cfg.env.env_name == "mujoco" or self.cfg.env.env_name == "metaworld":
            self.obs_shape = self.envs.single_observation_space.shape
            self.continuous_actions = True
            self.action_dim = self.input_action_dim = (
                self.envs.single_action_space.shape[0]
            )
        else:
            raise ValueError(f"env_name {self.cfg.env.env_name} not supported")

        log(f"obs_shape: {self.obs_shape}, action_dim: {self.action_dim}")

        if cfg.best_metric == "max":
            self.best_metric = float("-inf")
        else:
            self.best_metric = float("inf")

        # update batch size if we are using XLA
        cfg.data.batch_size = cfg.data.batch_size * cfg.num_xla_devices

        assert (
            self.cfg.stage.name in NAME_TO_STAGE_CLS
            or self.cfg.stage.name in NAME_TO_BASELINE_CLS
        ), f"Unknown task name: {self.cfg.stage.name}"
        try:
            STAGE_CLS = NAME_TO_STAGE_CLS[self.cfg.stage.name]
        except:
            STAGE_CLS = NAME_TO_BASELINE_CLS[self.cfg.stage.name]

        log(f"running {self.cfg.stage.name} training", "yellow")

        log("loading train and eval datasets", "blue")
        # load offline datasets
        # training, validation and a separate dataloader to
        # get prompt trajectories for ICL
        # val dataloaders is a dict of env_id -> dataloader
        self.train_dataloader, self.eval_dataloaders, self.prompt_dataloader = (
            load_data_tfds(cfg)
        )

        # print batch item shapes
        # determine obs_shape based on the dataset
        batch = next(self.train_dataloader.as_numpy_iterator())

        log("=" * 50)
        log("Shapes of batch items:")
        for k, v in batch.items():
            log(f"{k}: {v.shape}, {v.dtype}, {v.min()}, {v.max()}, {v.mean()}")

        observations = batch["observations"]
        if len(observations.shape) == 5:
            obs_shape = observations.shape[2:]
        elif len(observations.shape) == 4:
            obs_shape = observations.shape[1:]
        else:
            obs_shape = (observations.shape[-1],)
        log(f"observation shape: {obs_shape}")

        # figure out how many update steps between each validation step
        if self.cfg.eval_every != -1:
            self.eval_every = self.cfg.eval_every
        elif self.cfg.num_evals != -1:
            self.eval_every = int(self.cfg.num_updates // self.cfg.num_evals)
        elif self.cfg.eval_perc != -1:
            self.eval_every = int(self.cfg.num_updates * self.cfg.eval_perc)
        else:
            raise ValueError("no eval interval specified")

        log(f"evaluating model every: {self.eval_every}")

        self.stage = STAGE_CLS(
            cfg=self.cfg.stage,
            observation_shape=obs_shape,  # determine obs shape from the dataset
            action_dim=self.action_dim,
            input_action_dim=self.input_action_dim,
            continuous_actions=self.continuous_actions,
            task_dim=self.cfg.env.task_dim,
            key=next(self.rng_seq),
            num_devices=min(self.cfg.num_xla_devices, self.num_devices),
            init_kwargs=default_weight_init,
        )

    def create_ts(self):
        ts = self.stage.create_train_state(next(self.rng_seq))

        # compute number of model parameters
        log("=" * 50)
        log("finished model initialization")
        log(f"param keys: {ts.params.keys()}")
        # breakdown by components
        for k, v in ts.params.items():
            param_count = gutl.count_params(v)
            log(f"{k} parameters: {param_count}")
        log("total parameters: {}".format(gutl.count_params(ts.params)))
        return ts

    def setup_logging(self):
        pass

    def eval(self, step: int):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def save_model(self, ckpt_dict: Dict, metrics: Dict, iter: int = None):
        # use orbax?
        if self.cfg.save_key and self.cfg.save_key in metrics:
            key = self.cfg.save_key
            if (self.cfg.best_metric == "max" and metrics[key] > self.best_metric) or (
                self.cfg.best_metric == "min" and metrics[key] < self.best_metric
            ):
                self.best_metric = metrics[key]
                ckpt_file = self.ckpt_dir / "best.pkl"
                log(
                    f"new best value: {metrics[key]}, saving best model at epoch {iter} to {ckpt_file}"
                )
                with open(ckpt_file, "wb") as f:
                    pickle.dump(ckpt_dict, f)

                # create a file with the best metric in the name, use a placeholder
                best_ckpt_file = self.ckpt_dir / "best.txt"
                with open(best_ckpt_file, "w") as f:
                    f.write(f"{iter}, {metrics[key]}")

        # also save model to ckpt everytime we run evaluation
        ckpt_file = Path(self.ckpt_dir) / f"ckpt_{iter:06d}.pkl"
        log(f"saving checkpoint to {ckpt_file}")
        with open(ckpt_file, "wb") as f:
            pickle.dump(ckpt_dict, f)

    def log_to_wandb(self, metrics: Dict, prefix: str = "", step: int = None):
        if self.wandb_run is not None:
            metrics = gutl.prefix_dict_keys(metrics, prefix=prefix)
            self.wandb_run.log(metrics, step=step)

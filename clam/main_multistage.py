import sys

import jax

jax.config.update("jax_debug_nans", True)
jax.config.parse_flags_with_absl()
import os

# remove tensorflow debug messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# prevent jax from preallocating all gpu memory
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# make sure that the model training is deterministic
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

import tensorflow as tf
from loguru import logger

from prompt_dtla.utils.logger import log

tf_config = tf.compat.v1.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf.config.experimental.set_visible_devices([], "GPU")

import datetime

import hydra
from omegaconf import DictConfig, OmegaConf

import prompt_dtla.resolvers
from prompt_dtla.trainers.offline_trainer import OfflineTrainer
from prompt_dtla.trainers.offline_trainer_lam import LAMOfflineTrainer
from prompt_dtla.utils.general_utils import omegaconf_to_dict, print_dict


@hydra.main(version_base=None, config_name="multistage_config", config_path="cfg")
def main(cfg: DictConfig):
    # remove tensorflow debug messages
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    # prevent jax from preallocating all gpu memory
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    # make sure that the model training is deterministic
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

    import tensorflow as tf

    tf_config = tf.compat.v1.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf.config.experimental.set_visible_devices([], "GPU")

    log("start")
    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    # figure out the experiment directory here

    # =============== STAGE 1: Train the lam (and action decoder) ===============
    # patch up the config for the first stage of training
    stage_1_cfg = cfg.copy()
    dict = omegaconf_to_dict(stage_1_cfg)
    dict["stage"] = dict["stage1"]
    del dict["stage1"]
    del dict["stage2"]
    stage_1_cfg = OmegaConf.create(dict)

    # combine with stage1 specific updates
    stage_1_cfg = OmegaConf.merge(stage_1_cfg, cfg.stage_1_overrides)

    if cfg.debug:
        log("RUNNING IN DEBUG MODE", "red")
        stage_1_cfg.num_updates = 10
        stage_1_cfg.eval_every = 1000
        stage_1_cfg.skip_first_eval = True
        stage_1_cfg.num_eval_steps = 10

    lam_trainer = LAMOfflineTrainer(stage_1_cfg)
    lam_trainer.train()

    # ================ STAGE 2: Train the LA-BC policy ================

    # patch up the config for the second stage of training
    stage_2_cfg = cfg.copy()
    dict = omegaconf_to_dict(stage_2_cfg)
    dict["stage"] = dict["stage2"]
    del dict["stage1"]
    del dict["stage2"]
    stage_2_cfg = OmegaConf.create(dict)

    # combine with stage1 specific updates
    stage_2_cfg = OmegaConf.merge(stage_2_cfg, cfg.stage_2_overrides)

    if cfg.debug:
        stage_2_cfg.num_updates = 10
        stage_2_cfg.num_evals = 1
        stage_2_cfg.skip_first_eval = True
        stage_2_cfg.num_eval_steps = 10

    # always keep
    stage_2_cfg.stage.lam_and_decoder_ckpt = lam_trainer.exp_dir

    labc_trainer = OfflineTrainer(stage_2_cfg)
    labc_trainer.train()

    # end program
    log("end")
    sys.exit(0)


if __name__ == "__main__":
    main()

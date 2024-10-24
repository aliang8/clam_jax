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
from prompt_dtla.utils.logger import *

tf_config = tf.compat.v1.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf.config.experimental.set_visible_devices([], "GPU")

import datetime

import hydra
import prompt_dtla.resolvers
from omegaconf import DictConfig, OmegaConf
from prompt_dtla.trainers.offline_trainer import OfflineTrainer
from prompt_dtla.trainers.offline_trainer_lam import LAMOfflineTrainer
from prompt_dtla.utils.general_utils import omegaconf_to_dict, print_dict


@hydra.main(version_base=None, config_name="config", config_path="cfg")
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

    # env = submitit.JobEnvironment()
    # log(f"Process ID {os.getpid()} executing task, {env}")

    log("start")
    time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # local rank (GPU id) in a current multi-gpu mode
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    # global rank (GPU id) in multi-gpu multi-node mode
    global_rank = int(os.getenv("RANK", "0"))

    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)
    if "lam" in cfg.stage.name:
        trainer = LAMOfflineTrainer(cfg)
    else:
        trainer = OfflineTrainer(cfg)
    trainer.train()

    # end program
    log("end")
    sys.exit(0)


if __name__ == "__main__":
    main()

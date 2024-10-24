"""
Generates the commands to run experiments.
"""

import argparse
import copy
import os
import subprocess as sp
from subprocess import DEVNULL, STDOUT

parser = argparse.ArgumentParser()
parser.add_argument(
    "-r", action="store_true", help="Run the experiments (otherwise only prints)"
)
parser.add_argument(
    "-n", type=int, default=1, help="Number of duplicates of each experiments to run"
)
parser.add_argument(
    "-d", action="store_true", help="Debug mode (only run 1 experiment)"
)
parser.add_argument(
    "--seed", type=int, default=123, help="Random seed for randomness in this file"
)
parser.add_argument("--wb", action="store_true", help="Use Weights & Biases")

if __name__ == "__main__":
    args = parser.parse_args().__dict__

    stages = [
        "lam",
        "lam_and_action_decoder",
        "relabel",
        "action_decoder",
        "bc",
        "la_bc",
        "vpt",
        "vpt_bc",
    ]
    env = "mujoco"
    env_id = "halfcheetah-medium-expert-v2"
    dataset_name = "d4rl_mujoco_halfcheetah/v2-medium-expert"
    # env = "metaworld"
    # env_id = "pick-place-v2"
    # dataset_name = "metaworld/pick-place-v2"

    base_configs = {"exp_prefix": "i004"}

    lam_configs = {
        "stage/vq": "vq",
        "stage/encoder_idm": "mlp",
        "stage/encoder": "mlp",
        "stage/decoder": "mlp",
        "data": {
            "data_type": "lapo",
            "train_frac": 0.8,
            "context_len": 1,
            "dataset_name": dataset_name,
        },
        "stage.idm.apply_quantization": [False],
        "stage.vq.code_dim": [8],
        "seed": [args["seed"] + i for i in range(args["n"])],
    }
    lam_configs.update(base_configs)

    lam_and_action_decoder_configs = copy.deepcopy(lam_configs)

    lam_and_action_decoder_configs.update(
        {
            "stage.la_decoder.num_labelled_examples": [
                1_000,
                5_000,
                10_000,
                20_000,
                50_000,
            ]
        }
    )

    relabel_configs = {
        "data": {
            "num_trajs": -1,
            "batch_size": 5000,
            "dataset_name": dataset_name,
        },
        "stage.idm_ckpt": "path/to/idm/ckpt",
    }

    action_decoder_configs = {
        "data": {
            "data_type": "transitions",
            "load_latent_actions": True,
            "train_frac": 0.8,
            "num_examples": [128, 1024, 5000],
            "dataset_name": dataset_name,
        },
        "num_updates": 5_000,
        "log_terminal_every": 1000,
    }

    bc_configs = {
        "stage/encoder": "mlp",
        "data": {
            "data_type": "transitions",
            "dataset_name": dataset_name,
            "train_frac": 0.8,
            "num_trajs": [10, 50, 100],
        },
        "run_eval_rollouts": True,
        "seed": [args["seed"] + i for i in range(args["n"])],
    }

    labc_configs = {
        "stage/encoder": "mlp",
        "data": {
            "data_type": "lapo",
            "dataset_name": dataset_name,
            "train_frac": 0.8,
            "num_trajs": [200],
        },
        "run_eval_rollouts": True,
        # "decoder_ne": [128, 1024, 5000],
        # "stage.decoder_ckpt": "path/to/la_decoder/ckpt",
        "stage.lam_and_decoder_ckpt": "path/to/la_decoder/ckpt",
        "seed": [args["seed"] + i for i in range(args["n"])],
    }

    vpt_configs = {
        "data": {
            "data_type": "lapo",
            "num_trajs": [10, 50, 100],
            "dataset_name": dataset_name,
        },
        "stage/encoder": "mlp",
    }

    vpt_bc_configs = {
        "stage/encoder": "mlp",
        "data": {
            "data_type": "lapo",
            "dataset_name": dataset_name,
            "train_frac": 0.8,
            "num_trajs": [10, 50, 100],
        },
        "run_eval_rollouts": True,
        "stage.vpt_ckpt": "path/to/vpt/ckpt",
        "seed": [args["seed"] + i for i in range(args["n"])],
    }

    STAGE_TO_CFGS = {
        "lam": lam_configs,
        "lam_and_action_decoder": lam_and_action_decoder_configs,
        "relabel": relabel_configs,
        "action_decoder": action_decoder_configs,
        "bc": bc_configs,
        "la_bc": labc_configs,
        "vpt": vpt_configs,
        "vpt_bc": vpt_bc_configs,
    }

    print("List of commands: ")

    for stage in stages:
        print(f"Command for stage: {stage}")

        if stage == "relabel":
            command = f"python scripts/relabel_dataset_with_latent_actions.py \\\n\tstage=lam \\\n\tenv={env} \\\n\tenv.env_id={env_id} \\\n"

        else:
            command = f"python main.py \\\n\tstage={stage} \\\n\tenv={env} \\\n\tenv.env_id={env_id} \\\n"

        # add base configs
        for k, values in STAGE_TO_CFGS[stage].items():
            if isinstance(values, dict):
                for k_, v in values.items():
                    if isinstance(v, list):
                        if args["d"]:
                            v = v[:1]
                        command += f"\t{k}.{k_}={','.join(map(str, v))} \\\n"
                    else:
                        command += f"\t{k}.{k_}={v} \\\n"

            elif isinstance(values, list):
                if not args["r"]:
                    values = values[:1]
                command += f"\t{k}={','.join(map(str, values))} \\\n"
            else:
                command += f"\t{k}={values} \\\n"

        if args["r"] and stage != "relabel":
            command += "\tuse_wandb=True hydra/launcher=slurm --multirun"

        print(command)
        print()

    if args["r"]:
        print("All experiments COMPLETE :)\n")
    else:
        print("All experiments PRINTED :)\n")

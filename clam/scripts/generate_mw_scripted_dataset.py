from pathlib import Path

import matplotlib.pyplot as plt
import metaworld
import numpy as np
from metaworld import MT1

# from metaworld.envs import (
#     ALL_V2_ENVIRONMENTS_GOAL_HIDDEN,
#     ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
# )
from metaworld.policies import (
    SawyerAssemblyV2Policy,
    SawyerBasketballV2Policy,
    SawyerBinPickingV2Policy,
    SawyerBoxCloseV2Policy,
    SawyerButtonPressTopdownV2Policy,
    SawyerButtonPressTopdownWallV2Policy,
    SawyerButtonPressV2Policy,
    SawyerButtonPressWallV2Policy,
    SawyerCoffeeButtonV2Policy,
    SawyerCoffeePullV2Policy,
    SawyerCoffeePushV2Policy,
    SawyerDialTurnV2Policy,
    SawyerDisassembleV2Policy,
    SawyerDoorCloseV2Policy,
    SawyerDoorLockV2Policy,
    SawyerDoorOpenV2Policy,
    SawyerDoorUnlockV2Policy,
    SawyerDrawerCloseV2Policy,
    SawyerDrawerOpenV2Policy,
    SawyerFaucetCloseV2Policy,
    SawyerFaucetOpenV2Policy,
    SawyerHammerV2Policy,
    SawyerHandInsertV2Policy,
    SawyerHandlePressSideV2Policy,
    SawyerHandlePressV2Policy,
    SawyerHandlePullSideV2Policy,
    SawyerHandlePullV2Policy,
    SawyerLeverPullV2Policy,
    SawyerPegInsertionSideV2Policy,
    SawyerPegUnplugSideV2Policy,
    SawyerPickOutOfHoleV2Policy,
    SawyerPickPlaceV2Policy,
    SawyerPickPlaceWallV2Policy,
    SawyerPlateSlideBackSideV2Policy,
    SawyerPlateSlideBackV2Policy,
    SawyerPlateSlideSideV2Policy,
    SawyerPlateSlideV2Policy,
    SawyerPushBackV2Policy,
    SawyerPushV2Policy,
    SawyerPushWallV2Policy,
    SawyerReachV2Policy,
    SawyerReachWallV2Policy,
    SawyerShelfPlaceV2Policy,
    SawyerSoccerV2Policy,
    SawyerStickPullV2Policy,
    SawyerStickPushV2Policy,
    SawyerSweepIntoV2Policy,
    SawyerSweepV2Policy,
    SawyerWindowCloseV2Policy,
    SawyerWindowOpenV2Policy,
)

policies = dict(
    {
        "assembly-v2": SawyerAssemblyV2Policy,
        "basketball-v2": SawyerBasketballV2Policy,
        "bin-picking-v2": SawyerBinPickingV2Policy,
        "box-close-v2": SawyerBoxCloseV2Policy,
        "button-press-topdown-v2": SawyerButtonPressTopdownV2Policy,
        "button-press-topdown-wall-v2": SawyerButtonPressTopdownWallV2Policy,
        "button-press-v2": SawyerButtonPressV2Policy,
        "button-press-wall-v2": SawyerButtonPressWallV2Policy,
        "coffee-button-v2": SawyerCoffeeButtonV2Policy,
        "coffee-pull-v2": SawyerCoffeePullV2Policy,
        "coffee-push-v2": SawyerCoffeePushV2Policy,
        "dial-turn-v2": SawyerDialTurnV2Policy,
        "disassemble-v2": SawyerDisassembleV2Policy,
        "door-close-v2": SawyerDoorCloseV2Policy,
        "door-lock-v2": SawyerDoorLockV2Policy,
        "door-open-v2": SawyerDoorOpenV2Policy,
        "door-unlock-v2": SawyerDoorUnlockV2Policy,
        "drawer-close-v2": SawyerDrawerCloseV2Policy,
        "drawer-open-v2": SawyerDrawerOpenV2Policy,
        "faucet-close-v2": SawyerFaucetCloseV2Policy,
        "faucet-open-v2": SawyerFaucetOpenV2Policy,
        "hammer-v2": SawyerHammerV2Policy,
        "hand-insert-v2": SawyerHandInsertV2Policy,
        "handle-press-side-v2": SawyerHandlePressSideV2Policy,
        "handle-press-v2": SawyerHandlePressV2Policy,
        "handle-pull-v2": SawyerHandlePullV2Policy,
        "handle-pull-side-v2": SawyerHandlePullSideV2Policy,
        "peg-insert-side-v2": SawyerPegInsertionSideV2Policy,
        "lever-pull-v2": SawyerLeverPullV2Policy,
        "peg-unplug-side-v2": SawyerPegUnplugSideV2Policy,
        "pick-out-of-hole-v2": SawyerPickOutOfHoleV2Policy,
        "pick-place-v2": SawyerPickPlaceV2Policy,
        "pick-place-wall-v2": SawyerPickPlaceWallV2Policy,
        "plate-slide-back-side-v2": SawyerPlateSlideBackSideV2Policy,
        "plate-slide-back-v2": SawyerPlateSlideBackV2Policy,
        "plate-slide-side-v2": SawyerPlateSlideSideV2Policy,
        "plate-slide-v2": SawyerPlateSlideV2Policy,
        "reach-v2": SawyerReachV2Policy,
        "reach-wall-v2": SawyerReachWallV2Policy,
        "push-back-v2": SawyerPushBackV2Policy,
        "push-v2": SawyerPushV2Policy,
        "push-wall-v2": SawyerPushWallV2Policy,
        "shelf-place-v2": SawyerShelfPlaceV2Policy,
        "soccer-v2": SawyerSoccerV2Policy,
        "stick-pull-v2": SawyerStickPullV2Policy,
        "stick-push-v2": SawyerStickPushV2Policy,
        "sweep-into-v2": SawyerSweepIntoV2Policy,
        "sweep-v2": SawyerSweepV2Policy,
        "window-close-v2": SawyerWindowCloseV2Policy,
        "window-open-v2": SawyerWindowOpenV2Policy,
    }
)
import tensorflow as tf
import tensorflow_datasets as tfds
import tqdm

from prompt_dtla.envs.utils import make_envs
from prompt_dtla.scripts.data import save_dataset
from prompt_dtla.utils.logger import log


def trajectory_generator(env, policy, act_noise_pct, res=(64, 64), save_imgs=False):
    action_space_ptp = env.action_space.high - env.action_space.low

    env.reset()
    env.call("reset_model")
    o, info = env.reset()

    trajectory = {
        "observations": [],
        "actions": [],
        "rewards": [],
    }

    if save_imgs:
        trajectory["images"] = []

    task_success = np.zeros((o.shape[0],))

    for _ in tqdm.tqdm(range(500)):  # TODO: fix, hardcoded path length for now
        a = np.array([policy.get_action(o_) for o_ in o])
        # clamp action between -1 and 1
        a = np.clip(a, -1, 1)
        a = np.random.normal(a, act_noise_pct * action_space_ptp)

        o, r, done, terminal, info = env.step(a)
        # Camera is one of ['corner', 'topview', 'behindGripper', 'gripperPOV']

        trajectory["observations"].append(o)
        trajectory["actions"].append(a)
        trajectory["rewards"].append(r)
        # trajectory["dones"].append(done)

        if save_imgs:
            imgs = env.call("render")

            # flip this vertically
            imgs = [img[::-1, :, :] for img in imgs]

            # reshape the image
            imgs = [tf.image.resize(img, res) for img in imgs]
            trajectory["images"].append(imgs)

        success = info["success"]
        task_success = np.logical_or(task_success, success)

    # convert to numpy arrays
    trajectory = {k: np.array(v) for k, v in trajectory.items()}

    # convert to N T ...
    trajectory = {k: np.swapaxes(v, 0, 1) for k, v in trajectory.items()}

    log(
        f"Number of successful trajectories: {np.sum(task_success)} / {len(task_success)}"
    )

    # filter trajectories where the task was successful
    for k, v in trajectory.items():
        trajectory[k] = v[task_success]

    return trajectory


if __name__ == "__main__":
    data_dir = Path("/scr/shared/prompt_dtla")
    print("List of envs: ")
    # print(ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE)
    print(metaworld.ML1.ENV_NAMES)

    save_imgs = False

    num_parallel_envs = 100
    num_rollouts_per_task = 500

    num_iters = num_rollouts_per_task // num_parallel_envs

    trajectories = []
    returns = []
    lengths = []

    all_env_names = metaworld.ML1.ENV_NAMES
    all_env_names = [
        # "peg-insert-side-v2",
        # "lever-pull-v2",
        # "pick-place-v2",
        # "faucet-open-v2",
        "window-open-v2",
        "assembly-v2",
    ]

    # for env_name, env_cls in tqdm.tqdm(
    #     ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE.items(), desc="envs"
    # ):

    assert save_imgs is False, "Not implemented yet, doesn't work with parallel envs"

    for env_name in tqdm.tqdm(all_env_names, desc="envs"):
        ml1 = metaworld.ML1(env_name)
        env_cls = ml1.train_classes[env_name]
        log(f"Generating trajectories for {env_name}")

        policy = policies[env_name.replace("-goal-observable", "")]()

        for _ in tqdm.tqdm(range(num_iters), desc="iters"):
            envs = make_envs(
                num_envs=num_parallel_envs, env_name="metaworld", env_id=env_name
            )
            trajectory_data = trajectory_generator(
                envs, policy, 0.0, save_imgs=save_imgs
            )

            # convert to list of trajectories
            for idx in range(trajectory_data["observations"].shape[0]):
                trajectory = {k: v[idx] for k, v in trajectory_data.items()}
                rewards = trajectory["rewards"]
                trajectory.update(
                    {
                        "discount": np.ones_like(rewards),
                        "is_last": np.zeros_like(rewards),
                        "is_first": np.zeros_like(rewards),
                        "is_terminal": np.zeros_like(rewards),
                    }
                )

                trajectory["is_last"][-1] = 1
                trajectory["is_terminal"][-1] = 1
                trajectory["is_first"][0] = 1

                episode_return = np.sum(rewards)
                returns.append(episode_return)
                lengths.append(len(rewards))

                trajectories.append(trajectory)

        log(f"Number of trajectories: {len(trajectories)}")
        log(f"Average return: {np.mean(returns)}")
        log(f"Average length: {np.mean(lengths)}")

        save_file = Path(data_dir) / "tensorflow_datasets" / "metaworld"
        if save_imgs:
            save_file = save_file / env_name + "_images"
        else:
            save_file = save_file / env_name

        save_dataset(trajectories, save_file, env_name="metaworld", save_imgs=save_imgs)

        # save stats to a file
        with open(save_file / "dataset_stats.txt", "w") as f:
            stats = {
                "num_trajectories": len(trajectories),
                "avg_return": np.mean(returns),
                "avg_length": np.mean(lengths),
                "std_return": np.std(returns),
                "std_length": np.std(lengths),
                "min_return": np.min(returns),
                "max_return": np.max(returns),
                "min_length": np.min(lengths),
                "max_length": np.max(lengths),
            }

            for k, v in stats.items():
                f.write(f"{k}: {v}\n")

        envs.close()

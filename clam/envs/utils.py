from functools import partial

import d4rl
import numpy as np
from procgen import ProcgenEnv

ALL_PROCGEN_GAMES = [
    "bigfish",
    "bossfight",
    "caveflyer",
    "chaser",
    "climber",
    "coinrun",
    "dodgeball",
    "fruitbot",
    "heist",
    "jumper",
    "leaper",
    "maze",
    "miner",
    "ninja",
    "plunder",
    "starpilot",
]

# def normalize_return(ep_ret, env_name):
#     """normalizes returns based on URP and expert returns above"""
#     return doy.normalize_into_range(
#         lower=urp_ep_return[env_name],
#         upper=expert_ep_return[env_name],
#         v=ep_ret,
#     )

procgen_action_meanings = np.array(
    [
        "LEFT-DOWN",
        "LEFT",
        "LEFT-UP",
        "DOWN",
        "NOOP",
        "UP",
        "RIGHT-DOWN",
        "RIGHT",
        "RIGHT-UP",
        "D",
        "A",
        "W",
        "S",
        "Q",
        "E",
    ]
)


# taken from: https://github.com/schmidtdominik/LAPO/blob/main/lapo/env_utils.py
def make_procgen_envs(num_envs, env_id, gamma, **kwargs):
    import gym

    envs = ProcgenEnv(
        num_envs=num_envs,
        env_name=env_id,
        num_levels=0,
        start_level=0,
        distribution_mode="easy",
        rand_seed=0,  # TODO: set this
    )

    envs = gym.wrappers.TransformObservation(envs, lambda obs: obs["rgb"])
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space["rgb"]
    envs.is_vector_env = True
    # envs = gym.wrappers.RecordEpisodeStatistics(envs)
    envs = gym.wrappers.NormalizeReward(envs, gamma=gamma)
    envs = gym.wrappers.TransformReward(envs, lambda reward: np.clip(reward, -10, 10))

    assert isinstance(
        envs.action_space, gym.spaces.discrete.Discrete
    ), "only discrete action space is supported"

    # envs.normalize_return = partial(normalize_return, env_name=env_id)
    return envs


def make_mujoco_envs(num_envs, env_id, **kwargs):
    import gym

    # wrap multienv
    # envs = gym.vector.make(env_id, num_envs=num_envs)

    # use sync vector env
    env_fn = lambda: gym.make(env_id)
    if num_envs == 1:
        envs = gym.vector.SyncVectorEnv([env_fn])
    else:
        envs = gym.vector.AsyncVectorEnv([env_fn for _ in range(num_envs)])
    return envs


def make_metaworld_envs(num_envs, env_id, **kwargs):
    import gymnasium as gym
    import metaworld

    ml1 = metaworld.ML1(env_id)
    env_cls = ml1.train_classes[env_id]

    def env_fn(env_idx):
        # this is required to ensure that each environment
        # has a different random seed
        st0 = np.random.get_state()
        np.random.seed(env_idx)
        env = env_cls(render_mode="rgb_array", camera_name="corner")
        env._partially_observable = False
        env._freeze_rand_vec = False
        env._set_task_called = True
        env.reset()
        env._freeze_rand_vec = True
        env.seed(env_idx)
        np.random.set_state(st0)
        return env

    envs = [partial(env_fn, env_idx=i) for i in range(num_envs)]
    if num_envs == 1:
        envs = gym.vector.SyncVectorEnv(envs)
    else:
        envs = gym.vector.AsyncVectorEnv(envs)
    return envs


def make_envs(env_name: str, num_envs: int, env_id: str, **kwargs):
    if env_name == "procgen":
        envs = make_procgen_envs(num_envs=num_envs, env_id=env_id, gamma=1.0)
    elif env_name == "mujoco":
        envs = make_mujoco_envs(num_envs=num_envs, env_id=env_id)
    elif env_name == "metaworld":
        envs = make_metaworld_envs(num_envs=num_envs, env_id=env_id)

    return envs

import copy
import time
from typing import Dict, List, Tuple, Union

import cv2
import einops
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import tqdm
import wandb
from matplotlib import cm, font_manager
from omegaconf import DictConfig
from PIL import Image, ImageDraw, ImageFont

from prompt_dtla.envs.utils import make_envs, procgen_action_meanings
from prompt_dtla.utils.data_utils import (
    Transition,
    normalize_obs,
    unnormalize_obs,
)
from prompt_dtla.utils.logger import log
from prompt_dtla.utils.training import TrainState


def annotate_videos(
    env_name: str,
    videos: np.ndarray,
    meta: Dict,
    img_size: Tuple[int, int] = (256, 256),
    label: str = "",
):
    annotated_videos = []
    for video_idx, video in enumerate(videos):
        # add a step index to the meta
        meta_ = {"S": np.arange(video.shape[0]), "Ret": np.cumsum(meta["R"][video_idx])}
        meta_.update({k: v[video_idx] for k, v in meta.items()})

        annotated_video = annotate_single_video(
            env_name, video=video, meta=meta_, img_size=img_size, label=label
        )
        annotated_videos.append(annotated_video)

    return np.array(annotated_videos)


def annotate_single_video(
    env_name: str,
    video: np.ndarray,
    meta: Dict,
    img_size: Tuple[int, int] = (256, 256),
    label: str = "",
):
    # load a nice big readable font
    font = font_manager.FontProperties(family="sans-serif", weight="bold")
    file = font_manager.findfont(font)
    font = ImageFont.truetype(file, 12)

    annotated_imgs = []
    for step, frame in enumerate(video):
        frame = Image.fromarray(frame)
        frame = frame.resize(img_size)

        # add border on top of image
        extra_border_height = 100
        annotated_img = Image.new(
            "RGB",
            (frame.width, frame.height + extra_border_height),
            color=(255, 255, 255),
        )
        annotated_img.paste(frame, (0, extra_border_height))
        draw = ImageDraw.Draw(annotated_img)

        count = 0
        lines = []
        to_display = ""
        num_keys_per_line = 2

        for key, values in meta.items():
            if isinstance(values[step], np.ndarray):
                values = np.round(values[step], 2)
                to_add = f"{key}: {values}  "
            elif isinstance(values[step], float) or isinstance(
                values[step], np.float32
            ):
                to_add = f"{key}: {values[step]:.2f}  "
            else:
                to_add = f"{key}: {values[step]}  "

            if count < num_keys_per_line:
                to_display += to_add
                count += 1
            else:
                lines.append(to_display)
                count = 1
                to_display = to_add

        # add the last line
        if lines:
            lines.append(to_display)

        for i, line in enumerate(lines):
            # make font size bigger
            draw.text((10, 10 + i * 20), line, fill="black", font=font)

        # convert to numpy array
        annotated_imgs.append(np.array(annotated_img))

    return np.array(annotated_imgs)


def run_rollouts(
    rng: jax.random.PRNGKey,
    agent_ts,
    cfg: DictConfig,
    env_id: str,
    action_dim: int,
    wandb_run=None,
    prompt: Dict = None,
    fps: int = 5,
    log_videos: bool = False,
    **kwargs,
):
    """
    Run evaluation rollouts in the Procgen environment
    """
    # update env_id in config
    cfg.env.env_id = env_id

    rollout_kwargs = dict(
        cfg=cfg,
        rng=rng,
        agent_ts=agent_ts,
        action_dim=action_dim,
        env_id=env_id,
        log_videos=log_videos,
        **kwargs,
    )

    if "dt" in cfg.stage.name or "icl" in cfg.stage.name:
        rollout_fn = run_rollouts_dt
    else:
        rollout_fn = run_rollouts_helper

    if prompt is not None:
        log("using prompt")
        rollout_kwargs["prompt"] = prompt

    rollout_time = time.time()
    eval_metrics, transitions = rollout_fn(**rollout_kwargs)
    rollout_time = time.time() - rollout_time

    eval_metrics["time"] = rollout_time / cfg.num_eval_rollouts

    # compute normalized returns if d4rl mujoco env
    if cfg.env.env_name == "mujoco":
        # create a dummy env
        env_cfg = copy.deepcopy(cfg.env)
        env_cfg.num_envs = 1
        dummy_env = make_envs(training=False, **env_cfg)
        normalized_return = dummy_env.envs[0].get_normalized_score(
            eval_metrics["ep_ret"]
        )
        eval_metrics["normalized_ret"] = normalized_return

    # flatten list of transitions
    transitions = jtu.tree_map(lambda *x: np.stack(x), *transitions)

    # stack list of dicts
    if cfg.env.env_name != "metaworld":
        stacked_infos = jtu.tree_map(lambda *x: np.stack(x, axis=-1), *transitions.info)
        object.__setattr__(transitions, "info", stacked_infos)

    # swap first two axes
    transitions = jtu.tree_map(lambda x: np.swapaxes(x, 0, 1), transitions)

    # for some environments, we need to run it sequentially to generate the videos
    # this is just for the rendering rollout videos
    if log_videos and cfg.env.env_name == "metaworld":
        transitions = generate_videos(cfg=cfg, rng=rng, agent_ts=agent_ts)

    if log_videos:
        log("saving evaluation videos to wandb...")
        videos = transitions.image
        videos = videos[: cfg.num_eval_rollouts_render]
        rewards = transitions.reward
        actions = transitions.action
        dones = transitions.done
        infos = transitions.info

        meta = {
            "R": rewards,
            "D": dones,
            "A": actions,
        }

        # select keys from the environment infos to log in our rollout video
        if cfg.env.env_name == "metaworld":
            env_meta = {
                "Grasp": infos["grasp_success"].squeeze().astype(int),
                "Dist": infos["obj_to_target"].squeeze(),
                "SUC": infos["success"].squeeze().astype(int),
            }
        elif cfg.env.env_name == "mujoco":
            env_meta = {
                "Rew_ctrl": infos["reward_ctrl"].squeeze(),
                "Rew_run": infos["reward_run"].squeeze(),
            }
        else:
            env_meta = {}

        meta.update(env_meta)

        rollout_videos = annotate_videos(
            cfg.env.env_name, videos, meta, label="rollout"
        )

        # also make prompts into videos if exist
        if prompt is not None:
            prompt_cpy = prompt.copy()
            # should be (N, T, H, W, C) shape, 1:1 mapping to the rollout videos

            def truncate(x):
                return x[
                    : cfg.num_eval_rollouts_render,
                    : cfg.env.max_episode_steps,
                ]

            prompt_cpy = jtu.tree_map(truncate, prompt_cpy)

            observations = unnormalize_obs(prompt_cpy["observations"])
            import ipdb

            ipdb.set_trace()
            prompt_videos = annotate_video(
                cfg.env.env_name,
                observations,
                prompt_cpy["rewards"],
                prompt_cpy["actions"],
                prompt_cpy["is_terminal"],
                label="prompt",
            )

            if prompt_videos.shape != rollout_videos.shape:
                if prompt_videos.shape[1] < rollout_videos.shape[1]:
                    # pad prompt videos with zeros
                    pad_len = rollout_videos.shape[1] - prompt_videos.shape[1]
                    pad = np.zeros(
                        (
                            cfg.num_eval_rollouts_render,
                            pad_len,
                            *prompt_videos.shape[2:],
                        )
                    )
                    prompt_videos = np.concatenate([prompt_videos, pad], axis=1)
                elif prompt_videos.shape[1] > rollout_videos.shape[1]:
                    # pad rollout videos with zeros
                    pad_len = prompt_videos.shape[1] - rollout_videos.shape[1]
                    pad = np.zeros(
                        (
                            cfg.num_eval_rollouts_render,
                            pad_len,
                            *rollout_videos.shape[2:],
                        )
                    )
                    rollout_videos = np.concatenate([rollout_videos, pad], axis=1)

            # stack this with rollouts, (N, T, H, W, C) -> (2, N, T, H, W, C)
            rollout_videos = np.stack([prompt_videos, rollout_videos], axis=0)

            rollout_videos = einops.rearrange(
                rollout_videos, "p n t h w c -> (n p) t h w c"
            )

        # split into even chunks
        rollout_videos = np.array_split(rollout_videos, 2)

        if wandb_run is not None:
            for chunk, video in enumerate(rollout_videos):
                video = einops.rearrange(video, "n t h w c -> n t c h w")
                wandb_run.log(
                    {f"{env_id}/rollout/{chunk}": wandb.Video(video, fps=fps)}
                )

    return eval_metrics, transitions


def generate_videos(
    cfg: DictConfig,
    rng: jax.random.PRNGKey,
    agent_ts: TrainState,
):
    log("generate video rollouts...", color="green")

    # need to generate videos sequentially
    cfg = copy.deepcopy(cfg)
    cfg.env.num_envs = 1
    env = make_envs(training=False, **cfg.env)

    jit_apply = jax.jit(agent_ts.apply_fn, static_argnames=("is_training"))

    rollouts = []

    for _ in tqdm.tqdm(range(cfg.num_eval_rollouts_render), desc="generating videos"):
        if cfg.env.env_name == "metaworld":
            obs, info = env.reset()
        else:
            obs = env.reset()

        rollout = {
            "observation": [],
            "action": [],
            "reward": [],
            "done": [],
            "image": [],
            "info": [],
        }
        while True:
            rng, policy_rng, dropout_rng = jax.random.split(rng, 3)

            policy_output = jit_apply(
                {"params": agent_ts.params, **agent_ts.mparams},
                x=obs,
                is_training=False,
                rngs={"sample": policy_rng, "dropout": dropout_rng},
            )
            action = np.array(policy_output.action)

            if cfg.env.env_name == "metaworld":
                obs, reward, done, terminate, info = env.step(action)
                done = done or terminate
            else:
                obs, reward, done, info = env.step(action)

            if done:
                break

            if cfg.env.env_name == "metaworld":
                image = env.call("render")[0][None]  # cause we are using SyncVectorEnv
            else:
                image = env.render()
            image = np.array(image)

            if cfg.env.env_name == "metaworld":
                # flip this vertically
                image = np.flipud(image)
                image = np.fliplr(image)

            rollout["observation"].append(obs)
            rollout["action"].append(action)
            rollout["reward"].append(reward)
            rollout["done"].append(done)
            rollout["image"].append(image)
            rollout["info"].append(info)

        # pad the transitions to the max length, just copy the last transition
        max_len = cfg.env.max_episode_steps
        num_steps = len(rollout["observation"])
        if num_steps < max_len:
            pad_len = max_len - num_steps

            # pad the transitions
            for _ in range(pad_len):
                for k in rollout.keys():
                    rollout[k].append(rollout[k][-1])

        for k in rollout.keys():
            if k == "info":
                continue
            rollout[k] = np.concatenate(rollout[k])

        # flatten the keys in info and make each key a list
        new_infos = {}
        for k in rollout["info"][-1].keys():
            for info in rollout["info"]:
                if not info:  # stupid metaworld issue
                    info = rollout["info"][-1]

                if k not in new_infos:
                    new_infos[k] = []
                new_infos[k].append(info[k])

            new_infos[k] = np.array(new_infos[k])

        rollout["info"] = new_infos
        rollouts.append(rollout)

    # stack the rollouts together
    stacked_rollouts = {}

    for k in rollouts[0].keys():
        if (
            type(rollouts[0][k]) is dict
        ):  # need this to handle the infos which is a dict
            stacked_rollouts[k] = {}
            for k2 in rollouts[0][k].keys():
                stacked_rollouts[k][k2] = np.stack([r[k][k2] for r in rollouts])
        else:
            stacked_rollouts[k] = np.stack([r[k] for r in rollouts])

    stacked_rollouts = Transition(**stacked_rollouts)
    return stacked_rollouts


def run_rollouts_helper(
    cfg: DictConfig,
    rng: jax.random.PRNGKey,
    agent_ts: TrainState,
    log_videos: bool = False,
    **kwargs,
) -> Union[Dict[str, jnp.ndarray], List[Transition]]:
    """
    Return:
        eval_metrics: Dict of evaluation metrics
        transitions: List of transitions
    """
    # gym version needs to be gym==0.23.1 for this to work
    log("running rollouts...", color="green")
    envs = make_envs(training=False, **cfg.env)
    obs = envs.reset()

    if cfg.env.env_name == "metaworld":
        obs, info = obs

    curr_timestep = 0
    dones = np.zeros((cfg.num_eval_rollouts,))
    ep_returns = np.zeros((cfg.num_eval_rollouts,))
    ep_lengths = np.zeros((cfg.num_eval_rollouts,))
    ep_success = np.zeros((cfg.num_eval_rollouts,))
    transitions = []

    jit_apply = jax.jit(agent_ts.apply_fn, static_argnames=("is_training"))

    while not np.all(dones):
        # break after max timesteps
        if curr_timestep >= cfg.stage.max_timesteps:
            break

        if cfg.env.env_name == "procgen":
            obs = normalize_obs(obs.astype(jnp.float32))

        rng, policy_rng, dropout_rng = jax.random.split(rng, 3)

        # run forward pass to get the action
        policy_output = jit_apply(
            {"params": agent_ts.params, **agent_ts.mparams},
            x=obs,
            is_training=False,
            rngs={"sample": policy_rng, "dropout": dropout_rng},
        )
        action = np.array(policy_output.action)

        # step in the environment
        if cfg.env.env_name == "metaworld":
            obs, reward, done, _, info = envs.step(action)
        else:
            obs, reward, done, info = envs.step(action)

        # update episode returns and lengths
        dones = np.logical_or(dones, done)
        ep_rew = reward * (1 - dones)
        step = np.ones_like(dones) * (1 - dones)
        ep_returns += ep_rew
        ep_lengths += step

        if cfg.env.env_name == "metaworld":
            if "final_info" in info:
                info = transitions[-1].info  # use the last info
            else:
                ep_success = np.logical_or(ep_success, info["success"])

        # generate image frames for the video
        # metaworld render doesn't work with AsyncVectorEnvs for some reason
        if log_videos and cfg.env.env_name != "metaworld":
            if cfg.env.env_name == "procgen":
                image = obs
            else:
                # need to render each env separately
                image = envs.call("render", mode="rgb_array")
                # tuple to array
                image = np.array(image)
        else:
            image = None

        # fix the last info timestep for mujoco hc because DMC env returns
        # extra keys in the info dict
        if cfg.env.env_name == "mujoco" and "terminal_observation" in info[0]:
            for i in range(cfg.num_eval_rollouts):
                info[i].pop("terminal_observation")
                info[i].pop("TimeLimit.truncated")

        transition = Transition(
            observation=obs,
            action=action,
            reward=reward,
            done=dones,
            image=image,
            info=info,
        )

        transitions.append(transition)

        curr_timestep += 1

    eval_metrics = {
        "ep_ret": jnp.mean(ep_returns),
        "ep_ret_std": jnp.std(ep_returns),
        "avg_len": jnp.mean(ep_lengths),
        "std_len": jnp.std(ep_lengths),
        "max_len": jnp.max(ep_lengths),
        "min_len": jnp.min(ep_lengths),
        "success_rate": jnp.mean(ep_success),
        "success_rate_std": jnp.std(ep_success),
    }
    return eval_metrics, transitions


def populate_prompt(cfg: DictConfig, traj_history: Dict, prompt: Dict = None):
    if prompt is None:
        return traj_history, 0

    # make a copy of prompt
    prompt_cpy = prompt.copy()

    start_index = 0
    if cfg.stage.model.task_conditioning:
        import ipdb

        ipdb.set_trace()
        task = task.reshape(1, 1, -1)
        prompt = task
    elif cfg.data.num_few_shot_prompts > 0:
        assert prompt is not None
        log("adding prompt info before trajectory")

        # # prepend prompt into the input tokens
        # prompt_steps = int(prompt["mask"].sum())
        prompt_steps = prompt["mask"].shape[1]

        # # filter prompt for valid steps in case we do some padding
        # filtered_prompt = jtu.tree_map(lambda x: x[:prompt_steps], prompt)

        # TODO: might add this back if the traj length gets too long
        # sample a couple steps from the prompt
        # si = np.random.randint(0, prompt_steps - cfg.stage.context_window)
        # filtered_prompt = jtu.tree_map(
        #     lambda x: x[si : si + cfg.stage.context_window],
        #     filtered_prompt,
        # )

        # prepend prompt to each key in our trajectory history
        prompt_cpy["actions"] = prompt_cpy["actions"][..., None]
        prompt_cpy["rewards"] = prompt_cpy["rewards"][..., None]
        for key in traj_history.keys():
            if key in prompt_cpy and traj_history[key] is not None:
                traj_history[key][:, :prompt_steps] = prompt_cpy[key]

        # different keys
        traj_history["states"][:, :prompt_steps] = prompt_cpy["observations"]

        # fix timesteps
        timesteps = traj_history["timestep"].shape[1]
        remaining_timesteps = timesteps - prompt_steps
        traj_history["timestep"][:, prompt_steps:] = np.arange(remaining_timesteps)

        if "lam" in cfg.stage.name:
            # label latent actions if ground truth actions not provided
            pass

            # rng, latent_rng = jax.random.split(rng)
            # # relabel demo trajectory with latent actions
            # latent_actions = jit_relabel(
            #     latent_rng,
            #     prompt["states"][jnp.newaxis].astype(jnp.float32),
            # )
            # # TODO: might be an index issue here
            # latent_actions = latent_actions[:, :prompt_steps]
            # actions[:, :prompt_steps] = latent_actions

        traj_index = np.zeros(traj_history["mask"].shape)
        if "traj_index" in prompt:
            traj_index[:, :prompt_steps] = prompt_cpy["traj_index"][:prompt_steps]
        traj_index = np.array(traj_index)
        traj_history["traj_index"] = traj_index

        start_index = prompt_steps
    else:
        traj_index = None

    return traj_history, start_index


def run_rollouts_dt(
    rng,
    env,
    agent_ts: TrainState,
    cfg: DictConfig,
    action_dim: int,
    prompt: Dict = None,
    **kwargs,
):
    # gym version needs to be gym==0.23.1 for this to work
    log("rollout procgen dt...")
    env = make_procgen_envs(training=False, **cfg.env)
    obs = env.reset()

    jit_apply = jax.jit(agent_ts.apply_fn, static_argnames=("is_training"))

    N = cfg.num_eval_rollouts
    dones = jnp.zeros((N,))
    ep_returns = np.zeros((N,))
    ep_lengths = np.zeros((N,))

    # first dimension is batch
    observation_shape = obs.shape[1:]
    max_traj_len = cfg.stage.model.max_timesteps
    prompt_len = prompt["mask"].shape[1] if prompt is not None else 0
    max_traj_len += prompt_len
    timesteps = np.arange(max_traj_len)
    timesteps = einops.repeat(timesteps, "t -> n t", n=N)
    traj_history = {
        "states": np.zeros((N, max_traj_len, *observation_shape)),
        "actions": np.zeros((N, max_traj_len, action_dim)),
        "rewards": np.zeros((N, max_traj_len, 1)),
        "mask": np.zeros((N, max_traj_len)),
        "timestep": timesteps,
        "traj_index": None,
        "prompt": None,
    }

    traj_history, start_index = populate_prompt(cfg, traj_history, prompt)
    curr_index = start_index  # index for prompt + current trajectory
    curr_timestep = 0  # timestep in the current trajectory

    # set the first observation and mask
    # first normalize the observation
    obs = normalize_obs(obs.astype(jnp.float32))
    traj_history["states"][:, start_index] = obs
    traj_history["mask"][:, start_index] = 1.0
    if traj_history["traj_index"] is not None:
        next_traj_index = traj_history["traj_index"].max() + 1
        traj_history["traj_index"][:, start_index:] = next_traj_index

    transitions = []

    while not jnp.all(dones):
        if curr_timestep >= cfg.stage.model.max_timesteps:
            break

        # make sure we are getting new rngs each step
        rng, policy_rng, dropout_rng, model_rng = jax.random.split(rng, 4)
        keys = {
            "sample": policy_rng,
            "dropout": dropout_rng,
            "model": model_rng,
        }

        # todo get context window of K steps, check the indexing
        si = max(0, curr_index - cfg.stage.context_window)
        ei = max(curr_index, cfg.stage.context_window)
        traj_history_context = jtu.tree_map(lambda x: x[:, si:ei], traj_history)

        policy_output = jit_apply(
            {"params": agent_ts.params, **agent_ts.mparams},
            **traj_history_context,
            is_training=False,
            rngs=keys,
        )

        # take the final action predicted by the Transformer
        action = policy_output.action[:, curr_index]
        obs, reward, done, _ = env.step(action)

        # normalize obs
        obs_norm = normalize_obs(obs.astype(jnp.float32))

        # add transition to the trajectory
        traj_history["actions"][:, curr_index] = action.reshape(-1, action_dim)
        traj_history["rewards"][:, curr_index] = reward.reshape(-1, 1)

        if curr_index + 1 < max_traj_len:
            traj_history["states"][:, curr_index + 1] = obs_norm
            traj_history["mask"][:, curr_index + 1] = 1.0

        for i in range(cfg.num_eval_rollouts):
            if not dones[i]:
                ep_returns[i] = ep_returns[i] + reward[i]
                ep_lengths[i] = ep_lengths[i] + 1
            else:
                dones = dones.at[i].set(1)

        dones = jnp.logical_or(dones, done)

        transition = Transition(
            observation=obs,
            action=action,
            reward=reward,
            done=dones,
        )
        transitions.append(transition)

        curr_index += 1
        curr_timestep += 1

    eval_metrics = {
        "episode_return": jnp.mean(ep_returns),
        "avg_len": jnp.mean(ep_lengths),
    }
    return eval_metrics, transitions

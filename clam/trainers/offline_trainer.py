import time
from collections import defaultdict
from functools import partial

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import tqdm
from omegaconf import DictConfig
from rich.pretty import pretty_repr

import prompt_dtla.utils.general_utils as gutl
from prompt_dtla.baselines import NAME_TO_BASELINE_CLS
from prompt_dtla.stage import NAME_TO_STAGE_CLS
from prompt_dtla.trainers.base_offline_trainer import BaseOfflineTrainer
from prompt_dtla.utils.data_utils import Batch
from prompt_dtla.utils.logger import log
from prompt_dtla.utils.rollouts import run_rollouts
from prompt_dtla.utils.tfds_data_utils import load_data as load_data_tfds
from prompt_dtla.utils.training import default_weight_init, save_dict


class OfflineTrainer(BaseOfflineTrainer):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        self.update_jit = jax.jit(
            self.stage.update, static_argnames=("is_training", "labelled")
        )

        # sample prompts for in-context evaluation
        self.prompt = None
        if self.cfg.data.num_few_shot_prompts > 0:
            self.prompt = self.sample_prompt()

    def train(self):
        ts = self.create_ts()

        # first eval
        if not self.cfg.skip_first_eval:
            eval_metrics = self.eval(ts, step=0)

        train_iter = self.train_dataloader.repeat().as_numpy_iterator()

        for train_step in tqdm.tqdm(
            range(self.cfg.num_updates),
            desc=f"{self.cfg.stage.name} train batches",
            disable=False,
            total=self.cfg.num_updates,
        ):
            batch_load_time = time.time()
            batch = next(train_iter)
            batch_load_time = time.time() - batch_load_time
            batch = Batch(**batch)

            # perform a single gradient step
            update_time = time.time()
            ts, metrics, extra = self.update_jit(
                next(self.rng_seq), ts=ts, batch=batch, is_training=True
            )
            metrics["lr"] = ts.opt_state.hyperparams["lr"].item()
            metrics["time/batch_load"] = batch_load_time
            metrics["time/update"] = time.time() - update_time
            self.log_to_wandb(metrics, prefix="train/")

            # log the gradient values
            if "grads" in extra:
                grad_stats = gutl.log_pytree_stats(extra["grads"])
                self.log_to_wandb(grad_stats, prefix="grads/")

            # log stats about the model params
            param_stats = gutl.log_pytree_stats(ts.params)
            self.log_to_wandb(param_stats, prefix="params/")

            # log a step counter for wandb
            self.log_to_wandb({"_update": train_step}, prefix="step/")

            # run evaluation for each evaluation environment
            if ((train_step + 1) % self.eval_every) == 0:
                self.eval(ts, step=train_step + 1)

            # log to terminal
            if ((train_step + 1) % self.cfg.log_terminal_every) == 0:
                metrics_print = jtu.tree_map(lambda x: np.round(float(x), 5), metrics)
                log(f"step: {train_step}, train:")
                log(f"{pretty_repr(metrics_print)}")

        self.eval(ts, step=self.cfg.num_updates)

        if self.wandb_run is not None:
            self.wandb_run.finish()

    def eval(self, ts, step: int):
        log("running evaluation", "blue")

        for eval_env_id in self.eval_dataloaders.keys():
            eval_metrics = self.eval_single_env(ts, step, eval_env_id)

        # save model for the training environment
        if self.cfg.mode == "train":
            self.save_model(
                save_dict(ts, self.cfg, self.stage.num_devices),
                eval_metrics,
                step,
            )

        return

    def eval_single_env(self, ts, step: int, eval_env_id: str):
        """
        Run evaluation for data from a specific environment
        Also run evaluation rollouts for policy training
        """
        log(f"\n\nrunning evaluation for {eval_env_id}")
        eval_metrics = defaultdict(list)

        # run on eval batches
        if self.eval_dataloaders is not None:
            eval_time = time.time()
            eval_dataloader = self.eval_dataloaders[eval_env_id]
            eval_iter = eval_dataloader.as_numpy_iterator()

            for batch in tqdm.tqdm(
                eval_iter,
                desc=f"{self.cfg.stage.name} eval batches",
            ):
                batch = Batch(**batch)

                _, metrics, extra = self.update_jit(
                    next(self.rng_seq), ts=ts, batch=batch, is_training=False
                )

                for k, v in metrics.items():
                    # make sure it is scalar
                    if not v or not v.ndim == 0:
                        continue  # skip non-scalar metrics
                    eval_metrics[k].append(v)

            # average metrics over all eval batches
            for k, v in eval_metrics.items():
                eval_metrics[k] = jnp.mean(jnp.array(v))

            eval_metrics["time"] = time.time() - eval_time

        self.log_to_wandb(eval_metrics, prefix=f"eval/{eval_env_id}/")

        # write evaluation metrics to log file
        log_eval_metrics = dict(
            jtu.tree_map(lambda x: np.round(float(x), 5), eval_metrics)
        )

        with open(self.log_dir / f"eval_{eval_env_id}.txt", "a+") as f:
            f.write(f"{step}, {log_eval_metrics}\n")

        log(f"eval [{eval_env_id}]: {pretty_repr(log_eval_metrics)}")

        # run evaluation rollouts
        if self.cfg.run_eval_rollouts:
            if self.cfg.env.env_name == "atari":
                rollout_fn = run_rollouts_atari
            else:
                rollout_fn = run_rollouts

            if (
                self.cfg.data.num_few_shot_prompts > 0
                and self.cfg.data.resample_prompt_every_eval
            ):
                # sample new prompts for evaluation
                new_prompt = self.sample_prompt()
                self.prompt = new_prompt[eval_env_id]

            rollout_metrics, *_ = rollout_fn(
                cfg=self.cfg,
                rng=next(self.rng_seq),
                agent_ts=ts,
                env_id=eval_env_id,
                action_dim=self.stage.input_action_dim,
                wandb_run=self.wandb_run,
                prompt=self.prompt,
                log_videos=self.cfg.log_rollout_videos,
                fps=self.cfg.video_fps,
            )
            self.log_to_wandb(rollout_metrics, prefix=f"eval_rollout/{eval_env_id}/")

            log_rollout_metrics = dict(
                jtu.tree_map(lambda x: np.round(float(x), 5), rollout_metrics)
            )

            with open(self.log_dir / f"eval_{eval_env_id}.txt", "a+") as f:
                f.write(f"{step}, {log_rollout_metrics}\n")

            log(f"eval rollout [{eval_env_id}]: {pretty_repr(log_rollout_metrics)}")

        return eval_metrics

    def sample_prompt(self):
        """
        Sample a new set of prompts for evaluation
        """
        assert self.prompt_dataloader is not None

        prompt = {}
        log("sampling new prompt for evaluation")
        for eval_env_id, prompt_dl in self.prompt_dataloader.items():
            prompt_iter = prompt_dl.as_numpy_iterator()
            prompt[eval_env_id] = next(prompt_iter)

        return prompt

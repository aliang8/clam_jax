import copy
import math
import time
from collections import defaultdict
from functools import partial

import einops
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import umap
import wandb
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from omegaconf import DictConfig
from rich.pretty import pretty_repr
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import prompt_dtla.utils.general_utils as gutl
from prompt_dtla.stage import NAME_TO_STAGE_CLS
from prompt_dtla.trainers.base_offline_trainer import BaseOfflineTrainer
from prompt_dtla.utils.data_utils import Batch, get_latent_action_dim
from prompt_dtla.utils.logger import log
from prompt_dtla.utils.tfds_data_utils import load_data as load_data_tfds
from prompt_dtla.utils.training import default_weight_init, save_dict
from prompt_dtla.utils.visualization import custom_to_pil, make_image_grid


class LAMOfflineTrainer(BaseOfflineTrainer):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        # load data for action decoder
        # make copy of config
        log("loading labelled data for training action decoder")
        cfg_cpy = copy.deepcopy(cfg)
        # make sure i have enough trajs here for getting labelled data
        cfg_cpy.data.num_trajs = 100
        cfg_cpy.data.num_examples = cfg_cpy.data.batch_size = (
            cfg.stage.la_decoder.num_labelled_examples
        )
        self.labelled_dataloader, *_ = load_data_tfds(config=cfg_cpy)
        self.labelled_data = next(self.labelled_dataloader.as_numpy_iterator())

        log("Shapes of labelled data items:")
        for k, v in self.labelled_data.items():
            log(f"{k}: {v.shape}, {v.dtype}, {v.min()}, {v.max()}, {v.mean()}")
        log("=" * 50, "blue")

        self.update_jit = jax.jit(
            self.stage.update, static_argnames=("is_training", "labelled")
        )

        # sample prompts for in-context evaluation
        self.prompt = None
        if self.cfg.data.num_few_shot_prompts > 0:
            self.prompt = self.sample_prompt()

    def train(self):
        ts = self.create_ts()

        # first eval without any training
        if not self.cfg.skip_first_eval:
            eval_metrics = self.eval(ts, step=0)
            self.eval_action_decoder(ts_lam=ts, step=0)

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
            metrics["time/batch_update"] = time.time() - update_time
            self.log_to_wandb(metrics, prefix="train/lam/")

            # log the gradient values
            if "grads" in extra:
                grad_stats = gutl.log_pytree_stats(extra["grads"])
                self.log_to_wandb(grad_stats, prefix="grads/")

            # log stats about the model params
            param_stats = gutl.log_pytree_stats(ts.params)
            self.log_to_wandb(param_stats, prefix="params/")

            # log a step counter for wandb
            self.log_to_wandb({"_update": train_step}, prefix="step/")

            # for LAM, sometimes we also train on the labelled
            if self.cfg.stage.name == "lam_and_action_decoder" and (
                (train_step + 1) % self.cfg.stage.train_action_decoder_every == 0
            ):
                # train on the labelled data for a couple of epochs
                decoder_train_start = time.time()

                # make mini batches of data from the labelled data

                num_labelled_exs = self.labelled_data["observations"].shape[0]
                batch_size = self.cfg.stage.la_decoder.batch_size
                num_batches = math.ceil(num_labelled_exs / batch_size)

                for _ in range(self.cfg.stage.update_steps_on_labelled_batch):
                    # split into minibatches
                    indices = np.arange(num_labelled_exs)
                    np.random.shuffle(indices)
                    batch_indices = np.array_split(indices, num_batches)

                    for batch_inds in batch_indices:
                        batch = {
                            k: v[batch_inds] for k, v in self.labelled_data.items()
                        }
                        batch = Batch(**batch)

                        ts, metrics, _ = self.update_jit(
                            next(self.rng_seq),
                            ts=ts,
                            batch=batch,
                            is_training=True,
                            labelled=True,
                        )

                metrics["la_decoder/time"] = time.time() - decoder_train_start
                self.log_to_wandb(metrics, prefix="train/la_decoder/")

            # run evaluation for each evaluation environment
            if ((train_step + 1) % self.eval_every) == 0:
                eval_metrics = self.eval(ts, step=train_step + 1)

                for eval_env_id, env_eval_metrics in eval_metrics.items():
                    self.log_to_wandb(env_eval_metrics, prefix=f"eval/{eval_env_id}/")

                # visualize some of the next observation predictions
                if (
                    self.wandb_run is not None
                    and "next_obs_pred" in extra
                    and self.cfg.env.image_obs
                ):
                    visualize_time = time.time()
                    self.visualize("train", batch, extra, self.cfg.env.env_id)
                    visualize_time = time.time() - visualize_time
                    self.log_to_wandb({"time/visualize": visualize_time})

                # evaluate action decoding accuracy
                self.eval_action_decoder(ts_lam=ts, step=train_step)

            # log to terminal
            if ((train_step + 1) % self.cfg.log_terminal_every) == 0:
                metrics_print = jtu.tree_map(lambda x: np.round(float(x), 5), metrics)
                log(f"step: {train_step}")
                log(f"{pretty_repr(metrics_print)}")

        self.eval(ts, step=self.cfg.num_updates)

        if self.wandb_run is not None:
            self.wandb_run.finish()

    def eval(self, ts, step: int):
        log("running evaluation", "blue")
        metrics = {}

        for eval_env_id in self.eval_dataloaders.keys():
            eval_start = time.time()
            eval_metrics = self.eval_single_env(ts, eval_env_id)
            eval_time = time.time() - eval_start
            eval_metrics["time"] = eval_time
            metrics[eval_env_id] = eval_metrics

            # write evaluation metrics to log file
            log_eval_metrics = dict(
                jtu.tree_map(lambda x: np.round(float(x), 2), eval_metrics)
            )

            with open(self.log_dir / f"eval_{eval_env_id}.txt", "a+") as f:
                f.write(f"{step}, {log_eval_metrics}\n")

            log(f"eval [{eval_env_id}]: {pretty_repr(log_eval_metrics)}")

        # save model for the training environment
        if self.cfg.mode == "train":
            self.save_model(
                save_dict(ts, self.cfg, self.stage.num_devices),
                eval_metrics,
                step,
            )

        return metrics

    def eval_single_env(self, ts, eval_env_id: str):
        """
        Run evaluation for data from a specific environment
        Also run evaluation rollouts for policy training
        """
        log(f"\n\nrunning evaluation for {eval_env_id}")
        eval_metrics = defaultdict(list)

        # run on eval batches
        if self.eval_dataloaders is not None:
            eval_dataloader = self.eval_dataloaders[eval_env_id]
            eval_iter = eval_dataloader.repeat().as_numpy_iterator()

            # collect some latent actions prequantized to visualize latent space for
            latent_actions = []
            images = []  # also collect the images for visualization

            for eval_step in tqdm.tqdm(
                range(self.cfg.num_eval_steps),
                desc=f"{self.cfg.stage.name} eval batches",
            ):
                batch = next(eval_iter)
                batch = Batch(**batch)

                _, metrics, extra = self.update_jit(
                    next(self.rng_seq), ts=ts, batch=batch, is_training=False
                )

                for k, v in metrics.items():
                    # make sure it is scalar
                    if not v or not v.ndim == 0:
                        continue  # skip non-scalar metrics
                    eval_metrics[k].append(v)

                if eval_step == 0 and self.cfg.env.image_obs:
                    # visualize some of the next observation predictions
                    self.visualize("eval", batch, extra, eval_env_id)

                if eval_step < 50:
                    latent_actions.extend(extra["latent_actions"])
                    if batch.images is not None:
                        images.extend(batch.images)

            # average metrics over all eval batches
            for k, v in eval_metrics.items():
                eval_metrics[k] = jnp.mean(jnp.array(v))

            # run t-sne to visualize latent space
            latent_actions = jnp.array(latent_actions)
            if len(images) == 0:
                images = None
            else:
                images = jnp.array(images)

            if self.cfg.visualize_latent_space:
                self.visualize_latent_space(
                    latent_actions=latent_actions, images=images, env_id=eval_env_id
                )

        return eval_metrics

    def eval_action_decoder(self, ts_lam, step: int):
        # train action decoder for a couple epochs and evaluate decoding accuracy
        # to measure the usefulness of the learned latent action space
        log("=" * 50)
        log("evaluating latent action decoder accuracy", "blue")

        input_action_dim = get_latent_action_dim(self.cfg.stage.idm)

        # initialize action decoder weights randomly
        decoder_stage = NAME_TO_STAGE_CLS["latent_action_decoder"](
            cfg=self.cfg.stage.la_decoder,
            observation_shape=self.obs_shape,
            action_dim=self.action_dim,
            input_action_dim=input_action_dim,
            continuous_actions=self.continuous_actions,
            gaussian_policy=self.cfg.stage.la_decoder.gaussian_policy,
            task_dim=self.cfg.env.task_dim,
            key=next(self.rng_seq),
            num_devices=min(self.cfg.num_xla_devices, self.num_devices),
            init_kwargs=default_weight_init,
        )

        ts_decoder = decoder_stage.create_train_state(next(self.rng_seq))

        if self.cfg.stage.name == "lam_and_action_decoder":
            # update the params
            ts_decoder = ts_decoder.replace(params={"params": ts_lam.params["decoder"]})

        decoder_update_jitted = jax.jit(
            decoder_stage.update,
            static_argnames=("is_training"),
        )

        if self.cfg.stage.name != "lam_and_action_decoder":
            # --------- train action decoder ------------

            # split up the labelled data into batches
            num_labelled_exs = self.labelled_data["observations"].shape[0]
            indices = np.arange(num_labelled_exs)

            # split into batches
            batch_size = self.cfg.stage.la_decoder.batch_size
            num_batches = math.ceil(len(indices) / batch_size)
            batch_indices = np.array_split(indices, num_batches)

            latent_actions = []
            log("getting latent actions for labelled data")
            for batch_inds in batch_indices:
                batch = {k: v[batch_inds] for k, v in self.labelled_data.items()}
                batch = Batch(**batch)
                _, _, lam_extra = self.update_jit(
                    next(self.rng_seq), ts=ts_lam, batch=batch, is_training=False
                )
                latent_actions.extend(lam_extra["latent_actions"])

            latent_actions = jnp.array(latent_actions)

            # get latent actions for batch
            batch = Batch(**self.labelled_data)

            if self.cfg.stage.idm.encoder_cfg.name in ["vit"]:
                # if using vit, we need to flatten the latent actions
                latent_actions = latent_actions[:, :-1]
                latent_actions_flat = einops.rearrange(
                    latent_actions, "b t ... -> (b t) ..."
                )
                gt_actions = batch.actions[:, :-1]
                gt_actions_flat = einops.rearrange(gt_actions, "b t ... -> (b t) ...")
                batch = batch.replace(
                    latent_actions=latent_actions_flat,
                    actions=gt_actions_flat,
                )
            else:
                batch = batch.replace(
                    latent_actions=jnp.array(latent_actions),
                    actions=batch.actions[:, -2],
                )

            ad_train_start = time.time()
            for _ in tqdm.tqdm(
                range(self.cfg.stage.la_decoder.num_updates),
                desc="action decoder train batches",
                disable=False,
                total=self.cfg.stage.la_decoder.num_updates,
            ):
                # perform a single gradient step
                ts_decoder, metrics, extra = decoder_update_jitted(
                    next(self.rng_seq), ts=ts_decoder, batch=batch, is_training=True
                )
                metrics["lr"] = ts_decoder.opt_state.hyperparams["lr"].item()

            metrics["time"] = time.time() - ad_train_start
            # take the final metrics after training
            metrics_print = jtu.tree_map(lambda x: np.round(float(x), 5), metrics)

            log(f"\n\nstep: {step} la_decoder train metrics:")
            log(pretty_repr(metrics_print))

            # log to wandb
            self.log_to_wandb(metrics, prefix="train/la_decoder/")

        #  ---------- evaluate accuracy on validation set ------------
        log("evaluating action decoder on validation set")
        train_id = self.cfg.env.env_id
        # loop over the evaluation data once
        eval_iter = self.eval_dataloaders[train_id].as_numpy_iterator()

        all_eval_metrics = []
        ad_eval_start = time.time()

        for batch in tqdm.tqdm(eval_iter, desc="action decoder eval batches"):
            batch = Batch(**batch)

            # get latent actions for eval batch from LAM
            _, _, lam_extra = self.update_jit(
                next(self.rng_seq), ts=ts_lam, batch=batch, is_training=False
            )

            if self.cfg.stage.idm.encoder_cfg.name in ["vit"]:
                import ipdb

                ipdb.set_trace()
                latent_actions = lam_extra["latent_actions"][:, :-1]
                latent_actions_flat = einops.rearrange(
                    latent_actions, "b t ... -> (b t) ..."
                )
                gt_actions = batch.actions[:, :-1]
                gt_actions_flat = einops.rearrange(gt_actions, "b t ... -> (b t) ...")
                batch = batch.replace(
                    latent_actions=latent_actions_flat,
                    actions=gt_actions_flat,
                )
            else:
                batch = batch.replace(
                    latent_actions=lam_extra["latent_actions"],
                    actions=batch.actions[:, -2],
                )

            # run evaluation
            _, eval_metrics, _ = decoder_update_jitted(
                next(self.rng_seq), ts=ts_decoder, batch=batch, is_training=False
            )
            all_eval_metrics.append(eval_metrics)

        # average metrics over all eval batches
        eval_metrics = {}
        for k in all_eval_metrics[0].keys():
            eval_metrics[k] = jnp.mean(jnp.array([m[k] for m in all_eval_metrics]))

        ad_eval_time = time.time() - ad_eval_start
        eval_metrics["time"] = ad_eval_time
        # take the final metrics after training
        metrics_print = jtu.tree_map(lambda x: np.round(float(x), 5), eval_metrics)

        log("la_decoder eval metrics:")
        log(pretty_repr(metrics_print))
        self.log_to_wandb(eval_metrics, prefix="eval/la_decoder/")

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

    def visualize(self, stage: str, batch: Batch, extra: dict, env_id: str):
        """
        Visualize next observation predictions from the FDM
        """

        if "next_obs_pred" in extra:
            # if we are using ViT-based IDM, this is [B, T-1, H, W, C]
            # else this is [B, H, W, C]
            next_obs_pred = extra["next_obs_pred"]

            # sample a couple of examples to visualize
            if self.cfg.stage.idm.encoder_cfg.name == "vit":
                # number of examples to show
                # each example is a sequence of observations
                num_ex = 4
                sample_indices = np.random.choice(
                    batch.observations.shape[0], size=num_ex, replace=False
                )
                gt_next_obs = batch.observations[sample_indices, 1:]
            else:
                num_ex = 16
                sample_indices = np.random.choice(
                    batch.observations.shape[0], size=num_ex, replace=False
                )
                # the ground truth o_t+1 is the last observation in the sequence
                gt_next_obs = batch.observations[sample_indices, -1]

            next_obs_pred = next_obs_pred[sample_indices]
            next_obs_pred = np.array(next_obs_pred)

            # stack and interleave ground truth and prediction
            to_show = np.stack([gt_next_obs, next_obs_pred], axis=1)
            to_show = to_show.reshape(-1, *to_show.shape[2:])

            if to_show.ndim == 5:
                # flatten first two dimensions if we are using ViT
                to_show = einops.rearrange(to_show, "b t h w c -> (b t) h w c")

            # to_show = [custom_to_pil(x) for x in to_show]
            to_show = [
                custom_to_pil(x, grayscale=self.cfg.env.grayscale) for x in to_show
            ]

            if self.cfg.stage.idm.encoder_cfg.name == "vit":
                # each row is a sequence of frames
                to_show = make_image_grid(to_show, num_rows=num_ex * 2)
            else:
                to_show = make_image_grid(to_show, num_rows=4)

            if self.wandb_run is not None:
                self.wandb_run.log(
                    {f"{stage}/{env_id}/next_obs_pred": wandb.Image(to_show)}
                )

    def visualize_latent_space(
        self, latent_actions: jnp.ndarray, images: jnp.ndarray = None, env_id: str = ""
    ):
        """
        Visualize latent action space using t-SNE

        Args:
            latent_actions: [B, D] latent actions
            images: [B, T, H, W, C] images corresponding to the latent actions
            env_id: environment id

        """
        # log("visualizing latent action space using t-SNE")
        # tsne = TSNE(n_components=2, random_state=0, n_jobs=-1)
        # # latent_actions = latent_actions.reshape(-1, latent_actions.shape[-1])
        # latent_tsne = tsne.fit_transform(latent_actions[:1000])

        # log("finished running tsne")

        # fig, ax = plt.subplots(figsize=(12, 12))
        # x = latent_tsne[:, 0]
        # y = latent_tsne[:, 1]
        # ax.scatter(x, y)
        # ax.set_title(f"TSNE Latent Action Space [{env_id}]")

        # # add images to certain points
        # if images is not None:
        #     artists = []
        #     images = images[:, -2:]  # only take o_t and o_t+1
        #     zoom = 1

        #     import ipdb

        #     ipdb.set_trace()

        #     for x0, y0, image in zip(x, y, images):
        #         o_t = image[0]
        #         o_tplus1 = image[1]
        #         # stack them as side by side images
        #         image = np.concatenate([o_t, o_tplus1], axis=1)
        #         im = OffsetImage(image, zoom=zoom)
        #         ab = AnnotationBbox(im, (x0, y0), xycoords="data", frameon=False)
        #         artists.append(ax.add_artist(ab))
        #     ax.update_datalim(np.column_stack([x, y]))
        #     ax.autoscale()

        # if self.wandb_run is not None:
        #     self.wandb_run.log({f"tsne/{env_id}": wandb.Image(plt)})

        # plt.close(fig)

        log("visualizing latent action space using PCA")
        pca = PCA(n_components=2)
        latent_pca = pca.fit_transform(latent_actions)

        fig, ax = plt.subplots(figsize=(12, 12))
        x = latent_pca[:, 0]
        y = latent_pca[:, 1]
        ax.scatter(x, y)
        ax.set_title(f"PCA Latent Action Space [{env_id}]")

        if self.wandb_run is not None:
            self.wandb_run.log({f"latent_space/pca/{env_id}": wandb.Image(plt)})

        plt.close(fig)

        log("visualizing latent action space using UMAP")
        # make another one using UMAP
        embedding = umap.UMAP(
            n_neighbors=5, min_dist=0.3, metric="euclidean"
        ).fit_transform(latent_actions)

        log("finished running umap")
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.scatter(embedding[:, 0], embedding[:, 1])
        ax.set_title(f"UMAP Level 1 Latent Action Space [{env_id}]")

        if self.wandb_run is not None:
            self.wandb_run.log({f"latent_space/umap/{env_id}": wandb.Image(plt)})

        plt.close(fig)

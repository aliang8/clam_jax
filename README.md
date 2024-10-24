# Prompt-Decision Transformer with Latent Actions

Some common paths to use:
```bash
data.dataset_name=procgen_dataset 

data.data_dir=/scr/shared/prompt_dtla/

stage.la_decoder_ckpt=/scr/yutaizho/projects/prompt_dtla/prompt_dtla/results/latent_action_decoder/2024-08-23-21-31-11/latent_action_decoder_s-521_e-bigfish_nt--1_d-procgen_dataset/


```


## Stage 1: Train Latent Action Model

Train latent action model on dataset of observation-only data. Currently have a CNN-backbone and ViT backbone implemented for the encoder/decoder. By default, we use CNN-based encoder/decoder.


```bash
python3 main.py \
    stage=lam \
    stage/vq=vq \
    env.env_id=bigfish \
    data.data_dir=/path/to/procgen_dataset \
    data.dataset_name=procgen_dataset \
    stage.normalize_obs_pred=True \
```

Example command to run with Hydra SLURM
```bash
python3 main.py \
    stage=lam \
    stage/vq=vq \
    env.env_id=bigfish \
    data.data_dir=/path/to/procgen_dataset \
    data.dataset_name=procgen_dataset \
    stage.normalize_obs_pred=True \
    seed=0,1,2 \
    hydra/launcher=slurm \
    --multirun
```

### ViT Transformer LAM
Example command to train a ViT based LAM
```bash
python3 main.py \
    stage=lam \
    stage/encoder_idm=tf_vit \
    stage/decoder=tf_vit \
    stage/vq=vq \
    env.env_id=bigfish \
    data.data_type=n_step \
    data.data_dir=/path/to/procgen_dataset \
    data.dataset_name=procgen_dataset \
    stage.normalize_obs_pred=True \
    data.batch_size=24 \
    data.context_len=16 \
    seed=0,1,2 \
    use_wandb=True \
    hydra/launcher=slurm \
    --multirun
```


### Relabel the dataset with latent actions

For each `(s, s')` pair, use the trained LAM to infer the latent actions between observation pair. Results in `(s, z, s')` where `z` is the latent action.

```bash
python3 scripts/relabel_dataset_with_latent_actions.py \
    stage=lam \
    data.data_dir=/path/to/procgen_dataset \
    data.dataset_name=procgen_dataset \
    data.num_trajs=-1 \
    data.batch_size=5000 \
    stage.idm_ckpt=/path/to/ckpt \
    data.data_dir=/path/to/procgen_dataset \
```

## Stage 2: Train Action Decoder

The action decoder is used to map $Z \rightarrow A$, latent action to ground-truth environment actions. We take a small subset of action-labelled examples, `(s, z, a)`, from our dataset and train a feedforward classifier to predict ground-truth actions. 

```bash
python3 main.py \
    stage=action_decoder \
    data.data_type=transitions \
    data.load_latent_actions=True \
    data.data_dir=/path/to/procgen_dataset \
    data.dataset_name=procgen_dataset \
    data.num_examples=128 \
    num_updates=5_000 \
    log_every=500 \
```

## Alternatively, you can train the latent action model and action decoder jointly

```bash
python3 main.py \
    stage=lam_and_action_decoder \
    env.env_id=bigfish \
    data.data_dir=/path/to/procgen_dataset \
    data.dataset_name=procgen_dataset \
    stage.normalize_obs_pred=True \
```


## Stage 3: Train a latent action policy 

```bash
python3 main.py \
    stage=la_bc \
    data.load_latent_actions=True \
    data.num_trajs=10 \
    data.data_type=transitions \
    data.data_dir=/path/to/procgen_dataset \
    data.dataset_name=procgen_dataset \
    run_eval_rollouts=True \
    stage.la_decoder_ckpt=/path/to/ckpt \
```

## Baselines 

### Behavioral Cloning

```bash
python3 main.py \
    stage=bc \
    stage/encoder=cnn_base \
    data.data_type=transitions \
    data.data_dir=/path/to/procgen_dataset \
    data.dataset_name=procgen_dataset \
    data.num_trajs=200 \
    num_updates=5_000 \
    run_eval_rollouts=True \
```

### VPT-IDM
```
python3 main.py \
    stage=vpt \
    data.data_type=lapo \
    data.num_trajs=10 \
    data.dataset_name=procgen_dataset \
    num_updates=10_000 \
    data.data_dir=/path/to/procgen_dataset \

# Relabel the dataset

python3 scripts/relabel_dataset_with_latent_actions.py \
    stage=vpt \
    data.dataset_name=procgen_dataset \
    data.num_trajs=-1 \
    data.batch_size=5000 \
    stage.idm_ckpt=/path/to/ckpt \
    data.data_dir=/path/to/procgen_dataset \
```

### VPT-MLP Policy
```
python3 main.py \
    stage=vpt_bc \
    stage.idm_nt=10 \
    data.data_type=transitions \
    data.num_trajs=50 \
    data.dataset_name=procgen_dataset \
    data.replace_action_with_la=True \
    data.load_latent_actions=True \
    run_eval_rollouts=True \
    num_updates=10_000 \
    data.data_dir=/path/to/procgen_dataset \
```


### Decision Transformer

Implemented following the paper [Decision Transformer: Reinforcement Learning via Sequence Modeling](https://arxiv.org/abs/2106.01345). Autoregressive sequence to sequence model that
inputs sequences of (s, a) and predicts the next action. 

```bash
python3 main.py \
    stage=dt \
    stage.context_window=100 \
    env/split=test \
    data.data_type=n_step \
    data.data_dir=/path/to/procgen_dataset \
    data.dataset_name=procgen_dataset \
    data.num_trajs=200 \
    data.context_len=100 \
    data.batch_size=32 \
    num_updates=10_000 \
    num_evals=20 \
    run_eval_rollouts=True \
    num_eval_rollouts=20 \
```


# Raparthy ICL experiments with In-Context Few-shot Prompting

Concatenates multiple trajectories from the same MDP into a single sequence and trains a DT model to predict actions. 
During evaluation, we sample a few-shot prompt from the evaluation dataset and deploy our policy autoregressively. 

```bash
python3 main.py \
    stage=dt \
    env/split=test \
    stage.context_window=100 \
    stage.model.demo_conditioning=True \
    data.num_trajs=200 \
    data.data_type=n_step \
    data.context_len=100 \
    data.data_dir=/path/to/procgen_dataset \
    data.dataset_name=procgen_dataset \
    data.batch_size=16 \
    num_updates=500_000 \
    run_eval_rollouts=True \
    num_eval_rollouts=16 \
    num_evals=100 \
```

# Continuous Action Space Experiments

## Continuous Action Space LAM

```bash
# Train LAM
python3 main.py \
    stage=lam \
    stage/vq=vq \
    stage/encoder_idm=mlp \
    stage/encoder=mlp \
    stage/decoder=mlp \
    env=mujoco \
    env.env_id=halfcheetah \
    data.dataset_name=d4rl_mujoco_halfcheetah/v0-expert \
    data.train_frac=0.9 \
    data.data_type=lapo \
    data.context_len=5

# VPT
python3 main.py \
    stage=vpt \
    stage/encoder=mlp \
    data.data_type=lapo \
    data.num_examples=5000 \
    env=mujoco \
    env.env_id=halfcheetah \
    data.train_frac=0.9 \
    data.dataset_name=d4rl_mujoco_halfcheetah/v0-expert \
    num_updates=100_000

# BC from offline dataset
python3 main.py \
    stage=bc \
    stage/encoder=mlp \
    data.data_type=transitions \
    data.dataset_name=d4rl_mujoco_halfcheetah/v0-medium \
    data.train_frac=0.8 \
    data.num_trajs=200 \
    env=mujoco \
    env.env_id=halfcheetah-medium-v0 \
    num_updates=5_000 \
    run_eval_rollouts=True \
    log_every=1_000
```


## Hierarchical Continuous Action Space LAM

```bash
python3 main.py \
    stage=lam \
    stage/vq=residual_vq \
    stage/encoder_idm=mlp \
    stage/encoder=mlp \
    stage/decoder=mlp \
    env=mujoco \
    env.env_id=halfcheetah \
    data.train_frac=0.9 \
    data.data_type=n_step \
    data.context_len=5 \
    stage.hierarchical_vq=True \
    stage.k_step_preds=5 \
```

## Generate data for MW experiments

```bash
python3 scripts/generate_mw_scripted_dataset.py
```
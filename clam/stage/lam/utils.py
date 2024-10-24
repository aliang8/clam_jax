import chex
import distrax
import jax.numpy as jnp

from prompt_dtla.models.policy import PolicyOutput
from prompt_dtla.models.vq.utils import VQOutput


@chex.dataclass
class IDMOutput:
    latent_actions: jnp.ndarray
    vq: VQOutput
    latent_action_dist: distrax.Distribution = None


@chex.dataclass
class LAMOutput:
    next_obs_pred: jnp.ndarray
    action_output: PolicyOutput
    idm: IDMOutput
    next_k_step_pred: jnp.ndarray = None

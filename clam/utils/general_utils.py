import collections
from typing import Dict

import jax
import jax.numpy as jnp
from loguru import logger
from omegaconf import DictConfig

from prompt_dtla.utils.logger import log


def count_params(params):
    return jnp.sum(
        jax.flatten_util.ravel_pytree(jax.tree_util.tree_map(lambda x: x.size, params))[
            0
        ]
    ).item()


def format_dict_keys(dictionary, format_fn):
    """Returns new dict with `format_fn` applied to keys in `dictionary`."""
    return collections.OrderedDict(
        [(format_fn(key), value) for key, value in dictionary.items()]
    )


def prefix_dict_keys(dictionary, prefix):
    """Add `prefix` to keys in `dictionary`."""
    return format_dict_keys(dictionary, lambda key: "%s%s" % (prefix, key))


def omegaconf_to_dict(d: DictConfig) -> Dict:
    """Converts an omegaconf DictConfig to a python Dict, respecting variable interpolation."""
    ret = {}
    for k, v in d.items():
        if isinstance(v, DictConfig):
            ret[k] = omegaconf_to_dict(v)
        else:
            ret[k] = v
    return ret


def print_dict(val, nesting: int = -4, start: bool = True):
    """Outputs a nested dictionory."""
    if type(val) == dict:
        if not start:
            print("")
        nesting += 4
        for k in val:
            print(nesting * " ", end="")
            print(k, end=": ")
            print_dict(val[k], nesting, start=False)
    else:
        print(val)


def check_for_nans(params):
    # check for nans
    has_nan = jax.tree_util.tree_map(lambda x: jnp.any(jnp.isnan(x)), params)
    has_nan = jax.tree_util.tree_reduce(lambda x, y: x or y, has_nan)
    log(f"has nan: {has_nan}")

    for layer_name, layer_params in params.items():
        for sublayer_name, sublayer_params in layer_params.items():
            leaves = jax.tree_util.tree_leaves(sublayer_params)
            max_value = max([jnp.max(leaf) for leaf in leaves])
            min_value = min([jnp.min(leaf) for leaf in leaves])

            log(f"{layer_name} {sublayer_name}, max: {max_value}, min: {min_value}")


def log_pytree_stats(tree):
    flattened, _ = jax.tree_util.tree_flatten_with_path(tree)

    stats = {}

    for key_path, value in flattened:
        key = "/".join([dkey.key for dkey in key_path])

        # skip bias values for now
        if "bias" in key:
            continue

        norm_val = jnp.linalg.norm(value)
        min_val = jnp.min(value)
        max_val = jnp.max(value)

        stats[key] = {
            "min": min_val,
            "max": max_val,
            "norm": norm_val,
        }

    return stats

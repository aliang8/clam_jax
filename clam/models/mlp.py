from typing import Callable, Dict, Sequence

import flax.linen as nn
import jax.numpy as jnp


class MLP(nn.Module):
    hidden_dims: Sequence[int]
    init_kwargs: Dict[str, Callable]
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.gelu
    activate_final: int = False

    def setup(self):
        self.layers = [nn.Dense(size, **self.init_kwargs) for size in self.hidden_dims]

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i + 1 < len(self.layers) or self.activate_final:
                x = self.activation(x)
        return x


class GaussianMLP(nn.Module):
    hidden_dims: Sequence[int]
    init_kwargs: Dict[str, Callable]
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.gelu
    activate_final: int = False

    def setup(self):
        self.layers = [
            nn.Dense(size, **self.init_kwargs) for size in self.hidden_dims[:-1]
        ]

        self.mu = nn.Dense(self.hidden_dims[-1], **self.init_kwargs)
        self.log_std = nn.Dense(self.hidden_dims[-1], **self.init_kwargs)

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i + 1 < len(self.layers) or self.activate_final:
                x = self.activation(x)

        mu = self.mu(x)
        log_std = self.log_std(x)
        return mu, log_std

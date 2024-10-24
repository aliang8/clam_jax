# FashionMnist VQ experiment with various settings.
# From https://github.com/minyoungg/vqtorch/blob/main/examples/autoencoder.py

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.auto import trange

from prompt_dtla.models.vq.vq import VectorQuantize

lr = 3e-4
train_iter = 1000
num_codes = 256
seed = 1234


class SimpleVQAutoEncoder(nn.Module):
    @nn.compact
    def __call__(self, x):
        layers = nn.Sequential(
            [
                nn.Conv(16, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.GELU(),
                nn.Conv(32, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(kernel_size=2, stride=2),
                VectorQuantize(dim=32, accept_image_fmap=True, **vq_kwargs),
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv(16, kernel_size=3, stride=1, padding=1),
                nn.GELU(),
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv(1, kernel_size=3, stride=1, padding=1),
            ]
        )

        for layer in layers:
            if isinstance(layer, VectorQuantize):
                x, indices, commit_loss = layer(x)
            else:
                x = layer(x)

        return x.clamp(-1, 1), indices, commit_loss


def train(train_loader, train_iterations=1000, alpha=10):
    def iterate_dataset(data_loader):
        data_iter = iter(data_loader)
        while True:
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(data_loader)
                x, y = next(data_iter)

    rng = jax.random.PRNGKey(seed)
    vq = SimpleVQAutoEncoder()
    params = vq.init(rng, jnp.ones([1, 28, 28, 1]))["params"]
    tx = optax.adam(lr)
    ts = train_state.TrainState.create(apply_fn=vq.apply, params=params, tx=tx)

    for _ in (pbar := trange(train_iterations)):
        x, _ = next(iterate_dataset(train_loader))

        def loss_fn(params, x):
            out, indices, cmt_loss = vq.apply({"params": params}, x)
            rec_loss = (out - x).abs().mean()
            return rec_loss + alpha * cmt_loss

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, (rec_loss, cmt_loss)), grad = grad_fn(ts.params, x)
        ts = ts.apply_gradients(grads=grad)

        # pbar.set_description(
        #     f"rec loss: {rec_loss.item():.3f} | "
        #     + f"cmt loss: {cmt_loss.item():.3f} | "
        #     + f"active %: {indices.unique().numel() / num_codes * 100:.3f}"
        # )
    return


transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
train_dataset = DataLoader(
    datasets.FashionMNIST(
        root="data/fashion_mnist",
        train=True,
        download=True,
        transform=transform,
    ),
    batch_size=256,
    shuffle=True,
)

print("baseline")
train(train_dataset, train_iterations=train_iter)

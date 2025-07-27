import numpy as np
import torch
from matplotlib import pyplot as plt

from src.latent_space.models.variational_auto_encoder import device


def plot_latent(autoencoder, data, num_batches=100):
    for i, (x, y) in enumerate(data):
        z = autoencoder.encoder(x.to(device))
        z = z.to("cpu").detach().numpy()
        plt.scatter(z[:, 0], z[:, 1], c=y, cmap="tab10")
        if i > num_batches:
            plt.colorbar()
            break


def plot_reconstructed(
    autoencoder,
    r0: tuple[float, float] = (-5, 10),
    r1: tuple[float, float] = (-10, 5),
    n=12,
):
    w = 28
    img = np.zeros((n * w, n * w))
    for i, y in enumerate(np.linspace(*r1, n)):
        for j, x in enumerate(np.linspace(*r0, n)):
            z = torch.Tensor([[x, y]]).to(device)
            x_hat = autoencoder.decoder(z)
            x_hat = x_hat.reshape(28, 28).to("cpu").detach().numpy()
            img[(n - 1 - i) * w : (n - 1 - i + 1) * w, j * w : (j + 1) * w] = x_hat
    plt.imshow(img, extent=(*r0, *r1))


def interpolate(autoencoder, x_1, x_2, n=12):
    z_1 = autoencoder.encoder(x_1)
    z_2 = autoencoder.encoder(x_2)
    z = torch.stack([z_1 + (z_2 - z_1) * t for t in np.linspace(0, 1, n)])
    interpolate_list = autoencoder.decoder(z)
    interpolate_list = interpolate_list.to("cpu").detach().numpy()

    w = 28
    img = np.zeros((w, n * w))
    for i, x_hat in enumerate(interpolate_list):
        img[:, i * w : (i + 1) * w] = x_hat.reshape(28, 28)
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])

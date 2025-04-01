import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src.utils.constants import img_shape


def plot_one_letter(df: pd.DataFrame | np.ndarray, index: int) -> None:
    plt.figure(figsize=(1, 1))
    plt.axis("off")

    if isinstance(df, pd.DataFrame):
        img: np.ndarray = df.iloc[index].to_numpy().reshape(img_shape)
    else:
        img: np.ndarray = df[index].reshape(img_shape)
    plt.imshow(img, cmap=plt.cm.binary)
    plt.show()


def plot_grid_of_letters(
    df: pd.DataFrame | np.ndarray,
    nb_rows: int = 5,
    nb_cols: int = 5,
    figsize: tuple[int, int] = (6, 6),
    title: str | None = None,
) -> None:
    fig, axs = plt.subplots(nrows=nb_rows, ncols=nb_cols, figsize=figsize)

    for index, ax in enumerate(axs.ravel()):
        if isinstance(df, pd.DataFrame):
            img: np.ndarray = df.iloc[index].to_numpy().reshape(img_shape)
        else:
            img: np.ndarray = df[index].reshape(img_shape)
        ax.imshow(img, cmap=plt.cm.binary)

        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=16)

    plt.show()

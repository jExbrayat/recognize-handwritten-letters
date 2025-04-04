"""Plotting images from the dataset."""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src.utils.constants import img_shape


def plot_one_letter(df: pd.DataFrame | np.ndarray, index: int) -> None:
    """Display a single letter image from a dataset.

    Parameters
    ----------
    df : pd.DataFrame or np.ndarray
        The dataset containing flattened image data (1D arrays).
        Each row should represent one image.
    index : int
        Index of the image to display.
    """
    plt.figure(figsize=(1, 1))
    plt.axis("off")

    img: np.ndarray
    if isinstance(df, pd.DataFrame):
        img = df.iloc[index].to_numpy().reshape(img_shape)
    else:
        img = df[index].reshape(img_shape)

    plt.imshow(img, cmap=plt.cm.binary)
    plt.show()


def plot_grid_of_letters(
    df: pd.DataFrame | np.ndarray,
    nb_rows: int = 5,
    nb_cols: int = 5,
    figsize: tuple[int, int] = (6, 6),
    title: str | None = None,
) -> None:
    """Display a grid of letter images from a dataset.

    Parameters
    ----------
    df : pandas.DataFrame or np.ndarray
        The dataset containing flattened image data (1D arrays).
        Each row should represent one image.
    nb_rows : int, optional
        Number of rows in the grid. Default is 5.
    nb_cols : int, optional
        Number of columns in the grid. Default is 5.
    figsize : tuple of int, optional
        Size of the figure in inches (width, height). Default is (6, 6).
    title : str or None, optional
        Optional title to display above the grid. Default is None.
    """
    fig, axs = plt.subplots(nrows=nb_rows, ncols=nb_cols, figsize=figsize)

    for index, ax in enumerate(axs.ravel()):
        img: np.ndarray
        if isinstance(df, pd.DataFrame):
            img = df.iloc[index].to_numpy().reshape(img_shape)
        else:
            img = df[index].reshape(img_shape)
        ax.imshow(img, cmap=plt.cm.binary)

        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=16)

    plt.show()

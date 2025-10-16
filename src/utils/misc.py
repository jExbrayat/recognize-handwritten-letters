"""Utils for the project."""

import os
from pathlib import Path

import numpy as np
import seaborn as sns
from numpy.random import Generator


def get_project_root() -> Path:
    """Return the project root, useful for getting absolute paths in other files.

    Returns
    -------
    The project root.

    Examples
    --------
    >>> root = get_project_root()
    """
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent
    return project_root


def go_to_root_folder() -> None:
    """Go to the root folder, useful for accessing data folder simply in notebooks.

    Examples
    --------
    >>> go_to_root_folder()
    """
    os.chdir(get_project_root())


def set_seed(seed: int = 0) -> Generator:
    """Set the global random seed and returns a NumPy random number generator.

    Parameters
    ----------
    seed : int, optional
        Seed value for reproducibility. Default is 0.

    Returns
    -------
    Generator
        A NumPy random number generator initialized with the given seed.

    Examples
    --------
    >>> seed = set_seed()
    """
    rng = np.random.default_rng(seed)
    return rng


def set_plot_options() -> None:
    """Set default plotting options.

    Examples
    --------
    >>> set_plot_options()
    """
    sns.set_theme()


def init_notebook(seed: int = 0) -> Generator:
    """Initialize notebook with consistent settings and random number generator.

    This function sets the working directory, configures plotting options, and sets a
    global seed for reproducibility. It returns a random number generator instance.

    Parameters
    ----------
    seed : int, optional
        Seed value for random number generation. Default is 0.

    Returns
    -------
    Generator
        A NumPy-compatible random number generator initialized with the given seed.

    Examples
    --------
    >>> seed = init_notebook()
    """
    go_to_root_folder()
    set_plot_options()
    rng = set_seed(seed)

    return rng

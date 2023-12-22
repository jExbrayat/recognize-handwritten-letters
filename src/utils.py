"""Utils for the project."""
import os
from pathlib import Path

import numpy as np
import seaborn as sns


def get_project_root() -> Path:
    """
    Returns the project root, useful for getting absolute paths in other files.

    Returns
    -------
    The project root.
    """
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
    return project_root


def go_to_root_folder() -> None:
    """
    Goes to the root folder, useful for accessing data folder simply in notebooks.

    Returns
    -------
    None
    """
    os.chdir(get_project_root())


def set_seed(seed: int = 0) -> None:
    np.random.seed(seed)


def set_plot_options() -> None:
    """Sets default plotting options."""
    sns.set_theme()


def init_notebook(seed: int = 0) -> None:
    go_to_root_folder()
    set_plot_options()
    set_seed(seed)

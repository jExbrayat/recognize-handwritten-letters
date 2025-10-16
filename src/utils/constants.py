"""Constants used throughout the project."""

from typing import Final

data_folder: Final[str] = "data"
nb_classes: Final[int] = 26

img_shape: Final[tuple[int, int]] = 28, 28
img_dimension: Final[int] = 784
img_shape_flat: Final[tuple[int]] = (img_dimension,)

letters_list: Final[tuple[str, ...]] = (
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
)

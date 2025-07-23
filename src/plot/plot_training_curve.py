"""Plot training curves."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


def plot_training_curve(
    training_loss: list,
    validation_loss: list,
    plot_from_n_epoch: int,
    validation_accuracy: list | None = None,
    save_path: str | None = None,
) -> Figure:
    """Plot training curves of a keras model.

    (train loss, val loss, val accuracy if classification model)

    Args:
        training_loss (list): training loss
        gathered from history callback

        validation_loss (list): validation loss
        gathered from history callback

        validation_accuracy (list): validation accuracy gathered
        from history callback

        plot_from_n_epoch (int): epoch from which to plot when there are too many
        save_path (str): path to save plot in .png format

    """
    fig = plt.figure()
    num_epochs_to_display = len(training_loss) - plot_from_n_epoch
    step_x_ticks = max(int(num_epochs_to_display / 10), 1)

    plt.plot(training_loss[plot_from_n_epoch:], label="training loss")
    plt.plot(validation_loss[plot_from_n_epoch:], label="validation loss")

    if validation_accuracy is not None:
        plt.plot(validation_accuracy[plot_from_n_epoch:], label="validation accuracy")

    plt.legend()
    plt.title("training curve")

    plt.xticks(
        np.arange(0, num_epochs_to_display, step=step_x_ticks),
        np.arange(
            plot_from_n_epoch + 1,
            plot_from_n_epoch + num_epochs_to_display + 1,
            step=step_x_ticks,
        ),
    )

    if save_path is not None:
        plt.savefig(save_path)

    return fig

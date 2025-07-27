"""Model evaluation."""

from time import time

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline

from src.utils.constants import letters_list


def evaluate_model(
    model: Pipeline,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[float, np.ndarray[float], float, Figure]:
    """Affiche l'accuracy et la matrice de confusion du modèle."""
    debut = time()

    model.fit(x_train, y_train)

    fin = time()
    total_time = round(fin - debut, 2)
    print(f"temps de calcul {total_time} s")

    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"accuracy = {accuracy}")
    f1: np.ndarray[float] = f1_score(y_test, y_pred, average=None)
    print(f"f1_score = {f1}")
    cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=letters_list)

    fig, ax = plt.subplots(figsize=(14, 10))
    disp.plot(ax=ax)
    plt.grid(False)

    return accuracy, f1, total_time, fig

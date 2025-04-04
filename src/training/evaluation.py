from time import time

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline

from src.utils.constants import letters


def evaluate_model(
    model: Pipeline,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """Affiche l'accuracy et la matrice de confusion du modèle."""
    debut = time()

    model.fit(x_train, y_train)

    fin = time()
    print(f"temps de calcul {round(fin - debut, 2)} s")

    y_pred = model.predict(x_test)

    score = accuracy_score(y_test, y_pred)
    print(f"accuracy = {score}")
    score = f1_score(y_test, y_pred, average=None)
    print(f"f1_score = {score}")
    cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=letters)

    fig, ax = plt.subplots(figsize=(14, 10))
    disp.plot(ax=ax)
    plt.grid(False)
    plt.show()

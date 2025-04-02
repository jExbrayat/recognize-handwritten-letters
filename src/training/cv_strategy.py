from sklearn.model_selection import StratifiedKFold


def strategie_evaluation(seed: int = 0, n_splits: int = 5) -> StratifiedKFold:
    """Crée notre stratégie d'évaluation."""
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return kfold

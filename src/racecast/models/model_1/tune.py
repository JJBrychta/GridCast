"""Hyperparameter search for model_1's XGBoost: RandomizedSearchCV +
TimeSeriesSplit, walking forward through `train` only (2003-2021) — val
(2022) and test (2023-2025) both stay untouched by tuning, same as train.py.

Prints the best params found; copy the ones you want into train.py's
XGBClassifier call by hand. train.py stays a fixed, reviewable set of
hyperparameters, not something that silently reads a search result file.

    uv run python -m racecast.models.model_1.tune
"""

from __future__ import annotations

from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from xgboost import XGBClassifier

from racecast.models.model_1.train import SELECTED_FEATURES, _xy, get_split

PARAM_DIST = {
    "n_estimators": randint(100, 600),
    "max_depth": randint(2, 8),
    "learning_rate": uniform(0.01, 0.29),
    "subsample": uniform(0.6, 0.4),
    "colsample_bytree": uniform(0.6, 0.4),
    "min_child_weight": randint(1, 10),
    "reg_alpha": uniform(0, 5),
    "reg_lambda": uniform(0, 5),
}


def tune(n_iter: int = 50, n_splits: int = 5) -> dict:
    train_df, _val_df, _test_df = get_split()
    X_train, y_train, _ = _xy(train_df, SELECTED_FEATURES)

    # Same imbalance-correction approach as train.py's scale_pos_weight —
    # global ratio, not tuned per fold.
    pos = int(y_train.sum())
    neg = len(y_train) - pos
    scale_pos_weight = neg / pos

    search = RandomizedSearchCV(
        estimator=XGBClassifier(random_state=42, scale_pos_weight=scale_pos_weight),
        param_distributions=PARAM_DIST,
        n_iter=n_iter,
        cv=TimeSeriesSplit(n_splits=n_splits),
        scoring="average_precision",  # matches evaluate()'s honest metric under this imbalance
        n_jobs=-1,
        random_state=42,
        verbose=1,
    )
    search.fit(X_train, y_train)

    print("\nbest CV average_precision:", search.best_score_)
    print("best params:")
    for name, value in search.best_params_.items():
        print(f"    {name}={value!r},")

    return search.best_params_


if __name__ == "__main__":
    tune()

"""Predict podium probabilities for one race: fetch/refresh its features,
load the model trained by train.py, run it. Run after that race's qualifying.

    uv run python -m racecast.models.model_1.predict 2026 14
"""

from __future__ import annotations

import sys

import pandas as pd

from racecast.models.model_1.train import SELECTED_FEATURES, TARGET, _x_only, load_model
from racecast.refresh import refresh_and_get_race


def predict_race(year: int, round_number: int) -> pd.DataFrame:
    features = refresh_and_get_race(year, round_number)
    if features.empty:
        raise ValueError(f"no data for {year} round {round_number}")

    model = load_model()
    x, clean = _x_only(features, SELECTED_FEATURES)

    return pd.DataFrame({
        "driver_ref": clean["driver_ref"].to_numpy(),
        "actual_podium": clean[TARGET].to_numpy(),
        "pred_prob": model.predict_proba(x)[:, 1],
    }).sort_values("pred_prob", ascending=False)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        year, round_number = int(sys.argv[1]), int(sys.argv[2])
    else:
        year, round_number = 2026, 14
    print(predict_race(year, round_number).to_string(index=False))

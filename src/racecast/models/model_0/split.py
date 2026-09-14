"""Time-based train/val/test split.

Test = 2023-2025, three complete seasons all under the same (post-2022)
regulation era. Val = 2022, the season of that regulation reset — a useful
single-season check on how the model handles a form/field shakeup. Train =
everything before that. 2026 is excluded: an in-progress season under a new
2026 regulation reset, not a fair backtest window.
"""

from __future__ import annotations

import pandas as pd

from racecast.config import FEATURE_MATRIX_PATH


def get_split() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_parquet(FEATURE_MATRIX_PATH)

    train = df[df["year"] <= 2021]
    val = df[df["year"] == 2022]
    test = df[df["year"].between(2023, 2025)]
    # df[df["year"] == 2026] — set aside, not part of train/val/test
    return train, val, test

"""Back-test and score a trained model with a time-ordered split.

Placeholder — fill in once training exists.
"""

from __future__ import annotations

import pandas as pd


def evaluate(model, features: pd.DataFrame) -> dict:
    """Walk-forward evaluation — never score on a race before the training cut."""
    raise NotImplementedError

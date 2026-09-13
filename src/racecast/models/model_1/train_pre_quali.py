"""Pre-qualifying variant: podium prediction using only what's known before
qualifying happens (Thu/Fri) — no grid/quali position, no quali-session gaps.

A harder, different question than train.py's model ("who's on pole going to
end up on the podium") — this asks "who do we expect on the podium before
anyone's turned a qualifying lap this weekend," using only historical form.
"""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from racecast.features.build import feature_columns
from racecast.models.model_1.split import get_split
from racecast.models.model_1.train import EXCLUDE_FEATURES as _TIER2_EXCLUDES
from racecast.models.model_1.train import _xy, evaluate

# Only known once qualifying has actually happened — not available pre-weekend.
QUALI_SESSION_FEATURES = [
    "grid_position", "grid_position_pctile",
    "quali_position", "quali_position_pctile",
    "grid_penalty", "grid_penalty_pctile",
    "quali_gap_to_pole_pct", "quali_gap_to_teammate_ms",
]

EXCLUDE_FEATURES = _TIER2_EXCLUDES + QUALI_SESSION_FEATURES


def main() -> RandomForestClassifier:
    train, val, _test = get_split()
    feature_cols = [c for c in feature_columns(train) if c not in EXCLUDE_FEATURES]

    X_train, y_train, _ = _xy(train, feature_cols)
    X_val, _y_val, val_clean = _xy(val, feature_cols)

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_train, y_train)

    print("train rows:", len(X_train), "/ val rows:", len(X_val))
    print("val metrics:", evaluate(val_clean, model.predict_proba(X_val)[:, 1]))

    importances = pd.Series(model.feature_importances_, index=feature_cols)
    print(importances.sort_values(ascending=False).head(15))

    return model


if __name__ == "__main__":
    main()

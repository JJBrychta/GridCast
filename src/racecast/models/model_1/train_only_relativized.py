"""Post-quali variant: only relativized (_pctile) columns — no raw values for
any feature that has a relativized counterpart. Same split/target/model as
train.py; tests whether the raw, era/field-size-sensitive scale of those
columns is pulling its weight or just adding redundant, noisier columns.
"""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from racecast.features.build import DRIVER_FEATURES, TEAM_FEATURES, feature_columns
from racecast.models.model_1.split import get_split
from racecast.models.model_1.train import EXCLUDE_FEATURES as _TIER2_EXCLUDES
from racecast.models.model_1.train import _xy, evaluate

# Every raw column that has a _pctile counterpart — dropped, keeping only the
# relativized version.
RAW_RELATIVIZED = DRIVER_FEATURES + TEAM_FEATURES

EXCLUDE_FEATURES = _TIER2_EXCLUDES + RAW_RELATIVIZED + ["quali_position_pctile", "grid_position_pctile"]


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

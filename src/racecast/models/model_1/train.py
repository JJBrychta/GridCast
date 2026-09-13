"""Baseline model: Random Forest on the relativized feature matrix.

Simple on purpose — the point is a first working precision to beat later,
not a tuned model. See split.py for the train/val/test year cutoffs.
"""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from racecast.features.build import feature_columns
from racecast.models.model_1.split import get_split

TARGET = "podium"

# Dropped for this baseline: NaN on any driver's first-ever visit to a circuit
# (debut, or a new calendar addition), which alone accounts for ~80% of rows
# lost to dropna. Tier-2 feature (build.py) — add back with imputation later.
EXCLUDE_FEATURES = ["driver_circuit_finish", "driver_circuit_finish_pctile"]


def _xy(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Rows with a missing target (non-starters) or missing features (rookies /
    cold-start rolling windows) can't be fit on — drop them. Returns the clean
    frame too, so callers can align predictions back to year/round for grouping."""
    clean = df.dropna(subset=[*feature_cols, TARGET])
    X = clean[feature_cols].astype("float64")
    y = clean[TARGET].astype("int64")
    return X, y, clean


def evaluate(clean: pd.DataFrame, probs) -> dict[str, float]:
    """precision@3: of the top-3-by-predicted-probability per race, how many
    were real podium finishers — the metric that matches the actual decision
    rule. average_precision: PR-AUC, the honest single-number summary at this
    class imbalance (~3:17). roc_auc: kept for reference, but inflated here —
    its FPR denominator (the 17 negatives) swallows precision-side mistakes."""
    scored = clean[["year", "round_number"]].copy()
    scored["prob"] = probs
    scored["actual"] = clean[TARGET].to_numpy()
    scored["pred_rank"] = scored.groupby(["year", "round_number"])["prob"].rank(
        ascending=False, method="first"
    )
    predicted_podium = scored[scored["pred_rank"] <= 3]
    return {
        "precision_at_3": float(predicted_podium["actual"].mean()),
        "average_precision": float(average_precision_score(scored["actual"], scored["prob"])),
        "roc_auc": float(roc_auc_score(scored["actual"], scored["prob"])),
    }


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

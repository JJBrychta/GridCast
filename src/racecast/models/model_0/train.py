"""Model 0: Random Forest, trained on SELECTED_FEATURES.

SELECTED_FEATURES is the single source of truth for what the model fits on —
every feature the pipeline currently produces, listed explicitly rather than
derived by exclusion, so trimming one down is a one-line edit here rather
than a rule buried elsewhere. See split.py for the train/val/test cutoffs.

    uv run python -m racecast.models.model_0.train
"""

from __future__ import annotations

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from racecast.config import REPO_ROOT
from racecast.models.model_0.split import get_split

TARGET = "podium"
MODEL_PATH = REPO_ROOT / "data" / "models" / "model_0.joblib"

SELECTED_FEATURES = [
    # grid / qualifying position
    "grid_position", "quali_position", "grid_penalty",
    "grid_position_pctile", "quali_position_pctile", "grid_penalty_pctile",
    # qualifying pace
    "quali_gap_to_pole_pct", "quali_gap_to_teammate_ms",
    # season context
    "season_progress",
    # team form
    "team_finish_l5", "team_best_finish_l5",
    "team_podium_rate_l10", "team_dnf_rate_l10",
    "team_finish_l5_pctile", "team_best_finish_l5_pctile",
    "team_podium_rate_l10_pctile", "team_dnf_rate_l10_pctile",
    # driver form
    "driver_finish_l5", "driver_podium_rate_l5",
    "driver_podium_rate_ewm15", "driver_dnf_rate_l10",
    "driver_finish_l5_pctile", "driver_podium_rate_l5_pctile",
    "driver_podium_rate_ewm15_pctile", "driver_dnf_rate_l10_pctile",
    # driver history
    # "driver_circuit_finish", "driver_circuit_finish_pctile",  # NaN for every
    # driver on a new circuit (e.g. 2026's Madrid) -> drops the whole race, not
    # just cold-start rows. Needs imputation before it's safe to include.
    "driver_quali_pos_l5", "driver_quali_pos_l5_pctile",
    "driver_gain_l5",
    "driver_career_starts",
    "driver_career_podium_rate", "driver_career_podium_rate_pctile",
]


def _xy(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Rows with a missing target (non-starters) or missing features (rookies /
    cold-start rolling windows) can't be fit on — drop them. Returns the clean
    frame too, so callers can align predictions back to year/round for grouping."""
    clean = df.dropna(subset=[*feature_cols, TARGET])
    X = clean[feature_cols].astype("float64")
    y = clean[TARGET].astype("int64")
    return X, y, clean


def _x_only(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Like _xy but for inference: only drop rows with a missing FEATURE — the
    target is allowed to be unknown (unrun race, no result exists yet), unlike
    training/eval where a real label is required."""
    clean = df.dropna(subset=feature_cols)
    X = clean[feature_cols].astype("float64")
    return X, clean


def evaluate(clean: pd.DataFrame, probs) -> dict[str, float]:
    """precision@3: of the top-3-by-predicted-probability per race, how many
    were real podium finishers — the metric that matches the actual decision
    rule. average_precision: PR-AUC, the honest single-number summary at this
    class imbalance (~3:17). roc_auc: kept for reference, but inflated here —
    its FPR denominator (the many negatives) swallows precision-side mistakes."""
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


def save_model(model: RandomForestClassifier) -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)


def load_model() -> RandomForestClassifier:
    return joblib.load(MODEL_PATH)


def main() -> RandomForestClassifier:
    train, val, _test = get_split()

    missing = [c for c in SELECTED_FEATURES if c not in train.columns]
    if missing:
        raise ValueError(f"SELECTED_FEATURES has no columns in the matrix: {missing}")

    X_train, y_train, _ = _xy(train, SELECTED_FEATURES)
    X_val, _y_val, val_clean = _xy(val, SELECTED_FEATURES)

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_train, y_train)

    print("train rows:", len(X_train), "/ val rows:", len(X_val))
    print("val metrics:", evaluate(val_clean, model.predict_proba(X_val)[:, 1]))

    importances = pd.Series(model.feature_importances_, index=SELECTED_FEATURES)
    print(importances.sort_values(ascending=False).head(15))

    save_model(model)
    print(f"saved -> {MODEL_PATH.relative_to(REPO_ROOT)}")

    return model


if __name__ == "__main__":
    main()

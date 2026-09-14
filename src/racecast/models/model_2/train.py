"""Model 2: XGBoost learning-to-rank, fed the whole race at once via `qid`
groups rather than one independent row per driver. Trained on graded
relevance (field_size - finish position: P1 highest, DNFs tied at 0 — see
t_relevance in features/build.py) instead of binary podium, using
objective="rank:ndcg" so the model directly optimizes race ordering rather
than 20 separate yes/no guesses.

Self-contained on purpose, same shape as model_0/model_1: split, evaluate,
train, save/load all live in this one file rather than a shared module.

    uv run python -m racecast.models.model_2.train
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRanker

from racecast.config import FEATURE_MATRIX_PATH, REPO_ROOT

TARGET = "relevance"
MODEL_PATH = REPO_ROOT / "data" / "models" / "model_2.joblib"

SELECTED_FEATURES = [
    # grid / qualifying position

    # "grid_position",
    # "quali_position",
    # "grid_penalty",
    "grid_position_pctile",
    "quali_position_pctile",
    # "grid_penalty_pctile",

    # qualifying pace
    "quali_gap_to_pole_pct",
    # "quali_gap_to_teammate_ms",

    # season context
    # "season_progress",

    # team form
    # "team_finish_l5",
    # "team_best_finish_l5",
    # "team_podium_rate_l10",
    # "team_dnf_rate_l10",
    # "team_finish_l5_pctile",
    "team_best_finish_l5_pctile",
    # "team_podium_rate_l10_pctile",
    # "team_dnf_rate_l10_pctile",

    # driver form
    # "driver_finish_l5",
    # "driver_podium_rate_l5",
    # "driver_podium_rate_ewm15",
    # "driver_dnf_rate_l10",
    # "driver_finish_l5_pctile",
    # "driver_podium_rate_l5_pctile",
    "driver_podium_rate_ewm15_pctile",
    # "driver_dnf_rate_l10_pctile",

    # driver history
    # "driver_circuit_finish",         # NaN for every driver on a new circuit
    # "driver_circuit_finish_pctile",  # (e.g. 2026's Madrid) -> drops the whole
    # race, not just cold-start rows. Needs imputation before it's safe to include.
    # "driver_quali_pos_l5",
    "driver_quali_pos_l5_pctile",
    # "driver_gain_l5",
    # "driver_career_starts",
    # "driver_career_podium_rate",
    "driver_career_podium_rate_pctile",
]


def get_split() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Time-based train/val/test split. Test = 2023-2025, three complete
    seasons under the current (post-2022) regulation era. Val = 2022, the
    reset season itself — a useful single-season shakeup check. Train =
    everything before. 2026 excluded: in-progress, under yet another reset.
    A race never straddles a year boundary, so groups stay intact."""
    df = pd.read_parquet(FEATURE_MATRIX_PATH)
    train_df = df[df["year"] <= 2021]
    val_df = df[df["year"] == 2022]
    test_df = df[df["year"].between(2023, 2025)]
    return train_df, val_df, test_df


def _xy(
    df: pd.DataFrame, feature_cols: list[str]
) -> tuple[pd.DataFrame, pd.Series, np.ndarray, pd.DataFrame]:
    """Sort by race so qid groups are contiguous (XGBRanker requires this),
    drop rows with a missing target or missing features, and build the qid
    group array. Returns the clean frame too — it keeps podium/relevance
    alongside the identifiers for evaluate()."""
    ordered = df.sort_values(["year", "round_number"])
    clean = ordered.dropna(subset=[*feature_cols, TARGET])
    X = clean[feature_cols].astype("float64")
    y = clean[TARGET].astype("int64")
    qid = clean.groupby(["year", "round_number"], sort=False).ngroup().to_numpy()
    return X, y, qid, clean


def _x_only(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Like _xy but for inference: only drop rows with a missing FEATURE — the
    target is allowed to be unknown (unrun race, no result exists yet). No
    qid needed here — at inference you're only ever scoring one race, so
    there's nothing to group."""
    clean = df.dropna(subset=feature_cols)
    X = clean[feature_cols].astype("float64")
    return X, clean


def _ndcg_at_k(group: pd.DataFrame, k: int = 3) -> float:
    """NDCG@k for one race: how well predicted-score order matches true
    relevance order, credited more for getting the exact top spots right,
    not just including them somewhere in the top k."""
    discounts = 1.0 / np.log2(np.arange(2, k + 2))

    by_pred = group.sort_values("score", ascending=False)["relevance"].to_numpy()[:k]
    dcg = float((by_pred * discounts[: len(by_pred)]).sum())

    ideal = np.sort(group["relevance"].to_numpy())[::-1][:k]
    idcg = float((ideal * discounts[: len(ideal)]).sum())

    return dcg / idcg if idcg > 0 else 0.0


def evaluate(clean: pd.DataFrame, scores) -> dict[str, float]:
    """ndcg_at_3: the metric this model actually optimizes for — rewards
    getting P1/P2/P3 in the right order, not just in the top 3 somewhere.
    precision_at_3: same top-3-hit-rate check used by model_0/model_1, kept
    for cross-model comparability."""
    scored = clean[["year", "round_number", "podium", TARGET]].copy()
    scored["score"] = scores
    scored["pred_rank"] = scored.groupby(["year", "round_number"])["score"].rank(
        ascending=False, method="first"
    )

    predicted_podium = scored[scored["pred_rank"] <= 3]
    ndcg_values = scored.groupby(["year", "round_number"]).apply(
        lambda g: _ndcg_at_k(g, k=3), include_groups=False
    )

    return {
        "precision_at_3": float(predicted_podium["podium"].mean()),
        "ndcg_at_3": float(ndcg_values.mean()),
    }


def save_model(model: XGBRanker) -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)


def load_model() -> XGBRanker:
    return joblib.load(MODEL_PATH)


def train() -> XGBRanker:
    train_df, val_df, _test_df = get_split()

    missing = [c for c in SELECTED_FEATURES if c not in train_df.columns]
    if missing:
        raise ValueError(f"SELECTED_FEATURES has columns not in the matrix: {missing}")



    X_train, y_train, qid_train, _ = _xy(train_df, SELECTED_FEATURES)
    X_val, _y_val, _qid_val, val_clean = _xy(val_df, SELECTED_FEATURES)

    model = XGBRanker(
        objective="rank:ndcg",
        n_estimators=500,
        max_depth=2,
        learning_rate=0.05,
        random_state=42,
    )
    model.fit(X_train, y_train, qid=qid_train)

    print("train rows:", len(X_train), "/ val rows:", len(X_val))
    print("val metrics:", evaluate(val_clean, model.predict(X_val)))

    importances = pd.Series(model.feature_importances_, index=SELECTED_FEATURES)
    print(importances.sort_values(ascending=False).head(15))

    save_model(model)
    print(f"saved -> {MODEL_PATH.relative_to(REPO_ROOT)}")

    return model


if __name__ == "__main__":
    train()

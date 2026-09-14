# RaceCast — models

Predicts the podium (top 3) for a race. Three models, same time-based
train/val/test split, mostly the same features — compared to see whether
algorithm choice matters here (short answer: not much, see below).

## Results (val = 2022)

| | approach | precision@3 | other metric |
|---|---|---|---|
| model_0 | Random Forest, binary podium target | 0.667 | avg precision 0.661 |
| model_1 | XGBoost (tuned), binary podium target | 0.652 | avg precision 0.652 |
| model_2 | XGBoost, learning-to-rank (whole race at once) | 0.682 | ndcg@3 0.824 |

All three land in the same range — `grid_position`/`quali_position` dominate
feature importance in every model, so the ceiling here is set by the
features, not the algorithm. See git history / prior discussion for the
fuller writeup if needed.

## Predict a race

Each `model_N/predict.py` fetches/refreshes that race's features, loads the
model `train.py` already saved, and prints predicted podium order:

```bash
uv run python -m racecast.models.model_0.predict 2026 14
uv run python -m racecast.models.model_1.predict 2026 14
uv run python -m racecast.models.model_2.predict 2026 14
```

or via the Makefile (model_0 only, for now):

```bash
make predict YEAR=2026 ROUND=14
```

Retrain a model with `uv run python -m racecast.models.model_N.train`
(saves to `data/models/model_N.joblib`).

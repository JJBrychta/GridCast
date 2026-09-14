# RaceCast

Predicts the F1 podium (top 3 finishers) for a race, using historical
qualifying/race data pulled from FastF1/Ergast. Three models compared —
a Random Forest, a tuned XGBoost classifier, and an XGBoost
learning-to-rank model — trained on the same time-based split so the
comparison is fair.

## Results

| | approach | precision@3 | other metric |
|---|---|---|---|
| model_0 | Random Forest, binary podium target | 0.667 | avg precision 0.661 |
| model_1 | XGBoost (tuned), binary podium target | 0.652 | avg precision 0.650 |
| model_2 | XGBoost, learning-to-rank (whole race at once) | 0.682 | ndcg@3 0.824 |

precision@3 = of the 3 drivers predicted per race, how many actually
podiumed (evaluated on the 2022 season, held out from training). All three
land in a similar range — qualifying/grid position dominates feature
importance in every model, so the ceiling here is set by the features
available, not by which algorithm is used.

Details: [src/racecast/models/README.md](src/racecast/models/README.md).

## Project layout

```
racecast/
  ingest/     fetch schedules + session results from FastF1/Ergast -> raw/*.json
  db/         raw/ -> data/racecast.sqlite (idempotent, incremental)
  features/   sqlite -> data/datasets/feature_matrix.parquet (rolling-window features, relativized within each race)
  refresh.py  the whole ingest -> db -> features chain in one call, for a live/upcoming race
  models/     model_0 (Random Forest), model_1 (XGBoost), model_2 (XGBoost ranker) — each self-contained
```

More detail: [src/racecast/features/README.md](src/racecast/features/README.md)
for how the feature matrix is built, [src/racecast/models/README.md](src/racecast/models/README.md)
for the models themselves.

## Running the pipeline

One-time setup:

```bash
uv sync
make init-db
```

Full data pull + build (safe to re-run any time — both fetch and the DB
build are incremental, only touching what's new or changed):

```bash
make ingest      # fetch schedules + session results not already on disk
make build-db    # raw/ -> data/racecast.sqlite
make features    # sqlite -> data/datasets/feature_matrix.parquet
```

or all three in one call:

```bash
make refresh
```

Train a model (saves to `data/models/model_N.joblib`):

```bash
uv run python -m racecast.models.model_0.train   # Random Forest
uv run python -m racecast.models.model_1.train   # XGBoost
uv run python -m racecast.models.model_2.train   # XGBoost ranker
```

Predict a specific race — fetches/refreshes just what's needed for that
race, loads the saved model, prints predicted podium order. Run this after
that race's qualifying:

```bash
uv run python -m racecast.models.model_0.predict 2026 14
make predict YEAR=2026 ROUND=14   # same thing, model_0 only
```

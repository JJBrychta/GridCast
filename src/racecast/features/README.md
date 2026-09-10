# RaceCast — features

Turns the SQLite DB into a **model-ready matrix**: one row per race entry
(driver × Grand Prix), a column per feature, and a target.

```
db  ──►  feature_base.sql  ──►  load_base()  ──►  f_*(base)  ──►  build_matrix()  ──►  data/datasets/
         race + quali          typed frame       one feature      id cols + features   feature_matrix.parquet
                               1991+             each             + target
```

- `query.py` — thin DB → DataFrame reader (`query`, `save`), shared with notebooks.
- `feature_base.sql` — the join.
- `build.py` — `load_base`, the feature functions, `build_matrix`.

---

## The base table (`load_base`)

One row per **race entry**, that weekend's qualifying pivoted on:

| group | columns |
|---|---|
| identity | `year`, `round_number`, `event_date`, `race_date`, `event_format`, `driver_id/ref`, `constructor_id`/`team_ref`, `circuit_id/ref` |
| pre-race (feature inputs) | `grid_position`, `quali_position`, `q1_ms`, `q2_ms`, `q3_ms` |
| outcome (target source, never a feature) | `finish_position`, `classified_position`, `status`, `dnf`, `points`, `laps`, `time_ms` |
| race conditions (2018+, leak — circuit history only) | `rainfall_any`, `rainfall_pct`, `air_temp_avg`, `track_temp_avg` |

`_IDENTIFIERS` (carried into the matrix, excluded from `X`) is a subset —
`year`, `round_number`, `event_date`, and the six `*_id`/`*_ref` columns.

Two decisions baked into `load_base`:

- **History floor = 1991** (`HISTORY_FROM_YEAR`). Michael Schumacher (debut 1991)
  is the earliest debut of any driver racing in 2003+, so from 1991 every
  modelling row has that driver's — and their team's — full career feeding its
  rolling windows. `feature_base.sql` filters with `:from_year`; pass
  `from_season=None` for all of it back to 1950.
- **Pit-lane start → back of grid.** Ergast stores grid `0` for a pit-lane
  start; that's a nonsense value for the model and blows up `log(grid)`.
  Remapped to the field size — a real, large grid slot.

**Qualifying-data gaps.** `q1_ms`/`q2_ms`/`q3_ms` are only reliable from **2006**
(2003–05 used single-lap / aggregate qualifying — often only `q1_ms`, sometimes
nothing). A handful of 2006+ sessions (e.g. Miami 2025) have positions but no
times. `quali_position` is populated whenever there's any qualifying, so
features lean on it; time-gap features are `NaN` where times are missing (the
GBT handles it). A `made_q3` flag was dropped for this reason — it read a flat 0
for 2003–05 and is redundant with `quali_position` anyway.

---

## Two rules that make every feature valid

### 1. No leakage — a feature may use only what's known before the race

`feature_base.sql` is driven by **race results**, so a row exists only once the
race has happened. Rolling features enforce "before this race" with
`shift(1)`; within-race features (gap to pole/teammate) are fine unshifted
because qualifying precedes the race.

### 2. Finish vs classification

| | `finish_position` | `classified_position` |
|---|---|---|
| a real finish | `4` | `"4"` |
| **a DNF** | Ergast classifies them at the **back** (~P19) | `"R"` / `"W"` / `"D"` |

So:
- **podium target & podium-rate features** key on `classified_position ∈ {"1","2","3"}` — a leader who retires is `"R"`, not a podium.
- **"finishing form" features** use `_classified_finish` = `finish_position` but `NaN` for a DNF ("pace when they finished"). Retirements are left to the reliability features so the two aren't double-counted.

---

## The leak-safe primitives

### `rolling_prior(base, group, value, *, window=5, halflife=None, min_periods=3, stat="mean")`

Per group (driver, or `["driver_id","circuit_id"]`, …), sorted by date:
`shift(1)` (drop the current race) → `rolling(window)` (or `ewm(halflife)`) →
`stat`. Returns a Series aligned to `base`. `value` is a column name or a
derived Series.

- `halflife` set → EWMA over **all** prior races, weight halving every
  `halflife` races (distant past fades, no hard cutoff).
- `halflife` None → fixed `window` of the last N races.

### `team_prior(base, value, *, window=5, min_periods=3, race_stat="mean", roll_stat="mean")`

A constructor has **more than one car-row per race** (two since the 1990 two-car
rule), so `rolling_prior(…, "constructor_id", …)` would let the second car see
its teammate's *same-race* result — `shift(1)` steps one row, not one race.
`team_prior` collapses each `(team, race)` to one value first (`race_stat` =
`mean`/`min`/`max`), *then* rolls, *then* broadcasts back to every car row. It
doesn't assume a car count (a `shift(2)` hack would). Use it for **every**
constructor-grouped feature.

---

## Basic features — no history, no window

Row-wise or within-race. Available from a driver's very first weekend.

| column | fn | what / why |
|---|---|---|
| `grid_position` | `f_grid` | **The anchor.** ~85 % of podiums are decided by where you start. Nonlinear, so let the tree bend it. |
| `quali_position` | `f_grid` | Grid before any penalty — the pure result. |
| `grid_penalty` | `f_grid_penalty` | `grid − quali`. Large positive = engine/gearbox penalty dropped them back. |
| `quali_gap_to_pole_pct` | `f_quali_gap_to_pole` | `(best_lap − pole) / pole`, within the race. P2 that's 0.05 s off pole ≠ P2 that's 0.6 s off. |
| `quali_gap_to_teammate_ms` | `f_quali_gap_to_teammate` | Best lap minus the team's mean, same weekend. Isolates **driver** pace from **car** pace (same machinery, same track). |
| `season_progress` | `f_season_progress` | `round / rounds_in_season`. The pecking order is fuzzier early; some teams develop through the year. |

---

## Tier 1 — history features to build first

For podium probability the signal ranking is **car ≫ driver > reliability**, so
Tier 1 is mostly the car.

| column | fn | what / why |
|---|---|---|
| `team_finish_l5` | `f_team_form` | Mean classified finish of the team's cars over its last 5 races. Current car competitiveness. |
| `team_podium_rate_l10` | `f_team_form` | Fraction of the last 10 races the team put **a** car on the podium (`race_stat="max"`). "Is this a podium car right now." |
| `team_best_finish_l5` | `f_team_ceiling` | The **better** car's result per race (`race_stat="min"`), rolled. The car's *ceiling* — not dragged down by the slower car or by DNFs. Most predictive of podium capability. |
| `driver_finish_l5` | `f_driver_form` | Driver's mean classified finish, last 5. Recent form within whatever car they have. |
| `driver_podium_rate_l5` | `f_driver_form` | Podiums in the last 5 — a hot streak. |
| `driver_podium_rate_ewm15` | `f_driver_podium_rate_era` | EWMA podium rate, **halflife 15 races**. "Current era" level — a breakout after a slow start (Pérez: 2 % on Racing Point → 43 % on Red Bull) isn't dragged down by the distant past the way a career average is. |
| `driver_dnf_rate_l10` | `f_reliability` | Can't podium if you don't finish. Incident- and reliability-prone drivers. |
| `team_dnf_rate_l10` | `f_reliability` | Fraction of the team's car-races ending in a DNF, last 10 races. Car reliability. |

---

## Tier 2 — add once Tier 1 is scoring

| column | fn | what / why |
|---|---|---|
| `driver_circuit_finish` | `f_driver_circuit` | This driver's mean classified finish **at this track** (`[driver_id, circuit_id]`, last 3 visits). Some drivers own certain circuits. `NaN` for a first-ever visit — correct, there's no history. |
| `driver_quali_pos_l5` | `f_driver_quali_form` | Qualifying-position form. Less noisy than race results (no strategy, no incidents) — a cleaner read on raw pace. |
| `driver_gain_l5` | `f_positions_gained` | `log(grid) − log(finish)` rolled over the last 5 **finished** races. The `log` stretches the front: 2→1 is a big value, 15→14 barely registers — so it rewards *meaningful* overtaking, not midfield shuffling. DNFs excluded (not lost on merit). |
| `driver_career_starts` | `f_experience` | Career races so far. A real feature (experience) **and** the rookie / cold-start flag — it tells the model how much to trust the other rolling features. Never `NaN` (0 on debut). |
| `driver_career_podium_rate` | `f_driver_career_podium_rate` | Prior podiums / prior starts, expanding. Long-run caliber a 5-race window can't see — separates a proven front-runner from a journeyman regardless of current car. Near-constant within a season; noisy in the first ~10 starts (pair with `driver_career_starts`); `NaN` on debut. |

Three podium-rate timescales on purpose — `l5` (form), `ewm15` (era),
`career` (lifetime). They answer different questions; let feature-importance
tell you which to keep.

---

## Compose

```python
from racecast.features.build import (
    load_base, build_matrix, feature_columns, BASIC_FEATURES, HISTORY_FEATURES,
)

base   = load_base()                                              # 1991+, ~14 k rows
matrix = build_matrix(base, BASIC_FEATURES + HISTORY_FEATURES, target="podium")
X, y   = matrix[feature_columns(matrix)], matrix["podium"]        # 19 features, 9.7 k rows
```

Ablations — pass a subset: `build_matrix(base, ["f_grid"], ...)`, or
`BASIC_FEATURES + HISTORY_FEATURES` minus the ones you're testing.

- `build_matrix` runs the feature fns on the **full** base (pre-2003 races feed
  the windows), then trims rows before `min_year` (`MODEL_FROM_YEAR = 2003`).
- Identifiers ride along in the saved matrix — the time-split needs
  `event_date`, and predictions need `driver_ref` to say whose they are — but
  `feature_columns()` excludes them (and every `TARGETS` key) from what the
  model fits on. Training on `driver_id` or raw `year` would just memorise
  "Verstappen podiums" / "2023 = Red Bull".

### Targets

`TARGETS` is a registry: `build_matrix(base, features, target="podium")`. Add
`t_points`, `t_win`, `t_beat_teammate` there to predict other things — no other
change. Every target keys on the started / `classified_position` rules above and
returns `<NA>` for a non-starter.

### Predicting an upcoming race

The base SQL is race-driven, so a not-yet-run race isn't in it. Build its rows
from that weekend's **qualifying** (a `load_upcoming`-style query, outcomes
`NULL`), `pd.concat` with `load_base()`, then `build_matrix(..., min_year=None)`
and slice out the target race. The rolling features fill from real history
because the identical code ran on `history + upcoming`.

---

## Run

```bash
make features      # -> data/datasets/feature_base.parquet + feature_matrix.parquet
```

Save Parquet (round-trips dtypes exactly, ~10× smaller); a `.csv` name gets you
a human-readable copy. `data/datasets/` is the keepers dir; `data/queries/` is
scratch.

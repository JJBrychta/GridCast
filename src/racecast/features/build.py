"""Build the feature matrix from the derived DB.  See README.md.

``load_base()``    one row per race entry (driver x GP), 1991+, quali joined on.
``f_*(base)``      one feature: takes ``base``, returns a named Series / DataFrame.
``build_matrix()`` run feature fns on the full base, trim to the modelling window.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from racecast.features.query import DATASETS_DIR, query, save

FEATURE_BASE_SQL = Path(__file__).with_name("feature_base.sql")

# Columns to read as pandas nullable integers (SQLite hands them back as float
# once a NULL appears in the column).
_INT_COLS = (
    "round_number", "grid_position", "quali_position",
    "q1_ms", "q2_ms", "q3_ms", "finish_position", "laps", "time_ms",
    "dnf", "rainfall_any",
)

# Row identity — carried through to the matrix unchanged. IDs for grouping,
# refs for reading a row / a prediction, event_date for the time-ordered split.
# Kept in the saved matrix; excluded from what the model fits on (see feature_columns).
_IDENTIFIERS = [
    "year", "round_number", "event_date",
    "driver_id", "driver_ref", "constructor_id", "team_ref",
    "circuit_id", "circuit_ref",
]


MODEL_FROM_YEAR = 2003    # modelling window — build_matrix trims rows before this
HISTORY_FROM_YEAR = 1991  # how far back to load: Michael Schumacher (debut 1991) is
                          # the earliest debut of any driver active in 2003+, so from
                          # 1991 every modelling-window row has that driver's & team's
                          # full career history feeding its rolling features.


def load_base(*, from_season: int | None = HISTORY_FROM_YEAR) -> pd.DataFrame:
    """The joined race+qualifying table every feature is derived from.

    Loads from ``from_season`` (default ``HISTORY_FROM_YEAR``, enough pre-window
    history for every 2003+ driver); pass ``None`` for all history back to 1950.
    Dates parsed to datetime, cross-null int columns cast to nullable ``Int64``,
    rows sorted chronologically.
    """
    df = query(FEATURE_BASE_SQL.read_text(), {"from_year": from_season or 0})

    df["event_date"] = pd.to_datetime(df["event_date"])
    df["race_date"] = pd.to_datetime(df["race_date"])
    for col in _INT_COLS:
        df[col] = df[col].astype("Int64")

    # Pit-lane start: Ergast stores grid 0. Functionally it's the back of the
    # grid (they start behind everyone and clear the field on lap 1), so encode
    # it as the field size — a real, large grid slot instead of a nonsense 0.
    field_size = df.groupby(["year", "round_number"])["driver_id"].transform("size")
    df["grid_position"] = df["grid_position"].mask(df["grid_position"] == 0, field_size)

    return df.sort_values(["event_date", "round_number"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
#  targets — each: base -> Series (<NA> for a non-starter). Add more to predict
#  other things (points finish, win, top-10, beat-teammate, …).
# --------------------------------------------------------------------------- #

def _started(base: pd.DataFrame) -> pd.Series:
    return ~(base["classified_position"].isna() & base["finish_position"].isna())


def t_podium(base: pd.DataFrame) -> pd.Series:
    """1 if classified 1st-3rd. Keyed on ``classified_position`` so a driver
    classified 3rd but retired doesn't count as a podium finish."""
    y = base["classified_position"].isin(["1", "2", "3"]).astype("int8")
    return y.where(_started(base), other=pd.NA).astype("Int8")

def t_position(base: pd.DataFrame) -> pd.Series:
    """Position in race based on classified_position: the numeric finish order
    when classified, or the field size (last-place-equivalent) for a DNF/DSQ/
    withdrawal — mirrors the grid_position pit-lane-start convention in
    load_base(). NA for a non-starter (race not run yet)."""
    field_size = base.groupby(["year", "round_number"])["driver_id"].transform("size")
    numeric = base["classified_position"].str.isdigit().astype("boolean").fillna(False)
    position = pd.to_numeric(base["classified_position"].where(numeric))
    y = position.fillna(field_size).astype("Int64")
    return y.where(_started(base), other=pd.NA).astype("Int64")

TARGETS = {
    "podium": t_podium,
    "finish_position": t_position,
}


# --------------------------------------------------------------------------- #
#  basic features — no history / rolling window
#  row-wise, or within-race (groupby a race, but no shift -> not a leak:
#  qualifying happens before the race)
# --------------------------------------------------------------------------- #

def _best_quali_ms(base: pd.DataFrame) -> pd.Series:
    """A driver's fastest lap of the weekend (best of Q1/Q2/Q3)."""
    return base[["q1_ms", "q2_ms", "q3_ms"]].min(axis=1)


def f_grid(base: pd.DataFrame) -> pd.DataFrame:
    return base[["grid_position", "quali_position"]]


def f_grid_penalty(base: pd.DataFrame) -> pd.Series:
    """Started further back than qualified -> engine / gearbox penalty."""
    return (base["grid_position"] - base["quali_position"]).rename("grid_penalty")


def f_quali_gap_to_pole(base: pd.DataFrame) -> pd.Series:
    """Fraction slower than pole: (best - pole) / pole, within each race."""
    best = _best_quali_ms(base)
    pole = best.groupby([base["year"], base["round_number"]]).transform("min")
    return ((best - pole) / pole).rename("quali_gap_to_pole_pct")


def f_quali_gap_to_teammate(base: pd.DataFrame) -> pd.Series:
    """Gap in ms to the mean of your own team's cars this weekend (<0 = faster)."""
    best = _best_quali_ms(base)
    team_avg = best.groupby(
        [base["year"], base["round_number"], base["constructor_id"]]
    ).transform("mean")
    return (best - team_avg).rename("quali_gap_to_teammate_ms")


def f_season_progress(base: pd.DataFrame) -> pd.Series:
    """Round number as a fraction of that season's total rounds."""
    total = base.groupby("year")["round_number"].transform("max")
    return (base["round_number"] / total).rename("season_progress")


BASIC_FEATURES = [
    f_grid, f_grid_penalty,
    f_quali_gap_to_pole, f_quali_gap_to_teammate, f_season_progress,
]

# --------------------------------------------------------------------------- #
#  history features — history / rolling window
# --------------------------------------------------------------------------- #

def rolling_prior(base, group, value, *, window=5, halflife=None, min_periods=3, stat="mean"):
    """Aggregate `value` over each group's PRIOR races only. shift(1) drops the
    current race; the result is aligned to base.index.

    `halflife` set  -> EWMA over *all* prior races, weight halving every
                       `halflife` races (distant past fades, no hard cutoff).
    `halflife` None -> `stat` over a fixed `window` of the last races.
    """
    ordered = base.sort_values(["event_date", "round_number"])
    vals = (ordered[value] if isinstance(value, str)
            else pd.Series(value).reindex(ordered.index)).astype("float64")
    keys = ordered[group] if isinstance(group, str) else [ordered[g] for g in group]

    if halflife is not None:
        windower = lambda s: s.shift(1).ewm(halflife=halflife, min_periods=min_periods).mean()
    else:
        windower = lambda s: getattr(s.shift(1).rolling(window, min_periods=min_periods), stat)()

    return vals.groupby(keys, sort=False).transform(windower).reindex(base.index)

def _podium_flag(base):
    """0/1 podium (not <NA>) for use inside rolling windows — keyed on
    ``classified_position`` (a real classified top-3, not a retirement)."""
    return base["classified_position"].isin(["1", "2", "3"]).astype("float64")

def _classified_finish(base):
    """``finish_position`` but ``NaN`` for a DNF — "where they finished *when they
    finished*". Race pace, with the reliability signal left to f_reliability so
    the two aren't double-counted. For a DNF Ergast classifies the driver at the
    back (~P19) regardless of where they were running, which would otherwise
    quietly bury pace under retirements."""
    return base["finish_position"].astype("float64").where(base["dnf"].eq(0))


def team_prior(base, value, *, window=5, min_periods=3, race_stat="mean", roll_stat="mean"):
    """Roll a per-(constructor, race) aggregate of ``value`` over the team's
    PRIOR races, broadcast back to both car rows.

    A constructor has two car-rows per race — rolling on the raw rows lets the
    second car see its teammate's *same-race* result, because ``shift(1)`` only
    steps one row. Collapsing each (team, race) to one value first makes
    ``shift(1)`` mean "one race back". ``race_stat`` combines the two cars
    (``"mean"`` / ``"min"`` / ``"max"``); ``roll_stat`` reduces the window.
    """
    src = base[["constructor_id", "year", "round_number", "event_date"]].copy()
    src["_v"] = (base[value] if isinstance(value, str)
                 else pd.Series(value, index=base.index)).astype("float64")

    per = (src.groupby(["constructor_id", "year", "round_number"], as_index=False)
              .agg(_v=("_v", race_stat), event_date=("event_date", "first")))
    per["_r"] = (per.sort_values(["event_date", "round_number"])
                    .groupby("constructor_id")["_v"]
                    .transform(lambda s: getattr(
                        s.shift(1).rolling(window, min_periods=min_periods), roll_stat)()))

    lookup = per.set_index(["constructor_id", "year", "round_number"])["_r"]
    keys = list(zip(base["constructor_id"], base["year"], base["round_number"]))
    return pd.Series(lookup.reindex(keys).to_numpy(), index=base.index)

# Tier 1
def f_team_form(base):
    """The car — both drivers' results pooled per race, rolled over prior races."""
    return pd.DataFrame({
        "team_finish_l5":       team_prior(base, _classified_finish(base), window=5),
        "team_podium_rate_l10": team_prior(base, _podium_flag(base), window=10, race_stat="max"),
    })

def f_team_ceiling(base):
    """The better car's classified result per race, rolled — potential, not average."""
    return team_prior(base, _classified_finish(base), window=5, race_stat="min") \
             .rename("team_best_finish_l5")

def f_driver_form(base):
    return pd.DataFrame({
        "driver_finish_l5":      rolling_prior(base, "driver_id", _classified_finish(base), window=5),
        "driver_podium_rate_l5": rolling_prior(base, "driver_id", _podium_flag(base), window=5),
    })

def f_driver_podium_rate_era(base):
    """Podium rate over the driver's current era — EWMA, halflife 15 races, so a
    breakout after a slow start isn't dragged down by the distant past. Sits
    between driver_podium_rate_l5 (form) and driver_career_podium_rate (lifetime)."""
    return rolling_prior(base, "driver_id", _podium_flag(base),
                         halflife=15).rename("driver_podium_rate_ewm15")

def f_reliability(base):
    return pd.DataFrame({
        "driver_dnf_rate_l10": rolling_prior(base, "driver_id", "dnf", window=10),
        "team_dnf_rate_l10":   team_prior(base, "dnf", window=10, race_stat="mean"),
    })

# Tier 2
def f_driver_circuit(base):
    """This driver at this track. Few visits -> small window, min_periods=1."""
    return rolling_prior(base, ["driver_id", "circuit_id"], _classified_finish(base),
                         window=3, min_periods=1).rename("driver_circuit_finish")

def f_driver_quali_form(base):
    """Qualifying pace trend — less noisy than race results."""
    return rolling_prior(base, "driver_id", "quali_position", window=5).rename("driver_quali_pos_l5")

def f_positions_gained(base):
    """Weighted places gained over the driver's last 5 *finished* races:
    log(grid) - log(finish) stretches the front (2->1 counts, 15->14 barely).
    DNFs are excluded — not lost on merit. (grid 0 is already remapped to the
    field size in load_base.)"""
    grid = base["grid_position"].astype("float64")
    gain = np.log(grid) - np.log(_classified_finish(base))       # +ve = moved forward
    return rolling_prior(base.assign(_g=gain), "driver_id", "_g",
                         window=5, min_periods=1).rename("driver_gain_l5")

def f_experience(base):
    """Career races so far — a real feature and the rookie/cold-start signal."""
    starts = base.sort_values(["event_date", "round_number"]).groupby("driver_id").cumcount()
    return starts.rename("driver_career_starts")

def f_driver_career_podium_rate(base):
    """Prior podiums / prior starts, over all the driver's earlier races —
    long-run caliber a 5-race window can't capture. Near-constant within a
    season; noisy in the first ~10 starts (pair with driver_career_starts);
    NaN on debut. "Career" = within the loaded history (1991+)."""
    ordered = base.sort_values(["event_date", "round_number"])
    pod = _podium_flag(base).reindex(ordered.index)
    grp = pod.groupby(ordered["driver_id"], sort=False)

    prior_podiums = grp.cumsum() - pod              # podiums strictly before this race
    prior_starts = grp.cumcount()                   # races strictly before this one
    rate = prior_podiums / prior_starts.where(prior_starts > 0)
    return rate.reindex(base.index).rename("driver_career_podium_rate")


HISTORY_FEATURES = [
    # tier 1 — car first, then driver, then reliability
    f_team_form, f_team_ceiling, f_driver_form, f_driver_podium_rate_era, f_reliability,
    # tier 2
    f_driver_circuit, f_driver_quali_form, f_positions_gained,
    f_experience, f_driver_career_podium_rate,
]

# --------------------------------------------------------------------------- #
#  relativizing
# --------------------------------------------------------------------------- #
REL_GROUP_COLS = ["year", "round_number"]

DRIVER_FEATURES = [
    "grid_position", "quali_position", "grid_penalty",
    "driver_finish_l5", "driver_circuit_finish", "driver_quali_pos_l5",
    "driver_podium_rate_l5", "driver_podium_rate_ewm15",
    "driver_dnf_rate_l10", "driver_career_podium_rate",
]

TEAM_FEATURES = [
    "team_finish_l5", "team_best_finish_l5",
    "team_podium_rate_l10", "team_dnf_rate_l10",
]
def relativize(df: pd.DataFrame, driver_cols: list[str], team_cols: list[str]) -> pd.DataFrame:
    """Relativize driver-level stats within the race field,
       and team-level stats within the set of teams in that race."""

    for col in driver_cols:
        grp = df.groupby(REL_GROUP_COLS)[col]
        df[f"{col}_pctile"] = grp.rank(pct=True)

    for col in team_cols:
        team_level = (
            df.drop_duplicates(subset=REL_GROUP_COLS + ["constructor_id"])
              [REL_GROUP_COLS + ["constructor_id", col]]
        )
        team_level[f"{col}_pctile"] = (
            team_level.groupby(REL_GROUP_COLS)[col].rank(pct=True)
        )
        df = df.merge(
            team_level[REL_GROUP_COLS + ["constructor_id", f"{col}_pctile"]],
            on=REL_GROUP_COLS + ["constructor_id"],
            how="left",
        )

    return df

# --------------------------------------------------------------------------- #
#  compose
# --------------------------------------------------------------------------- #

def build_matrix(
    base: pd.DataFrame,
    features: list,
    target: list[str] | None = None,
    min_year: int | None = MODEL_FROM_YEAR,
) -> pd.DataFrame:
    """Identifier columns + the chosen ``features`` (+ the chosen ``target``(s), last).

    ``features``  list of feature functions — each ``base -> named Series / DataFrame``.
    ``target``    keys of ``TARGETS`` (``["podium", "finish_position"]`` …), or
                  ``None``/``[]`` for prediction rows where no outcome is known yet.
    ``min_year``  features are computed on the full ``base`` (so pre-window races
                  feed the rolling windows), then rows before ``min_year`` are
                  dropped. ``None`` keeps every row.
    """
    parts = [fn(base) for fn in features]
    matrix = pd.concat([base[_IDENTIFIERS], *parts], axis=1)
    for name in target or []:
        matrix[name] = TARGETS[name](base)
    if min_year is not None:
        matrix = matrix[matrix["year"] >= min_year]
    return matrix.reset_index(drop=True)


def feature_columns(matrix: pd.DataFrame) -> list[str]:
    """The columns to fit on — not an identifier, not a target."""
    return [c for c in matrix.columns if c not in _IDENTIFIERS and c not in TARGETS]


def features_for_race(year: int, round_number: int) -> pd.DataFrame:
    base = load_base()
    matrix = build_matrix(base, BASIC_FEATURES + HISTORY_FEATURES, target=["podium"])
    matrix = relativize(matrix, driver_cols=DRIVER_FEATURES, team_cols=TEAM_FEATURES)
    return matrix[(matrix["year"] == year) & (matrix["round_number"] == round_number)]

if __name__ == "__main__":
    base = load_base()
    print("base:", base.shape)
    save(base, "feature_base", dir=DATASETS_DIR)

    matrix = build_matrix(base, BASIC_FEATURES + HISTORY_FEATURES, target=["podium", "finish_position"])
    matrix = relativize(matrix, driver_cols=DRIVER_FEATURES, team_cols=TEAM_FEATURES)
    print("matrix:", matrix.shape)
    print("features:", feature_columns(matrix))
    print("podium rate:", round(float(matrix["podium"].mean()), 3))
    save(matrix, "feature_matrix", dir=DATASETS_DIR)

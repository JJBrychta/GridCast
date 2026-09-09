-- RaceCast derived DB — a faithful, rebuildable mirror of raw/.
-- No aggregation beyond `dnf` and the weather rollup; no training-window filter.
--
-- STRICT tables (declared types enforced). Idempotent: every statement is
-- CREATE ... IF NOT EXISTS, so db.connect() can run this on every open.
-- Requires `PRAGMA foreign_keys = ON` per connection (db.connect does it).
--
-- Conventions:
--   *_id   integer surrogate primary key
--   *_ref  stable natural key from the API (slug)
--   dates  ISO-8601 TEXT ('2024-03-02' / '2024-03-02T15:00:00')
--   bools  INTEGER 0/1
--   times  INTEGER milliseconds


CREATE TABLE IF NOT EXISTS seasons (
    season_id  INTEGER PRIMARY KEY,
    year       INTEGER NOT NULL UNIQUE
) STRICT;


CREATE TABLE IF NOT EXISTS circuits (
    circuit_id   INTEGER PRIMARY KEY,
    circuit_ref  TEXT    NOT NULL UNIQUE,   -- Ergast circuitId: "monza", "spa"
    name         TEXT    NOT NULL,          -- Ergast circuitName
    locality     TEXT,
    country      TEXT,
    lat          REAL,
    long         REAL
) STRICT;


CREATE TABLE IF NOT EXISTS events (
    event_id       INTEGER PRIMARY KEY,
    season_id      INTEGER NOT NULL REFERENCES seasons(season_id),
    round_number   INTEGER NOT NULL,
    circuit_id     INTEGER REFERENCES circuits(circuit_id),   -- nullable: a failed circuits fetch shouldn't block the event
    event_name     TEXT    NOT NULL,
    official_name  TEXT,
    event_format   TEXT,                                       -- schedule.EventFormat, verbatim (labels vary by era)
    event_date     TEXT,
    country        TEXT,                                       -- schedule.Country
    location       TEXT,                                       -- schedule.Location (host town, not the circuit)
    UNIQUE (season_id, round_number)
) STRICT;


CREATE TABLE IF NOT EXISTS sessions (
    session_id    INTEGER PRIMARY KEY,
    event_id      INTEGER NOT NULL REFERENCES events(event_id),
    session_type  TEXT    NOT NULL CHECK (session_type IN (
                      'fp1', 'fp2', 'fp3',
                      'qualifying', 'sprint_qualifying', 'sprint', 'race'
                  )),
    session_date  TEXT,
    UNIQUE (event_id, session_type)
) STRICT;


CREATE TABLE IF NOT EXISTS constructors (
    constructor_id  INTEGER PRIMARY KEY,
    team_ref        TEXT    NOT NULL UNIQUE,   -- FastF1 TeamId slug
    name            TEXT    NOT NULL
) STRICT;


CREATE TABLE IF NOT EXISTS drivers (
    driver_id     INTEGER PRIMARY KEY,
    driver_ref    TEXT    NOT NULL UNIQUE,     -- FastF1 DriverId slug: "max_verstappen"
    first_name    TEXT,
    last_name     TEXT,
    abbreviation  TEXT,                        -- e.g. VER — empty/absent for pre-~1980 drivers
    country_code  TEXT                         -- e.g. NED — empty/absent for old drivers
) STRICT;


-- One row per driver per session. Race-only columns are NULL on qualifying rows
-- and vice-versa. `session_type` is denormalised from `sessions` so the common
-- "all race results" filter needs no join. Sprint / sprint_qualifying will reuse
-- the same columns — no new table.
CREATE TABLE IF NOT EXISTS results (
    result_id            INTEGER PRIMARY KEY,
    session_id           INTEGER NOT NULL REFERENCES sessions(session_id),
    session_type         TEXT    NOT NULL,
    driver_id            INTEGER NOT NULL REFERENCES drivers(driver_id),
    constructor_id       INTEGER NOT NULL REFERENCES constructors(constructor_id),
    driver_number        INTEGER,
    position             INTEGER,             -- finish order (race) / classification (qualifying)

    -- race / sprint only:
    classified_position  TEXT,                -- "1".."20" / R / D / W / N
    grid_position        INTEGER,
    status               TEXT,                -- Finished / Lapped / Retired / Accident / Disqualified / Did not start
    dnf                  INTEGER CHECK (dnf IN (0, 1)),   -- derived: classified_position is not a plain integer
    points               REAL,
    laps                 INTEGER,             -- laps completed
    time_ms              INTEGER,             -- race time / gap to leader

    -- qualifying / sprint_qualifying only:
    q1_ms                INTEGER,
    q2_ms                INTEGER,             -- null if knocked out in Q1
    q3_ms                INTEGER,             -- null if knocked out in Q2

    UNIQUE (session_id, driver_id)
) STRICT;


-- 2018+ only. Aggregated by build_db.py from the per-minute weather timeseries
-- in the raw file. No row for sessions without weather.
CREATE TABLE IF NOT EXISTS session_weather (
    session_id      INTEGER PRIMARY KEY REFERENCES sessions(session_id),
    air_temp_min    REAL,
    air_temp_avg    REAL,
    air_temp_max    REAL,
    track_temp_min  REAL,
    track_temp_avg  REAL,
    track_temp_max  REAL,
    humidity_avg    REAL,
    pressure_avg    REAL,
    wind_speed_avg  REAL,
    wind_speed_max  REAL,
    rainfall_any    INTEGER CHECK (rainfall_any IN (0, 1)),
    rainfall_pct    REAL                      -- 0-100, share of the session with rain flagged
) STRICT;


-- Build checkpoint: which raw files (at which content hash) are already loaded.
-- Lives in this DB on purpose — drop the DB, lose the ledger, full rebuild.
CREATE TABLE IF NOT EXISTS ingested_files (
    raw_path      TEXT PRIMARY KEY,
    content_hash  TEXT NOT NULL,
    ingested_at   TEXT NOT NULL
) STRICT;


-- Join / lookup indexes (UNIQUE constraints already index their own columns).
CREATE INDEX IF NOT EXISTS idx_events_season        ON events(season_id);
CREATE INDEX IF NOT EXISTS idx_sessions_event       ON sessions(event_id);
CREATE INDEX IF NOT EXISTS idx_results_driver       ON results(driver_id);
CREATE INDEX IF NOT EXISTS idx_results_constructor  ON results(constructor_id);
CREATE INDEX IF NOT EXISTS idx_results_type         ON results(session_type);

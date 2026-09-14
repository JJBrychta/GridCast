-- Feature-engineering base table: one row per race entry (driver x Grand Prix),
-- 2003+, with that weekend's qualifying pivoted onto the race row.
--
-- IDs (driver_id / constructor_id / circuit_id) are the stable grouping keys for
-- rolling features; the *_ref slugs are the human handles (who a row / prediction
-- is for). Outcome columns (finish_position, points, ...) are the target source,
-- never features. Everything here is known the morning of the race EXCEPT the
-- session_weather columns (see note at the bottom).
--
-- combined = race LEFT JOIN quali, PLUS any quali row with no matching race yet
-- (a race not yet run doesn't exist as a row at all otherwise, not "exists with
-- a NULL outcome") — the live-prediction case: quali done, race hasn't happened.
-- Two UNION ALL branches rather than one "entries" table joined back to both:
-- each branch is a plain two-table LEFT JOIN, which lets SQLite build an
-- automatic covering index for it. Routing both race and quali through a third
-- CTE instead made SQLite fall back to a full nested-loop scan per side — 60s
-- instead of 0.25s on this DB. Confirm with EXPLAIN QUERY PLAN before changing
-- this shape; every branch should show "USING AUTOMATIC COVERING INDEX", not
-- "SCAN ... LEFT-JOIN".
-- grid_position falls back to quali_position when there's no race row yet, so
-- the main pre-race signal survives (only DNS/DNF-before-grid-forms and any
-- late grid penalty are lost — undetectable until the race itself runs).

WITH race AS (
    SELECT s.event_id, r.session_id, s.session_date AS race_date,
           r.driver_id, r.constructor_id,
           r.position            AS finish_position,
           r.classified_position,
           r.grid_position,
           r.status, r.dnf, r.points, r.laps, r.time_ms
    FROM results r
    JOIN sessions s ON s.session_id = r.session_id
    WHERE r.session_type = 'race'
),
quali AS (
    SELECT s.event_id, r.driver_id, r.constructor_id,
           r.position AS quali_position,
           r.q1_ms, r.q2_ms, r.q3_ms
    FROM results r
    JOIN sessions s ON s.session_id = r.session_id
    WHERE r.session_type = 'qualifying'
),
combined AS (
    SELECT
        race.event_id, race.race_date, race.driver_id, race.constructor_id,
        race.grid_position, race.finish_position, race.classified_position,
        race.status, race.dnf, race.points, race.laps, race.time_ms, race.session_id,
        quali.quali_position, quali.q1_ms, quali.q2_ms, quali.q3_ms
    FROM race
    LEFT JOIN quali ON quali.event_id = race.event_id AND quali.driver_id = race.driver_id

    UNION ALL

    -- quali entries with no matching race row: race not yet run, or a
    -- historical DNS/DNQ (e.g. excluded under the 107% rule) that never got a
    -- race-session result at all.
    SELECT
        quali.event_id, NULL AS race_date, quali.driver_id, quali.constructor_id,
        NULL AS grid_position, NULL AS finish_position, NULL AS classified_position,
        NULL AS status, NULL AS dnf, NULL AS points, NULL AS laps, NULL AS time_ms,
        NULL AS session_id,
        quali.quali_position, quali.q1_ms, quali.q2_ms, quali.q3_ms
    FROM quali
    LEFT JOIN race ON race.event_id = quali.event_id AND race.driver_id = quali.driver_id
    WHERE race.event_id IS NULL
)
SELECT
    se.year,
    e.round_number,
    e.event_date,
    e.event_format,                        -- 'conventional' | sprint variants
    combined.race_date,
    -- grouping keys + human handles
    combined.driver_id,        d.driver_ref,
    combined.constructor_id,   c.team_ref,
    e.circuit_id,              ci.circuit_ref,
    -- known before the race (feature inputs)
    COALESCE(combined.grid_position, combined.quali_position) AS grid_position,
    combined.quali_position,
    combined.q1_ms, combined.q2_ms, combined.q3_ms,
    -- outcome (target is derived from this; never a feature) — NULL until the
    -- race has actually been run
    combined.finish_position,
    combined.classified_position,
    combined.status,
    combined.dnf,
    combined.points,
    combined.laps,
    combined.time_ms,
    -- race-day conditions: 2018+, LEFT joined. LEAK for this race — use only to
    -- derive circuit-history features ("has this track run wet before").
    w.rainfall_any, w.rainfall_pct, w.air_temp_avg, w.track_temp_avg
FROM combined
JOIN events       e  ON e.event_id       = combined.event_id
JOIN seasons      se ON se.season_id     = e.season_id
JOIN drivers      d  ON d.driver_id      = combined.driver_id
JOIN constructors c  ON c.constructor_id = combined.constructor_id
LEFT JOIN circuits       ci ON ci.circuit_id = e.circuit_id
LEFT JOIN session_weather w  ON w.session_id = combined.session_id
-- The caller passes the floor (:from_year). It sits *below* the modelling start
-- year so the rolling/career features have a driver's & team's pre-window races;
-- build_matrix trims to the modelling window after the features are computed.
WHERE se.year >= :from_year
ORDER BY e.event_date, grid_position;

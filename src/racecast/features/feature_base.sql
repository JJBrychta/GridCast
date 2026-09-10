-- Feature-engineering base table: one row per race entry (driver x Grand Prix),
-- 2003+, with that weekend's qualifying pivoted onto the race row.
--
-- IDs (driver_id / constructor_id / circuit_id) are the stable grouping keys for
-- rolling features; the *_ref slugs are the human handles (who a row / prediction
-- is for). Outcome columns (finish_position, points, ...) are the target source,
-- never features. Everything here is known the morning of the race EXCEPT the
-- session_weather columns (see note at the bottom).

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
    SELECT s.event_id, r.driver_id,
           r.position AS quali_position,
           r.q1_ms, r.q2_ms, r.q3_ms
    FROM results r
    JOIN sessions s ON s.session_id = r.session_id
    WHERE r.session_type = 'qualifying'
)
SELECT
    se.year,
    e.round_number,
    e.event_date,
    e.event_format,                        -- 'conventional' | sprint variants
    race.race_date,
    -- grouping keys + human handles
    race.driver_id,        d.driver_ref,
    race.constructor_id,   c.team_ref,
    e.circuit_id,          ci.circuit_ref,
    -- known before the race (feature inputs)
    race.grid_position,
    q.quali_position,
    q.q1_ms, q.q2_ms, q.q3_ms,
    -- outcome (target is derived from this; never a feature)
    race.finish_position,
    race.classified_position,
    race.status,
    race.dnf,
    race.points,
    race.laps,
    race.time_ms,
    -- race-day conditions: 2018+, LEFT joined. LEAK for this race — use only to
    -- derive circuit-history features ("has this track run wet before").
    w.rainfall_any, w.rainfall_pct, w.air_temp_avg, w.track_temp_avg
FROM race
JOIN events       e  ON e.event_id       = race.event_id
JOIN seasons      se ON se.season_id     = e.season_id
JOIN drivers      d  ON d.driver_id      = race.driver_id
JOIN constructors c  ON c.constructor_id = race.constructor_id
LEFT JOIN circuits       ci ON ci.circuit_id = e.circuit_id
LEFT JOIN quali          q  ON q.event_id = race.event_id AND q.driver_id = race.driver_id
LEFT JOIN session_weather w  ON w.session_id = race.session_id
-- The caller passes the floor (:from_year). It sits *below* the modelling start
-- year so the rolling/career features have a driver's & team's pre-window races;
-- build_matrix trims to the modelling window after the features are computed.
WHERE se.year >= :from_year
ORDER BY e.event_date, race.grid_position;

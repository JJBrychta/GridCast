# RaceCast — data scraping layer

Pulls F1 results from the FastF1 / Ergast(Jolpica) APIs into a local `raw/` archive,
then mirrors it into a queryable SQLite DB. Feature engineering is a later step.

```
universe  ──►  session_loader  ──►  build_db  ──►  (features / model)
schedules      raw/**/*.json         data/racecast.sqlite
```

Everything is resumable: kill it, hit a rate limit, Ctrl-C — nothing is left
half-written, and re-running picks up exactly where it stopped.

---

## Step 1 — Universe (`universe.py`)

Produces the list of things to fetch: one **`Unit(season, round, session_type)`**
per session that has already started.

- For each season `FIRST_SEASON…now`, load the event schedule.
  - Past seasons: read the cached `raw/<season>/schedule.json`.
  - Current season: always re-fetched (GPs get added mid-season).
  - Alongside it, `raw/<season>/circuits.json` is fetched from Ergast
    (`get_race_schedule`) — `fastf1.get_event_schedule` drops circuit identity,
    keeping only `Country` / `Location` (a town). Ergast has the proper
    `circuitId` slug + full name; the build step needs it, `iter_units` doesn't.
  - Both files are the only thing the universe step persists.
- `iter_units` walks each event's five session slots (`Session1…Session5`),
  matches the slot **name** against `SESSION_NAME_TO_TYPE`
  (`"Qualifying"→"qualifying"`, `"Race"→"race"`), and yields a Unit only once
  that slot's `SessionNDateUtc` is in the past.

The universe is a `list[Unit]` recomputed every run — never stored.

`fetch_delay` (default `SCHEDULE_FETCH_DELAY`) throttles uncached schedule fetches.

---

## Step 2 — Session loader (`session_loader.py`)

```
todo = universe  −  {units whose raw file already exists}
```

A unit is **done** iff `raw/<season>/round-NN/<session_type>.json` exists. No
ledger, no status DB — the filesystem *is* the resume state.

For each `todo` unit: `get_session(...).load(laps=False, telemetry=False,
weather=<season≥2018>, messages=False)`, then classify:

| Outcome | Condition | Writes |
|---|---|---|
| `ok` | results present | `{meta, status:"ok", results:[…], weather:[…]}` |
| `no_data` | empty **and** session older than `RESULTS_LAG_DAYS` | `{meta, status:"no_data", results:[]}` marker |
| `retry` | empty and future / within the results-posting lag | nothing — retried next run |

- **All result columns are stored verbatim.** Column selection is the build
  step's job. Deciding later that we want a field we didn't parse never means
  re-scraping — and the FastF1 HTTP cache sits under that as a second backstop.
- **`weather`** is the per-minute timeseries (`session.weather_data`), verbatim.
  `[]` for pre-`WEATHER_FROM_SEASON` (2018) sessions — no request is made for
  those. The build step aggregates it to a `session_weather` row.
  `ok` files fetched *before* weather support lack the key — `make backfill-weather`
  re-loads only those 2018+ sessions and adds it (idempotent, rate-limit-safe).
- Writes are atomic (`tempfile` + `os.replace`): a raw file is complete or
  absent, never partial.
- The `no_data` marker matters because Ergast genuinely has no qualifying data
  before ~2003 (~800 units). Without the marker those re-fetch on every run
  forever and the backfill never "completes".

---

## Step 3 — Build (`build_db.py`)

Mirrors `raw/` into `data/racecast.sqlite` (schema in `schema.sql` / `schema.dbml`).
A faithful, disposable copy — no aggregation beyond `dnf` and the weather rollup,
no training-window filter. `make clean-db && make build-db` is the "migration".

Two passes:

1. **`raw/<season>/schedule.json` + `circuits.json`** → `seasons`, `circuits`,
   `events`, `sessions` (qualifying + race rows only)
2. **`raw/<season>/round-NN/*.json`** → `drivers`, `constructors`, `results`
   (one table, `session_type`-discriminated), `session_weather` (aggregated from
   the per-minute `weather` array; 2018+ only)

Idempotent, resumable:

- **`ingested_files(raw_path, content_hash)`** ledger — a file is reprocessed
  only if new or its hash changed (provisional→official re-fetch, weather
  backfill, a new GP in the current schedule). One transaction per file.
- **`DimCache`** — `{ref → id}` maps loaded once; entity names are last-write-wins
  but only written when they actually change.
- `dnf` = `classified_position` is a letter code (`R`/`D`/`W`/`N`), not a number.
- Pass 2 looks up `event_id` / `session_id` from the DB (built by pass 1, this
  run or a previous one). A session file whose schedule isn't built yet is
  skipped **without** being marked, so it's retried next run.

~37 k `results` rows for 1950–now; a full rebuild is a few seconds.

---

## Problems that had to be solved

### 1. FastF1 swallows the 429

FastF1 handles rate limits **inconsistently**:

| Call | On a 429 |
|---|---|
| `get_event_schedule` | raises `ValueError("Failed to load any schedule data")` once every backend fails |
| `session.load()` | `requests_cache` (`stale_if_error=True`) + FastF1's `soft_exceptions` catch-all **swallow it** — it logs the error and **returns an empty result with no exception** |

If we trusted that empty result, `_decide` would write a **false `no_data`
marker** — and because the marker permanently stops re-fetching, that's silent,
irreversible data loss.

**Fix** (`net.py`): `with_retries` attaches a logging handler
(`_RateLimitLogProbe`) to the root logger for the duration of each call. If any
log record — its message *or* its attached exception — contains `"429"` /
`"too many requests"`, it raises `RateLimited`, **even when `fn()` returned
normally**. Schedule 429 and session 429 now behave identically: the run stops
cleanly, nothing is written, a re-run resumes.

**Constraint:** the `fastf1` / `requests_cache` loggers must stay at `WARNING`
or below. Muting them (e.g. `setLevel(CRITICAL)` in `data_pipeline.py`) blinds
the probe.

**Rejected alternatives:**
- `Cache._requests_session_cached.settings.stale_if_error = False` — only
  disables layer 1; `session.load()`'s own catch-all still swallows.
- `FASTF1_DEBUG=1` / `LoggingManager.debug = True` — disables the catch-all, but
  also makes every pre-~2018 session raise `SessionNotAvailableError` (their
  `session_info` isn't on Ergast), which would lose ~50 years of race results.

### 2. "Genuinely empty" vs "rate-limited empty" look identical

A real empty session (1950s qualifying) and a 429'd one produce the *same*
empty DataFrame and the *same* `Failed to load driver list and session results!`
warning. The only distinguishing signal is the exception text — `429 Client
Error` vs `No data for this session`. So the probe keys specifically on `"429"`,
**not** on `"failed"` (FastF1 logs "failed" for the benign case too).

### 3. Rate limit → stop, don't hammer

On `RateLimited`, `fetch()` breaks immediately rather than backing off against a
closed door. Jolpica's limit is ~500 requests/hour; a re-run an hour later
resumes from the first missing file. `SESSION_FETCH_DELAY` throttles between
session loads, `SCHEDULE_FETCH_DELAY` between schedule fetches.

### 4. Schedule datetime format depends on the source

- Fresh `get_event_schedule` → `datetime64` columns.
- Cached `schedule.json` (written with `to_json`) → epoch-millis **integers**.

`_normalize_dates` runs `pd.to_datetime(col, unit="ms")` on both — a no-op on
`datetime64`, the correct conversion for millis — so `iter_units`' date gate
works regardless of where the schedule came from.

### 5. Qualifying isn't a fixed session slot

Conventional weekend: `Session4` = Qualifying. Sprint 2023: `Session2`. Sprint
2024+: `Session4`. So `iter_units` scans all five slots and matches by **name**,
never by position.

### 6. Old seasons re-request `session_info` every run

Pre-~2018 seasons have no `session_info` (an F1 live-timing concept). FastF1
requests it anyway, gets `403`/`404` from `livetiming.formula1.com`, and — since
the cache only stores `200`s — **re-requests it on every run**. These:

- hit `livetiming.formula1.com`, **not** Jolpica, so they don't count against
  the ~500/h Ergast limit — but they inflate the `req/min` stat and add ~4
  failing round-trips per old session.
- don't affect data: the Ergast **results** *are* cached (`from_cache=True`), so
  re-runs are 0-cost for the data we actually want.

Fully avoidable by fetching through `fastf1.ergast.Ergast` directly (1 request
per unit, every season, no `session_info`) — at the cost of a different column
schema. Not done.

### 7. Empty schedule frame

If `get_event_schedule` ever returns an empty frame instead of raising, it is
**not** cached — otherwise a past season would trust the empty file forever.
It's returned unwritten and retried next run.

---

## Observability

Per unit:

```
1974-R15-race  ->  ok    [312 loaded · 41 reqs · 720s · 3 req/min]
```

End of run:

```
loaded 312 units, 41 API requests, in 720s
  = 96s API + 624s throttle (0.3s/unit API, 3 req/min)
```

`reqs` is `fastf1.req.Cache._request_counter` — it counts cache hits too, so
`req/min` is an **upper bound** on real server load. Under ~8/min you're
definitely safe against Jolpica's 500/h.

Output is colour-coded (`ok` green, `no_data`/`retry` yellow, `error` /
rate-limit stop red). While a unit is throttling + fetching, a transient
`  <unit>  …  (Ns throttle + fetch)` line shows, overwritten by the result.
Both are disabled automatically when stdout isn't a TTY or `NO_COLOR` is set.
FastF1's own INFO/WARNING chatter is silenced by `enable_cache()`
(`fastf1.set_log_level("ERROR")`) — this only lowers FastF1's console handler,
so net.py's 429 log-probe is unaffected.

---

## Tests

```bash
uv run pytest                       # unit tests only (no network)
uv run pytest --run-network         # + live FastF1/Ergast integration tests
uv run pytest --cov=racecast        # with coverage
```

Unit tests never touch the network or the real cache — schedules are hand-built
DataFrames, sessions are `FakeSession` stubs, `time.sleep` is stubbed, and `raw/`
points at a `tmp_path`. The handful of tests that hit the real API are
`@pytest.mark.network` and skipped unless `--run-network` is passed; they exist
to catch upstream schema drift.

Covered in depth: `iter_units` (the date gate, name-vs-slot matching,
quali-before-race), `with_retries` (all four 429 paths incl. the swallowed one),
`_decide` (ok / no_data / retry), and `fetch` end to end (resume, rate-limit
stop, atomic writes).

## Config (`config.py`)

| Key | Meaning |
|---|---|
| `FIRST_SEASON` | 1950 |
| `SCHEDULE_FETCH_DELAY` / `SESSION_FETCH_DELAY` | throttle between uncached fetches |
| `RESULTS_LAG_DAYS` | empty result younger than this → `retry`, not `no_data` |
| `SESSION_NAME_TO_TYPE` | which schedule sessions to collect + their slugs |
| `RAW_DIR`, `FASTF1_CACHE` | local paths (both gitignored) |

---

## Running

```bash
make pipeline           # universe -> session_loader, 1950 -> current season
make fetch              # session_loader only
make universe           # universe only
make backfill-weather   # add weather to ok files fetched before weather support
make build-db              # step 3: raw/ -> data/racecast.sqlite (idempotent)
make init-db            # just create the empty DB with the schema
make clean-db           # rm data/racecast.sqlite
make clean-raw          # wipe raw/ (keeps .gitkeep)   — WARNING: discards fetched data
make clean-cache        # wipe fastf1_cache/
```

The backfill takes many runs (rate limits). Each run re-derives the universe
(cheap — schedules cached) and fetches only missing files. When `to fetch`
drops to a handful of `retry` units, `raw/` is effectively complete.

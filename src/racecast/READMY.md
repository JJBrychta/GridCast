# Universe and Session Loader

## Universe
Prepares folders for individual seasons raw data 
and generates input data for session loader that calls fastf1 api.
Input parameters data for endpoint fastf1.get_session(...) are year (Unit.season),
and gp round or name. (For simplicity we use gp round number = int) last is editable
session type that controls what all session we want (defined in config)

## Session loader
Takes list of units(season, round, session_type) and with same strategy
as in universe calls fastf1 api with retry and gradual backoff function.
Stores raw session data in "raw" folder -> no need for recalling for 
raw input from fastf1 api. One backfall is changing what colums from the raw
we want to store means rerunning the session loader for all sessions.
The selected data f

"""
artifactdetect -- artifact detection pipeline for LFP (local field
potential) recordings. Works with data from any device that can be
loaded into the expected timestamp + channel-columns CSV format (see
prepare_lfp_csv for converting raw device data into that shape).

Exposes the main entry points at the top level, so:
    from artifactdetect import detect_artifacts, run_from_data
works directly, without needing to know the internal file structure.


### For automated or deployed workflows: `detect_artifacts()` directly

`run_from_data()` always writes a CSV and a figure for every flagged
window -- ideal for minimal use, but can be inconvenient for larger runs.

`detect_artifacts()` is the pure detection function underneath
`run_from_data()` -- same required inputs (`lfp`, `t_s`, `channel_labels`,
`fs_nominal`, optionally `gaps_s`), but it does detection ONLY: no CSV,
no figures, no files written anywhere -- this is the function for
deployed/automated contexts

```python
from artifactdetect import load_lfp_csv, detect_artifacts

lfp, t_s, fs, labels, gaps = load_lfp_csv("session.csv", settings_path="Settings.json")

windows, debug = detect_artifacts(lfp, t_s, labels, fs, gaps_s=gaps)
```

`windows` is the identical list-of-dicts structure documented above for
`run_from_data()`. `debug` is a dict of the
underlying per-channel z-score arrays (`z_local`, `z_global`, `z_per_ch`,
`flagged_local`, `flagged_global`, plus `starts` and `channel_labels`) --
only needed if you want to build your own custom visualization on top

If you later want the full figure-and-CSV output for a specific session
-- e.g. to review what a deployed run actually flagged -- rerun that one
session through `run_from_data()` separately, rather than generating
that output on every automated run by default.
"""

from .artifact_finder import (
    load_lfp_csv,
    detect_artifacts,
    run_from_data,
    run,
    broadband_power,
    rolling_mad_z,
    mad_z,
    rolling_mad_band,
    compute_global_voltage_baseline,
    Z_THRESH,
    WIN_S,
    STEP_S,
    ROLL_S,
)
from .prepare_lfp_csv import prepare_lfp_csv, load_lfp_data

__version__ = "0.1.0"

__all__ = [
    "load_lfp_csv",
    "detect_artifacts",
    "run_from_data",
    "run",
    "broadband_power",
    "rolling_mad_z",
    "mad_z",
    "rolling_mad_band",
    "compute_global_voltage_baseline",
    "Z_THRESH",
    "WIN_S",
    "STEP_S",
    "ROLL_S",
    "prepare_lfp_csv",
    "load_lfp_data",
]

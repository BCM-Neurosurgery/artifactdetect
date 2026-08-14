# artifactdetect

Retrospective artifact-detection pipeline for local field potential (LFP)
recordings. Works with data from any device that can be loaded into the
expected timestamp + channel-columns CSV format -- see
`prepare_lfp_csv`/`load_lfp_data` for converting raw device data into
that shape. Flags time windows with unusual broadband
power, using two complementary statistical methods, and produces a
diagnostic figure plus a structured CSV for every flag.

**Not source-attributed -- does not distinguish between movement artifact, stimulation artifacts, etc.

## Security & privacy notes

- **Don't use patient-identifiable filenames.** The tool prints the input
  file path directly to the console (`loaded <path>: ...`), and that
  path can end up in terminal scrollback, screenshots, or copy-pasted
  error reports. Refrain from using PII in any filenames.


## Install

### 1. Create an isolated conda environment (recommended)

```bash
conda create -n artifactdetect python=3.10
conda activate artifactdetect
```

This keeps this package's dependencies (numpy, pandas, scipy, matplotlib)
separate from anything else installed on your system or in other conda
environments -- installing this won't touch or conflict with a different
project's package versions. Anyone without conda already set
up can install it via [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

### 2. Install the package

**If you've cloned the repo:**
```bash
git clone https://github.com/BCM-Neurosurgery/artifactdetect.git
cd artifactdetect
pip install -e .
```
`-e` installs in editable mode -- edits to the source take effect
immediately. This is the right choice if you might
want to read or modify the code.

**If you just want to use it, without cloning:**
```bash
pip install git+https://github.com/BCM-Neurosurgery/artifactdetect.git
```
This installs directly from GitHub in one step, for anyone who just
wants to run the tool, not edit it.


## Usage

> **If building this into an automated or deployed pipeline** Use
> `detect_artifacts()`, not `run_from_data()` -- same detection, but no
> CSV or figures written on every run. See "For automated or deployed
> workflows" in __init__.py

### Command line

```bash
artifact-finder <lfp.csv> --settings <Settings.json> --out <output_dir>
```

- `<lfp.csv>` **(required)** -- a `timestamp` column (integer ms, unix epoch)
  plus any number of channel columns. Column names are auto-detected, no
  fixed naming required.
- `--settings` *(optional)* -- a JSON matching Medtronic's `BaseSettings.json`
  structure, for the true sample rate and channel labels. Falls back to
  auto-detection from the CSV if omitted or malformed.
- `--out` *(optional)* -- output directory, default `artifact_output`.

If your raw data isn't already in this CSV format (e.g. it's a different
device's export, or you have raw arrays + a sample rate), see
"Loading raw data from any device" below.

### From another script or notebook

`run_from_data()` is the main function -- it does the whole pipeline
(detection, CSV, per-window figures, confidence subfolders) from
already-loaded data. Use once you've confirmed the
CLI works on your data and want to call the pipeline directly from your
own analysis code instead.

```python
from artifactdetect import load_lfp_csv, run_from_data

lfp, t_s, fs, labels, gaps = load_lfp_csv("session.csv", settings_path="Settings.json")

windows = run_from_data(lfp, t_s, fs, labels, gaps_s=gaps, out_dir="my_output")
```

**Required inputs:**

| parameter | type | what it is |
|---|---|---|
| `lfp` | ndarray, shape `(n_channels, n_samples)` | the LFP signal itself |
| `t_s` | ndarray, shape `(n_samples,)` | timestamp of each sample, in seconds from the first sample -- not raw sample indices, actual elapsed time |
| `fs_nominal` | float | sample rate in Hz |
| `channel_labels` | list of str, length `n_channels` | a name per channel, e.g. `["+2-0", "+3-1", "+9-8", "+11-10"]` |

**Optional inputs:**

| parameter | type | default | what it is |
|---|---|---|---|
| `gaps_s` | list of `(start_s, end_s, gap_ms)` tuples | `None` | known data dropouts, so windows overlapping a gap aren't flagged; `load_lfp_csv` produces this for you automatically |
| `out_dir` | str | `"artifact_output"` | where the CSV, figures, and confidence output folders get written |

All four required inputs are exactly what `load_lfp_csv()` (or
`load_lfp_data()`, if you're loading raw device data rather than a CSV --
see below) returns, in the same order

**Return value:** a list of dicts, one per flagged window, sorted by
start time. Each dict has these keys:

| key | type | meaning |
|---|---|---|
| `flag_start`, `flag_end` | float | window bounds, in seconds |
| `first_crossing_time_s` | float | time of the first threshold crossing in this window |
| `crossing_times_s` | list of float | every distinct crossing time (a signal dropping back under threshold and re-crossing counts as a new one) |
| `n_mad_crossings` | int | `len(crossing_times_s)` |
| `max_mad_z_broadband` | float | the strongest z-score reached by any contributing channel |
| `trigger_channel` | str | which channel had the strongest evidence |
| `channels_crossing` | list of str | every channel that crossed threshold in this window |
| `detection_methods` | list of str | `"local"`, `"global"`, or `"both"`, aligned index-for-index with `channels_crossing` |
| `peak_local_z`, `peak_global_z` | list of float | each contributing channel's peak z-score under that method, same alignment |
| `confidence` | str | `"single-channel"`, `"high"`, or `"priority-agreement"` |

This is the same data written to `artifact_times.csv` -- if you're doing
your own downstream analysis, you can work directly with this returned
list instead of re-reading the CSV back in.

Once installed, everything is importable directly from `artifactdetect`
without needing to know the internal file layout (`detect_artifacts`,
`Z_THRESH`, etc. are all available at the top level -- see
`artifactdetect/__init__.py` for the full list).

### Loading raw data from any device

## Package structure

```
artifactdetect/
├── pyproject.toml              <- package metadata + dependencies
├── README.md
├── .gitignore
└── artifactdetect/                <- actual importable package
    ├── __init__.py               <- exposes the public API
    ├── artifact_finder.py        <- detection logic, plotting, CLI
    └── prepare_lfp_csv.py        <- data-loading helper, does not require LFP data from a specific device
```

## How detection works:

1. Broadband power (>40 Hz) is computed per channel in a sliding 1.0s
   window (0.5s hop).
2. That series is scored two ways: a LOCAL rolling MAD z-score (5.0s
   centered window -- sensitive to brief transient events, not to long
   sustained events) and a GLOBAL whole-session MAD z-score (sensitive to
   sustained events, but only if they're a minority of the session). A
   channel is flagged if either crosses z > 6.5.
3. Flagged channels are overlaid in time to find cross-channel agreement,
   producing three confidence tiers: `single-channel`, `high`, and
   `priority-agreement` (based on whether channels 3 & 4 specifically
   agree, and how). Channels 3 & 4 are currently designated as "priority channels" and determine "priority-agreement flagged windows", which are considered highest-confidence artifacts. If more than one channel agree outside of this (e.g. Ch1 and Ch2), this is a "high-confidence" flagged window. Finally, if only one channel identifies a threshold crossing, it is labeled as a "single-channel" flagged window. All three flagged window types are separated into distinct output folders.
4. Every distinct threshold-crossing event is identified (a signal
   dropping back under threshold and re-crossing counts as a new event,
   not a continuation).

Each flagged window gets a diagnostic figure: a clean overview of all
channels, a 2x2 grid with each channel's own signal plus its LOCAL
(dashed) and GLOBAL (solid, darker-shaded) MAD bands in voltage units,
the same overview with crossings thickened, a detection-method strip, and
a spectrogram.

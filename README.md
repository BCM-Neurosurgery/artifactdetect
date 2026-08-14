# artifactdetect

Retrospective artifact-detection pipeline for local field potential (LFP)
recordings. Works with data from any device that can be loaded into the
expected timestamp + channel-columns CSV format -- see
`prepare_lfp_csv`/`load_lfp_data` for converting raw device data into
that shape. Includes optional support for Medtronic Summit RC+S session
settings files specifically, but the detection pipeline itself does not
require or assume RC+S data. Flags time windows with unusual broadband
power, using two complementary statistical methods, and produces a
diagnostic figure plus a structured CSV for every flag.

**Not source-attributed (a flag means "statistically unusual," not "this
is movement")** -- not intended for clinical use.

## Security & privacy notes

- **Don't use patient-identifiable filenames.** The tool prints the input
  file path directly to the console (`loaded <path>: ...`), and that
  path can end up in terminal scrollback, screenshots, or copy-pasted
  error reports.
- **Dependencies are currently unpinned.** Beyond the
  reproducibility issue already noted, this also means there's no fixed
  record of exactly which numpy/pandas/scipy/matplotlib version was
  running for a given result -- worth knowing if a dependency
  vulnerability is ever disclosed and you need to check whether a past
  run was affected.
- **`--out` / `out_dir` is not sanitized.** Low-risk in normal use (an
  interactive, single-user CLI writing to your own filesystem), but this
  tool was not built to safely accept an output path from another,
  untrusted program.
- **Installing via `pip install git+<url>` runs code at install time.**
  This is true of any pip-installable Python package, not something
  specific to this tool -- but it means anyone with push access to this
  repo can run code on every lab member's machine the next time they
  install or reinstall.

## Install

### 1. Create an isolated conda environment (recommended)

```bash
conda create -n artifactdetect python=3.10
conda activate artifactdetect
```

This keeps this package's dependencies (numpy, pandas, scipy, matplotlib)
separate from anything else installed on your system or in other conda
environments -- installing this won't touch or conflict with a different
project's package versions. Anyone in the lab without conda already set
up can install it via [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

### 2. Install the package

**If you've cloned the repo:**
```bash
git clone <lab-github-url>
cd artifactdetect
pip install -e .
```
`-e` installs in editable mode -- edits to the source take effect
immediately, no reinstall needed. This is the right choice if you might
want to read or modify the code.

**If you just want to use it, without cloning:**
```bash
pip install git+<lab-github-url>.git
```
This installs directly from GitHub in one step -- no manual clone, no
local copy of the source to manage. Right choice for anyone who just
wants to run the tool, not edit it. (Replace `<lab-github-url>` with the
actual repo URL once it's created on the lab GitHub.)

**This repo is private, lab-members-only** -- this method requires GitHub
authentication set up first (an SSH key or a personal access token), or
it will fail with a permission error. Same requirement applies to the
clone-first method above; a private repo can't be cloned or pip-installed
anonymously either way.

Either way, this adds an `artifact-finder` command to your PATH inside
the active conda environment.

## Usage

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

```python
from artifactdetect import load_lfp_csv, run_from_data

lfp, t_s, fs, labels, gaps = load_lfp_csv("session.csv", settings_path="Settings.json")

windows = run_from_data(lfp, t_s, fs, labels, gaps_s=gaps, out_dir="my_output")
```

`run_from_data()` does the whole pipeline -- detection, CSV, per-window
figures, confidence subfolders -- from already-loaded data. Once
installed, everything is importable directly from `artifactdetect`
without needing to know the internal file layout
(`detect_artifacts`, `Z_THRESH`, etc. are all available at the top level
-- see `artifactdetect/__init__.py` for the full list).

### Loading raw data from any device

`prepare_lfp_csv` / `load_lfp_data` convert raw LFP data (a dict of
channel arrays, a bare numpy array, or a DataFrame) into the CSV format
above, or directly into the in-memory pipeline format, without needing
real wall-clock timestamps (most devices don't provide them -- only a
sample rate is required):

```python
from artifactdetect import prepare_lfp_csv, load_lfp_data

# write a CSV file
prepare_lfp_csv({"ch1": arr1, "ch2": arr2}, "session.csv", fs=500.0)

# or skip the file entirely, go straight to in-memory pipeline format
lfp, t_s, fs, labels, gaps = load_lfp_data({"ch1": arr1, "ch2": arr2}, fs=500.0)
```

## Package structure

```
artifactdetect/
├── pyproject.toml              <- package metadata + dependencies
├── README.md
├── .gitignore
├── docs/
│   └── pipeline_walkthrough.pptx
└── artifactdetect/                <- the actual importable package
    ├── __init__.py               <- exposes the public API
    ├── artifact_finder.py        <- detection logic, plotting, CLI
    └── prepare_lfp_csv.py        <- device-agnostic data-loading helper
```

## How detection works, briefly

1. Broadband power (>40 Hz) is computed per channel in a sliding 1.0s
   window (0.5s hop).
2. That series is scored two ways: a LOCAL rolling MAD z-score (5.0s
   centered window -- sensitive to brief transients, blind to long
   sustained events) and a GLOBAL whole-session MAD z-score (sensitive to
   sustained events, but only if they're a minority of the session). A
   channel is flagged if either crosses z > 6.5.
3. Flagged channels are overlaid in time to find cross-channel agreement,
   producing three confidence tiers: `single-channel`, `high`, and
   `priority-agreement` (based on whether channels 3 & 4 specifically
   agree, and how).
4. Every distinct threshold-crossing event is identified (a signal
   dropping back under threshold and re-crossing counts as a new event,
   not a continuation).

Each flagged window gets a diagnostic figure: a clean overview of all
channels, a 2x2 grid with each channel's own signal plus its LOCAL
(dashed) and GLOBAL (solid, darker-shaded) MAD bands in voltage units,
the same overview with crossings thickened, a detection-method strip, and
a spectrogram.

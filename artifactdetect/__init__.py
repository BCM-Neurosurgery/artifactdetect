"""
artifactdetect -- artifact detection pipeline for LFP (local field
potential) recordings. Works with data from any device that can be
loaded into the expected timestamp + channel-columns CSV format (see
prepare_lfp_csv for converting raw device data into that shape). Includes
optional support for Medtronic Summit RC+S session settings files
specifically, but detection itself does not require or assume RC+S data.

Exposes the main entry points at the top level, so:
    from artifactdetect import detect_artifacts, run_from_data
works directly, without needing to know the internal file structure.
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

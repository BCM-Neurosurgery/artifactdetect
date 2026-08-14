"""
prepare_lfp_csv.py

Helper for Step 1 of the pipeline (load LFP data). artifact_finder.py's
load_lfp_csv() expects a CSV with a `timestamp` column (integer
milliseconds) plus one column per channel -- anything else needs to get
converted into that shape first. This module is that conversion step,
built to work from whatever a device actually gives you -- it does not
assume any particular device or manufacturer.

Most devices don't hand you real wall-clock timestamps -- they give you
raw channel arrays and a sample rate. This function accepts either:
  - real timestamps, if you have them, or
  - just a sample rate, and it generates a consistent timestamp column
    from sample count (starting at 0 ms by default -- the actual pipeline
    only needs correctly-SPACED timestamps, not true wall-clock time).

Accepts three common shapes of input data:
  1. dict of {channel_name: 1D array}          -- most device APIs
  2. numpy array, shape (n_channels, n_samples) or (n_samples, n_channels)
  3. a pandas DataFrame you already have loaded, with or without a time column
"""

import numpy as np
import pandas as pd


def prepare_lfp_csv(data, out_path, fs=None, channel_names=None,
                    start_time_ms=0, time_col=None):
    """
    Convert raw LFP data (from any device) into the CSV format
    artifact_finder.py expects: a `timestamp` column (int ms) + one
    column per channel.

    Parameters
    ----------
    data : dict, np.ndarray, or pd.DataFrame
        - dict: {channel_name: 1D array}. All arrays must be the same length.
        - np.ndarray: 2D, either (n_channels, n_samples) or
          (n_samples, n_channels) -- shape is auto-detected using
          `channel_names` length if given, otherwise assumes the longer
          axis is time.
        - pd.DataFrame: used as-is for channel columns; if it doesn't
          already have a proper timestamp column, pass `time_col` (the
          existing time-like column to convert) or leave both `time_col`
          and real timestamps out and supply `fs` to generate one.
    out_path : str
        Where to write the resulting CSV.
    fs : float, optional
        Sample rate in Hz. REQUIRED unless `data` is a DataFrame that
        already has a valid millisecond timestamp column.
    channel_names : list[str], optional
        Channel column names. If omitted: dict keys are used, DataFrame
        columns are used as-is, and a bare ndarray gets auto-named
        ch1, ch2, ... in device-agnostic fashion.
    start_time_ms : int, default 0
        Starting timestamp, in integer milliseconds. Defaults to 0 --
        artifact_finder.py only needs consistently-spaced timestamps, not
        true wall-clock time, so 0 is a safe default when a device gives
        you no real clock.
    time_col : str, optional
        If `data` is a DataFrame with an existing time-like column under
        a different name (e.g. "time_s", "t"), name it here and it will
        be converted to a proper `timestamp` (ms) column. Assumed to be
        in SECONDS unless the column name contains "ms".

    Returns
    -------
    str : the out_path written to (for convenience / chaining)
    """
    if isinstance(data, dict):
        channel_names = channel_names or list(data.keys())
        lengths = {len(v) for v in data.values()}
        if len(lengths) != 1:
            raise ValueError(
                f"All channel arrays must be the same length, got {lengths}")
        n_samples = lengths.pop()
        df = pd.DataFrame({name: np.asarray(data[name]) for name in channel_names})

    elif isinstance(data, np.ndarray):
        if data.ndim != 2:
            raise ValueError(f"Expected a 2D array, got shape {data.shape}")
        if channel_names and len(channel_names) in data.shape:
            n_ch = len(channel_names)
            data2 = data if data.shape[0] == n_ch else data.T
        else:
            data2 = data if data.shape[0] <= data.shape[1] else data.T
            n_ch = data2.shape[0]
        channel_names = channel_names or [f"ch{i+1}" for i in range(n_ch)]
        n_samples = data2.shape[1]
        df = pd.DataFrame({name: data2[i] for i, name in enumerate(channel_names)})

    elif isinstance(data, pd.DataFrame):
        df = data.copy()
        n_samples = len(df)
        if time_col and time_col in df.columns:
            t = df[time_col].values.astype(float)
            if "ms" not in time_col.lower():
                t = t * 1000.0
            df = df.drop(columns=[time_col])
            timestamps = (start_time_ms + (t - t[0])).astype(np.int64)
            df.insert(0, "timestamp", timestamps)
            df.to_csv(out_path, index=False)
            return out_path
        elif "timestamp" in df.columns:
            df.to_csv(out_path, index=False)
            return out_path

    else:
        raise TypeError(f"Unsupported data type: {type(data)}")

    if fs is None:
        raise ValueError(
            "fs (sample rate in Hz) is required when no real timestamp "
            "column is available.")
    dt_ms = 1000.0 / fs
    timestamps = (start_time_ms + np.arange(n_samples) * dt_ms).astype(np.int64)
    df.insert(0, "timestamp", timestamps)
    df.to_csv(out_path, index=False)
    return out_path


def load_lfp_data(data, fs=None, channel_names=None, start_time_ms=0,
                  time_col=None, settings_path=None):
    """
    Same inputs as prepare_lfp_csv, but skips the disk round-trip: returns
    the data directly in the exact tuple format artifact_finder.py's
    load_lfp_csv() produces -- (lfp, t_s, fs, channel_labels, gaps) --
    ready to hand straight to detect_artifacts() in a notebook or script
    without ever writing a persistent file.

    Internally, this still goes through a TEMPORARY csv and calls the
    real load_lfp_csv() on it, rather than reimplementing that parsing
    logic a second time -- so gap detection, sample-rate estimation, and
    channel labeling all stay identical to what the CLI actually does.
    Use this when you want the data in memory; use prepare_lfp_csv()
    directly when you want a CSV file you can keep, share, or feed to
    the CLI.

    Returns
    -------
    lfp : ndarray (n_channels, n_samples)
    t_s : ndarray (n_samples,), seconds from first sample
    fs_nominal : float
    channel_labels : list[str]
    gaps_s : list[(start_s, end_s, gap_ms)]
    """
    import tempfile
    import os as _os

    try:
        from .artifact_finder import load_lfp_csv
    except ImportError:
        import sys as _sys
        _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
        from artifact_finder import load_lfp_csv

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        prepare_lfp_csv(data, tmp_path, fs=fs, channel_names=channel_names,
                        start_time_ms=start_time_ms, time_col=time_col)
        return load_lfp_csv(tmp_path, settings_path=settings_path)
    finally:
        if _os.path.exists(tmp_path):
            _os.remove(tmp_path)



    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--demo", action="store_true",
                    help="run a self-test with synthetic multi-channel data")
    args = ap.parse_args()

    if args.demo:
        rng = np.random.default_rng(0)
        fs = 250.0
        n_samples = 5000
        raw = {f"channel_{i}": rng.standard_normal(n_samples) for i in range(4)}
        out = prepare_lfp_csv(raw, "/tmp/demo_lfp.csv", fs=fs)
        print(f"wrote {out}")

        check = pd.read_csv(out)
        print(check.head())
        print("dtypes:\n", check.dtypes)
        dt = check["timestamp"].diff().dropna().unique()
        print("timestamp deltas (ms):", dt[:5])

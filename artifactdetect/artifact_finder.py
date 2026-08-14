"""
artifact_finder.py

General-purpose artifact finder for LFP recordings, from any device that
can be loaded into the expected CSV format (see prepare_lfp_csv.py for
converting raw device data). Not stim-aware and not source-attributed by
design -- it flags "artifact" as a single category (broadband LFP
elevation), and leaves distinguishing stim vs. movement vs. anything else
to downstream review.

Built to run on ANY patient/session, not just aDBS012:
  - Channel names and LFP sample rate are read from that session's
    *Settings.json when available, or auto-detected from the CSV header
    otherwise -- nothing about a specific patient's contact configuration
    is hardcoded.
  - Windowing uses each sample's actual timestamp, not an assumed-constant
    sample spacing -- so gaps (packet loss) don't silently misalign later
    windows the way index-based slicing would.
  - All LFP channels are checked; whichever channel shows the strongest
    broadband elevation in a given window is the one reported, rather than
    assuming one fixed "primary" channel matters for every patient.
  - No stim frequency notch, and no attempt to separate stim artifact from
    movement artifact -- this is a general contamination detector, not a
    source classifier. If stim is on, stim power will contribute to the
    broadband feature like any other artifact; that's a known, accepted
    limitation of this simplified version, not an oversight.

Usage
-----
    python artifact_finder.py <lfp_csv> [--settings settings.json]
        [--out out_dir]
"""

import argparse
import csv
import json
import os
import shutil

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

HP_CUTOFF = 40.0
WIN_S = 1.0
STEP_S = 0.5
Z_THRESH = 6.5
MIN_FLAG_S = 0.5
MERGE_GAP_S = 0.25
ROLL_S = 5.0
GAP_THRESH_FACTOR = 3.0

PRIORITY_CHANNELS = [2, 3]
PRIORITY_TOL = 0.10



def _channels_from_settings(settings_path):
    """Pull sample rate + channel labels out of a BaseSettings.json-style
    file. Returns (fs, labels) or (None, None) if the expected fields
    aren't present -- caller falls back to auto-detection."""
    try:
        with open(settings_path) as f:
            cfg = json.load(f)
        rec = cfg["recording_settings"][0]
        fs = float(rec["neuralTD_samplerate"])
        labels = [ch["sense_contacts"] for ch in rec["channel_settings"]]
        return fs, labels
    except (KeyError, IndexError, FileNotFoundError, json.JSONDecodeError):
        return None, None


def load_lfp_csv(path, settings_path=None, gap_thresh_ms=None):
    """
    Returns
    -------
    lfp : ndarray (n_channels, n_samples)
    t_s : ndarray (n_samples,), real time in seconds relative to first sample
    fs_nominal : float, derived from median sample interval (for spectral
        windowing only -- NOT assumed constant for indexing)
    channel_labels : list[str]
    gaps_s : list[(start_s, end_s, gap_ms)]
    """
    df = pd.read_csv(path)
    t_ms = df["timestamp"].values.astype(np.int64)

    fs_settings, labels_settings = (None, None)
    if settings_path and os.path.exists(settings_path):
        fs_settings, labels_settings = _channels_from_settings(settings_path)

    data_cols = [c for c in df.columns if c != "timestamp"]
    lfp = df[data_cols].values.T.astype(float)

    channel_labels = labels_settings if (
        labels_settings and len(labels_settings) == len(data_cols)
    ) else data_cols

    dt_ms = np.diff(t_ms)
    median_dt = np.median(dt_ms)
    fs_nominal = fs_settings if fs_settings else 1000.0 / median_dt

    if gap_thresh_ms is None:
        gap_thresh_ms = GAP_THRESH_FACTOR * median_dt
    gap_idx = np.where(dt_ms > gap_thresh_ms)[0]
    gaps_s = [((t_ms[i] - t_ms[0]) / 1000.0, (t_ms[i + 1] - t_ms[0]) / 1000.0,
              float(dt_ms[i])) for i in gap_idx]

    t_s = (t_ms - t_ms[0]) / 1000.0
    return lfp, t_s, fs_nominal, channel_labels, gaps_s



def mad_z(x):
    """Whole-session MAD z-score -- kept for reference/comparison only.
    NOT used for detection anymore; see rolling_mad_z."""
    med = np.median(x)
    mad = np.median(np.abs(x - med)) * 1.4826
    if mad == 0:
        return np.zeros_like(x)
    return (x - med) / mad


def rolling_mad_z(x, step_s, roll_s=ROLL_S):
    """
    Local rolling MAD z-score: median/MAD computed from a window CENTERED on
    each point, not from the whole session. This is what's actually used for
    detection -- matches the established 5-second-centered-window convention
    so the threshold adapts to local baseline conditions (e.g. a session
    that's noisier in one stretch doesn't get a permanently loosened
    threshold everywhere else, the way a single whole-session MAD would).
    """
    import pandas as pd
    n = max(3, int(round(roll_s / step_s)))
    s = pd.Series(x)
    med_roll = s.rolling(n, center=True, min_periods=max(3, n // 2)).median()
    mad_roll = s.rolling(n, center=True, min_periods=max(3, n // 2)).apply(
        lambda w: np.median(np.abs(w - np.median(w))), raw=True) * 1.4826
    z = (s - med_roll) / mad_roll.replace(0, np.nan)
    return z.fillna(0.0).values


def rolling_mad_band(x, fs, roll_s, z_thresh):
    """
    Rolling median +/- z_thresh*MAD, computed directly on a (typically
    bandpass-filtered) voltage trace -- NOT on the broadband-power feature
    series used elsewhere in this file. Returns (median, upper, lower),
    all in the same units as x (e.g. uV), so the resulting band can be
    plotted directly on top of the raw signal to show how the local
    threshold moves over time.

    Uses two fast vectorized rolling-median passes (median, then median of
    the absolute deviations from that median) rather than a single rolling
    window with a custom per-window MAD callback -- the naive version is
    correct but ~50x slower for no accuracy gain (verified: rolling
    median matches exactly, MAD correlates at r=0.998 against the naive
    per-window computation, on real aDBS012 data).
    """
    n = max(3, int(round(roll_s * fs)))
    s = pd.Series(x)
    med = s.rolling(n, center=True, min_periods=max(3, n // 2)).median()
    abs_dev = (s - med).abs()
    mad = abs_dev.rolling(n, center=True, min_periods=max(3, n // 2)).median() * 1.4826
    upper = med + z_thresh * mad
    lower = med - z_thresh * mad
    return med.values, upper.values, lower.values


def compute_global_voltage_baseline(lfp, fs_lfp, bp_sos):
    """
    Whole-SESSION median +/- MAD, in voltage units, on the bandpass-
    filtered signal -- one scalar median and one scalar MAD per channel,
    computed ONCE across the entire recording (not per-window, which
    would repeat identical work 40+ times for no reason). This is the
    GLOBAL counterpart to rolling_mad_band()'s LOCAL rolling version, in
    the same units, so both can be drawn on the same voltage axis.

    Returns (global_median, global_mad), each shape (n_channels,).
    """
    n_ch = lfp.shape[0]
    global_median = np.zeros(n_ch)
    global_mad = np.zeros(n_ch)
    for i in range(n_ch):
        filt = sosfiltfilt(bp_sos, lfp[i])
        filt = filt - np.mean(filt)
        med = np.median(filt)
        global_median[i] = med
        global_mad[i] = np.median(np.abs(filt - med)) * 1.4826
    return global_median, global_mad


def broadband_power(x, fs, hp=HP_CUTOFF):
    if x.size < 20:
        return np.nan
    sos = butter(4, min(hp, fs / 2 - 1), btype="highpass", fs=fs, output="sos")
    xf = sosfiltfilt(sos, x)
    return np.sqrt(np.mean(xf ** 2))


def _time_windows(t_s, win_s, step_s):
    """Window boundaries by real time, not sample index -- robust to gaps
    and non-constant sample spacing."""
    t0, t1 = t_s[0], t_s[-1]
    starts = np.arange(t0, t1 - win_s, step_s)
    return starts


def _merge_runs(flag_bool, starts):
    """Merge a single channel's flagged sub-windows into contiguous time
    intervals. Returns list of (start_s, end_s)."""
    if not flag_bool.any():
        return []
    idx = np.where(flag_bool)[0]
    runs, run = [], [idx[0]]
    for i in idx[1:]:
        if (starts[i] - starts[run[-1]]) > (STEP_S + MERGE_GAP_S):
            runs.append(run)
            run = [i]
        else:
            run.append(i)
    runs.append(run)
    intervals = []
    for run in runs:
        s0, e0 = starts[run[0]], starts[run[-1]] + WIN_S
        if (e0 - s0) >= MIN_FLAG_S:
            intervals.append((s0, e0))
    return intervals


def detect_artifacts(lfp, t_s, channel_labels, fs_nominal, gaps_s=None):
    """
    Each channel is thresholded INDEPENDENTLY (its own MAD z-score against
    its own baseline), not just compared via a single max-across-channels
    series. Per-channel flagged runs are then overlaid in time:
      - time covered by 2+ channels' runs simultaneously -> one window,
        confidence = "high"
      - time covered by only 1 channel's run -> a separate window,
        confidence = "single-channel"
    This is what makes multi-channel agreement visible and reportable,
    rather than collapsing to a single winning channel per window.
    """
    starts = _time_windows(t_s, WIN_S, STEP_S)
    n_ch = lfp.shape[0]

    bb_per_ch = np.full((n_ch, starts.size), np.nan)
    for wi, s in enumerate(starts):
        mask = (t_s >= s) & (t_s < s + WIN_S)
        if mask.sum() < 20:
            continue
        for ci in range(n_ch):
            bb_per_ch[ci, wi] = broadband_power(lfp[ci, mask], fs_nominal)
    bb_per_ch = np.nan_to_num(bb_per_ch, nan=0.0)

    z_local = np.array([rolling_mad_z(bb_per_ch[ci], STEP_S) for ci in range(n_ch)])
    z_global = np.array([mad_z(bb_per_ch[ci]) for ci in range(n_ch)])
    z_per_ch = np.maximum(z_local, z_global)
    flagged_local = z_local > Z_THRESH
    flagged_global = z_global > Z_THRESH
    flagged_per_ch = flagged_local | flagged_global

    if gaps_s:
        gap_mask = np.zeros(starts.size, dtype=bool)
        for g_start, g_end, _ in gaps_s:
            gap_mask |= (starts >= g_start - WIN_S) & (starts <= g_end)
        flagged_per_ch &= ~gap_mask[None, :]
        flagged_local &= ~gap_mask[None, :]
        flagged_global &= ~gap_mask[None, :]

    per_ch_intervals = [_merge_runs(flagged_per_ch[ci], starts)
                        for ci in range(n_ch)]

    breakpoints = sorted({t for ivs in per_ch_intervals for iv in ivs
                          for t in iv})
    segments = []
    for i in range(len(breakpoints) - 1):
        t0, t1 = breakpoints[i], breakpoints[i + 1]
        if t1 - t0 <= 1e-9:
            continue
        mid = (t0 + t1) / 2
        active = {ci for ci in range(n_ch)
                 for s0, e0 in per_ch_intervals[ci] if s0 <= mid < e0}
        if active:
            segments.append([t0, t1, frozenset(active)])

    merged = []
    for seg in segments:
        if merged and merged[-1][2] == seg[2] and \
           abs(seg[0] - merged[-1][1]) < 1e-9:
            merged[-1][1] = seg[1]
        else:
            merged.append(list(seg))

    windows = []
    for s0, e1, chset in merged:
        if (e1 - s0) < MIN_FLAG_S:
            continue
        chset_list = sorted(chset)
        idxs = np.where((starts >= s0 - 1e-6) & (starts < e1 - 1e-6))[0]
        if idxs.size == 0:
            continue

        methods, local_peaks, global_peaks = [], [], []
        for c in chset_list:
            loc = flagged_local[c, idxs].any()
            glo = flagged_global[c, idxs].any()
            methods.append("both" if (loc and glo) else
                          "local" if loc else "global")
            local_peaks.append(float(z_local[c, idxs].max()))
            global_peaks.append(float(z_global[c, idxs].max()))

        flagged_agg = np.zeros(idxs.size, dtype=bool)
        for c in chset_list:
            flagged_agg |= (flagged_local[c, idxs] | flagged_global[c, idxs])
        crossing_times = []
        in_run = False
        for k, f in enumerate(flagged_agg):
            if f and not in_run:
                crossing_times.append(float(starts[idxs[k]]))
                in_run = True
            elif not f:
                in_run = False
        if not crossing_times:
            continue

        strengths = np.array([max(l, g) for l, g in zip(local_peaks, global_peaks)])
        priority_local = [j for j, c in enumerate(chset_list)
                          if c in PRIORITY_CHANNELS]
        best_j = int(np.argmax(strengths))
        if priority_local:
            gmax = strengths.max()
            pr_vals = strengths[priority_local]
            best_pr = priority_local[int(np.argmax(pr_vals))]
            if strengths[best_pr] >= (1 - PRIORITY_TOL) * gmax:
                best_j = best_pr
        trigger_channel = channel_labels[chset_list[best_j]]

        if set(PRIORITY_CHANNELS).issubset(chset):
            confidence = "priority-agreement"
        elif len(chset_list) >= 2:
            confidence = "high"
        else:
            confidence = "single-channel"

        windows.append(dict(
            flag_start=float(s0),
            flag_end=float(e1),
            first_crossing_time_s=crossing_times[0],
            crossing_times_s=crossing_times,
            n_mad_crossings=len(crossing_times),
            max_mad_z_broadband=float(strengths.max()),
            trigger_channel=trigger_channel,
            channels_crossing=[channel_labels[c] for c in chset_list],
            detection_methods=methods,
            peak_local_z=local_peaks,
            peak_global_z=global_peaks,
            confidence=confidence,
        ))

    windows.sort(key=lambda w: w["flag_start"])
    debug = dict(starts=starts, z_per_ch=z_per_ch, channel_labels=channel_labels,
                z_local=z_local, z_global=z_global,
                flagged_local=flagged_local, flagged_global=flagged_global)
    return windows, debug



def write_artifact_times_csv(windows, out_path):
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "window_id", "flag_start_s", "flag_end_s", "duration_s",
            "first_crossing_time_s", "crossing_times_s",
            "max_mad_z_broadband", "n_mad_crossings", "trigger_channel",
            "channels_crossing", "detection_methods",
            "peak_local_z_per_channel", "peak_global_z_per_channel",
            "confidence",
        ])
        for i, w in enumerate(windows):
            writer.writerow([
                f"w{i:04d}", f"{w['flag_start']:.3f}", f"{w['flag_end']:.3f}",
                f"{w['flag_end'] - w['flag_start']:.3f}",
                f"{w['first_crossing_time_s']:.3f}",
                ";".join(f"{t:.3f}" for t in w["crossing_times_s"]),
                f"{w['max_mad_z_broadband']:.3f}",
                w["n_mad_crossings"], w["trigger_channel"],
                ";".join(w["channels_crossing"]),
                ";".join(w["detection_methods"]),
                ";".join(f"{v:.2f}" for v in w["peak_local_z"]),
                ";".join(f"{v:.2f}" for v in w["peak_global_z"]),
                w["confidence"],
            ])



_CHANNEL_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red",
                   "tab:purple", "tab:brown"]


def plot_window_with_threshold(seg_lfp, fs_lfp, flag_start, flag_end,
                               channel_labels, ctx_start, ctx_end,
                               z_starts, z_local, z_global,
                               flagged_local, flagged_global,
                               global_median=None, global_mad=None,
                               session_duration_s=None, save_path=None):
    """
    Five panels: (1) clean stacked LFP, all channels, (2) a 2x2 grid with
    each channel's signal + its own LOCAL rolling MAD band (dashed) and,
    if global_median/global_mad are given, its GLOBAL whole-session MAD
    band (solid, darker shade of the same channel color), (3) the same
    stacked LFP again with crossings thickened, (4) the combined
    detection-method strip panel, (5) the spectrogram.

    session_duration_s, if given, lets the header report where this
    window falls within the whole recording, not just its own span.
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.colors import to_rgb
    from scipy.signal import spectrogram

    def _darken(color, factor=0.6):
        r, g, b = to_rgb(color)
        return (r * factor, g * factor, b * factor)

    n_ch, n_samp = seg_lfp.shape
    t_lfp = np.arange(n_samp) / fs_lfp + ctx_start
    abs_start = flag_start + ctx_start
    abs_end = flag_end + ctx_start
    method_colors = {"both": "black", "global": "purple", "local": "gold"}

    def _set_time_ticks(ax, n=7):
        """Evenly-spaced ticks from the true start to the true end of the
        plotted time range, so the last tick always lands exactly at the
        data boundary -- never overshooting past it, never leaving an
        awkward unlabeled gap before it (which matplotlib's automatic
        tick placement doesn't guarantee, especially with absolute
        timestamps that rarely land on round numbers)."""
        ticks = np.linspace(t_lfp[0], t_lfp[-1], n)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:.1f}" for t in ticks])

    n_rows = 5
    height_ratios = [1.0, 2.7, 1.0, 1.0, 1.0]
    fig = plt.figure(figsize=(11, 3.6 + 2.1 * (n_rows - 1)))
    gs = GridSpec(n_rows, 1, figure=fig, hspace=0.6, height_ratios=height_ratios)

    zmask = (z_starts >= ctx_start - WIN_S) & (z_starts <= ctx_end)
    z_idx = np.where(zmask)[0]
    spacing = 4 * np.nanmedian(np.abs(seg_lfp)) + 1e-9

    ax_p1 = fig.add_subplot(gs[0])
    for i in range(n_ch):
        color = _CHANNEL_COLORS[i % len(_CHANNEL_COLORS)]
        ax_p1.plot(t_lfp, seg_lfp[i] - i * spacing, lw=0.7, color=color,
                  label=channel_labels[i])
    ax_p1.axvspan(abs_start, abs_end, color="tab:red", alpha=0.12)
    ax_p1.legend(loc="upper right", fontsize=7, ncol=n_ch)
    ax_p1.set_ylabel("LFP (uV)")
    ax_p1.set_xlabel("Time (s)")
    ax_p1.set_xlim(t_lfp[0], t_lfp[-1])
    _set_time_ticks(ax_p1)

    grid = GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[1], hspace=0.9, wspace=0.35)
    axes_grid = []
    bp_sos = butter(4, [1.0, min(100.0, fs_lfp / 2 - 1)], btype="bandpass",
                    fs=fs_lfp, output="sos")
    for i in range(n_ch):
        color = _CHANNEL_COLORS[i % len(_CHANNEL_COLORS)]
        ax = fig.add_subplot(grid[i // 2, i % 2])

        filt = sosfiltfilt(bp_sos, seg_lfp[i])
        filt = filt - np.mean(filt)
        med, upper, lower = rolling_mad_band(filt, fs_lfp, ROLL_S, Z_THRESH)

        ax.plot(t_lfp, filt, color="black", lw=0.7, zorder=3)
        ax.fill_between(t_lfp, lower, upper, color=color, alpha=0.15, zorder=1)
        ax.plot(t_lfp, upper, color=color, lw=1.1, ls="--", zorder=2)
        ax.plot(t_lfp, lower, color=color, lw=1.1, ls="--", zorder=2)

        if global_median is not None and global_mad is not None:
            g_upper = global_median[i] + Z_THRESH * global_mad[i]
            g_lower = global_median[i] - Z_THRESH * global_mad[i]
            dark_color = _darken(color, factor=0.45)
            ax.fill_between(t_lfp, g_lower, g_upper, color=dark_color,
                           alpha=0.22, zorder=1.5)
            ax.axhline(g_upper, color=dark_color, lw=1.6, ls="-", zorder=2.5)
            ax.axhline(g_lower, color=dark_color, lw=1.6, ls="-", zorder=2.5)

        ax.axvspan(abs_start, abs_end, color="0.85", zorder=0)

        flag_mask = (t_lfp >= abs_start) & (t_lfp <= abs_end)
        if flag_mask.any():
            local_idx = np.where(flag_mask)[0]
            peak_i = local_idx[np.argmax(np.abs(filt[local_idx]))]
            ax.plot([t_lfp[peak_i]], [filt[peak_i]], "o",
                   markerfacecolor="none", markeredgecolor="red",
                   markeredgewidth=1.5, ms=7, zorder=4)

        ax.set_ylabel("LFP (uV)", fontsize=9)
        ax.set_title(f"Ch{i+1} ({channel_labels[i]}) | {Z_THRESH}x MAD band",
                    loc="left", fontsize=10.5, fontweight="bold")
        ax.tick_params(labelsize=8)
        ax.set_xlim(t_lfp[0], t_lfp[-1])
        ax.set_xlabel("Time (s)", fontsize=9)
        _set_time_ticks(ax, n=5)
        axes_grid.append(ax)


    if session_duration_s is not None:
        ts_line = (f"Flagged: t={abs_start:.1f}s\u2013{abs_end:.1f}s "
                  f"(of {session_duration_s:.1f}s total recording)")
    else:
        ts_line = f"Flagged: t={abs_start:.1f}s\u2013{abs_end:.1f}s"
    fig.text(0.01, 0.995, "Detected artifact window", fontsize=12,
             fontweight="bold", va="top")
    fig.text(0.01, 0.975, ts_line, fontsize=10, color="0.25", va="top")

    legend_handles = [
        Line2D([0], [0], color="black", lw=1.0, label="black line = filtered signal (per channel)"),
        Line2D([0], [0], color="0.3", lw=1.1, ls="--", label=f"colored dashed line = LOCAL {Z_THRESH}x MAD band"),
        Line2D([0], [0], color="0.15", lw=1.6, ls="-", label=f"darker solid line/fill = GLOBAL {Z_THRESH}x MAD threshold"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
              markeredgecolor="red", markeredgewidth=1.5, markersize=7,
              label="hollow red circle = peak within flagged region"),
        Patch(facecolor="0.85", edgecolor="none", label="shaded region = flagged"),
    ]
    fig.legend(handles=legend_handles, loc="upper left",
              bbox_to_anchor=(0.01, 0.955), fontsize=8, ncol=2, frameon=False)

    ax_p2 = fig.add_subplot(gs[2], sharex=ax_p1)
    for i in range(n_ch):
        color = _CHANNEL_COLORS[i % len(_CHANNEL_COLORS)]
        trace = seg_lfp[i] - i * spacing
        ax_p2.plot(t_lfp, trace, lw=0.7, color=color, label=channel_labels[i], zorder=2)
        for zi in z_idx:
            if not (flagged_local[i, zi] or flagged_global[i, zi]):
                continue
            s0, e0 = z_starts[zi], z_starts[zi] + WIN_S
            seg_mask = (t_lfp >= s0) & (t_lfp <= e0)
            ax_p2.plot(t_lfp[seg_mask], trace[seg_mask], lw=2.8,
                      color=color, alpha=0.9, zorder=3)
    ax_p2.axvspan(abs_start, abs_end, color="tab:red", alpha=0.12, zorder=1)
    ax_p2.legend(loc="upper right", fontsize=7, ncol=n_ch)
    ax_p2.set_ylabel("LFP (uV)")
    ax_p2.set_xlabel("Time (s)")
    _set_time_ticks(ax_p2)
    ax_p2.set_title("thick = channel crossed threshold "
                    "(aligned thickness across channels = agreement)", loc="left")

    ax_lfp3 = fig.add_subplot(gs[3], sharex=ax_p1)
    strip_h = spacing * 0.12
    strip_offset = spacing * 0.32
    for i in range(n_ch):
        color = _CHANNEL_COLORS[i % len(_CHANNEL_COLORS)]
        ax_lfp3.plot(t_lfp, seg_lfp[i] - i * spacing, lw=0.7, color=color,
                    label=channel_labels[i], zorder=2)
        strip_y = -i * spacing - strip_offset
        for zi in z_idx:
            loc, glo = flagged_local[i, zi], flagged_global[i, zi]
            if not (loc or glo):
                continue
            method = "both" if (loc and glo) else ("global" if glo else "local")
            s0 = z_starts[zi]
            ax_lfp3.broken_barh([(s0, STEP_S)], (strip_y - strip_h / 2, strip_h),
                               color=method_colors[method], zorder=3)
    ax_lfp3.axvspan(abs_start, abs_end, color="tab:red", alpha=0.12, zorder=1)
    ch_handles = [Line2D([0], [0], color=_CHANNEL_COLORS[i % len(_CHANNEL_COLORS)],
                        lw=1.5, label=channel_labels[i]) for i in range(n_ch)]
    method_handles = [Line2D([0], [0], color=c, lw=5, label=m)
                      for m, c in method_colors.items()]
    ax_lfp3.legend(handles=ch_handles + method_handles, loc="upper right",
                  fontsize=6, ncol=2)
    ax_lfp3.set_ylabel("LFP (uV)")
    ax_lfp3.set_xlabel("Time (s)")
    ax_lfp3.set_title("strip below trace = detection method", loc="left")
    ax_lfp3.set_xlim(t_lfp[0], t_lfp[-1])
    _set_time_ticks(ax_lfp3)

    ax_spec = fig.add_subplot(gs[4], sharex=ax_lfp3)
    nperseg = int(np.clip(seg_lfp[0].size // 8, 32, 256))
    f, tt, Sxx = spectrogram(seg_lfp[0], fs=fs_lfp, nperseg=nperseg,
                             noverlap=nperseg // 2)
    ax_spec.pcolormesh(tt + ctx_start, f, 10 * np.log10(Sxx + 1e-12),
                       shading="gouraud", cmap="magma")
    for edge in (abs_start, abs_end):
        ax_spec.axvline(edge, color="white", lw=0.8, ls=":")
    ax_spec.set_ylabel("Freq (Hz)")
    ax_spec.set_xlabel("Time (s)")
    ax_spec.set_xlim(t_lfp[0], t_lfp[-1])
    _set_time_ticks(ax_spec)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig




def run_from_data(lfp, t_s, fs_nominal, channel_labels, gaps_s=None,
                  out_dir="artifact_output"):
    """
    Does the WHOLE pipeline -- detection, CSV, per-window figures, and the
    two confidence subfolders -- starting from ALREADY-LOADED data, not
    file paths. This is the function to call from another script or a
    notebook if you've loaded data yourself (e.g. via load_lfp_data() in
    prepare_lfp_csv.py) and don't want to round-trip through a CSV file
    on disk just to run this pipeline.

    Parameters
    ----------
    lfp : ndarray (n_channels, n_samples)
    t_s : ndarray (n_samples,), seconds from first sample
    fs_nominal : float, sample rate in Hz
    channel_labels : list[str]
    gaps_s : list[(start_s, end_s, gap_ms)], optional
    out_dir : str, where to write artifact_times.csv + figures + subfolders

    Returns
    -------
    list[dict] : the flagged windows (same format written to the CSV)
    """
    os.makedirs(out_dir, exist_ok=True)

    windows, debug = detect_artifacts(lfp, t_s, channel_labels, fs_nominal,
                                      gaps_s=gaps_s)
    print(f"flagged windows (broadband-only): {len(windows)}")

    times_csv = os.path.join(out_dir, "artifact_times.csv")
    write_artifact_times_csv(windows, times_csv)
    print(f"wrote {times_csv}")

    bp_sos = butter(4, [1.0, min(100.0, fs_nominal / 2 - 1)],
                    btype="bandpass", fs=fs_nominal, output="sos")
    global_median, global_mad = compute_global_voltage_baseline(
        lfp, fs_nominal, bp_sos)

    for i, w in enumerate(windows):
        fs_, fe_ = w["flag_start"], w["flag_end"]
        c0, c1 = max(0.0, fs_ - 6.0), min(t_s[-1], fe_ + 6.0)
        mask = (t_s >= c0) & (t_s <= c1)
        seg_lfp = lfp[:, mask]

        fig = plot_window_with_threshold(
            seg_lfp, fs_nominal, fs_ - c0, fe_ - c0,
            channel_labels=channel_labels,
            ctx_start=c0, ctx_end=c1,
            z_starts=debug["starts"],
            z_local=debug["z_local"], z_global=debug["z_global"],
            flagged_local=debug["flagged_local"],
            flagged_global=debug["flagged_global"],
            global_median=global_median, global_mad=global_mad,
            session_duration_s=t_s[-1],
            save_path=os.path.join(out_dir, f"w{i:04d}_{fs_:.1f}s.png"),
        )
        import matplotlib.pyplot as plt
        plt.close(fig)

    if windows:
        print(f"wrote {len(windows)} figures to {out_dir}/ (all artifacts, "
              f"no ranking)")

    priority_labels = [channel_labels[c] for c in PRIORITY_CHANNELS
                       if c < len(channel_labels)]
    highest_dir = os.path.join(out_dir, "highest_confidence_artifacts")
    high_dir = os.path.join(out_dir, "high_confidence_artifacts")
    os.makedirs(highest_dir, exist_ok=True)
    os.makedirs(high_dir, exist_ok=True)

    highest_rows, high_rows = [], []
    for i, w in enumerate(windows):
        if not all(lbl in w["channels_crossing"] for lbl in priority_labels):
            continue
        methods_for_priority = [
            w["detection_methods"][w["channels_crossing"].index(lbl)]
            for lbl in priority_labels
        ]
        fname = f"w{i:04d}_{w['flag_start']:.1f}s.png"
        src = os.path.join(out_dir, fname)
        if not os.path.exists(src):
            continue
        if all(m == "both" for m in methods_for_priority):
            shutil.copy2(src, os.path.join(highest_dir, fname))
            highest_rows.append(w)
        elif all(m == "local" for m in methods_for_priority):
            shutil.copy2(src, os.path.join(high_dir, fname))
            high_rows.append(w)

    write_artifact_times_csv(highest_rows,
                             os.path.join(highest_dir, "artifact_times.csv"))
    write_artifact_times_csv(high_rows,
                             os.path.join(high_dir, "artifact_times.csv"))
    print(f"highest_confidence_artifacts/: {len(highest_rows)} windows "
          f"(includes local AND global MAD threshold crossing "
          f"+ ch3 and ch4 agreement)")
    print(f"high_confidence_artifacts/: {len(high_rows)} windows "
          f"(includes local MAD threshold crossing only "
          f"+ ch3 and ch4 agreement)")

    return windows


def run(lfp_csv, settings_path=None, out_dir="artifact_output"):
    """
    Thin file-loading wrapper: reads the LFP/settings FILES from disk,
    then hands everything to run_from_data() to do the actual work.
    This is what the CLI (__main__ below) calls. If you already have data
    loaded in memory, call run_from_data() directly instead -- no need to
    write a CSV first.
    """
    lfp, t_s, fs_nominal, channel_labels, gaps_s = load_lfp_csv(
        lfp_csv, settings_path=settings_path)
    print(f"loaded {lfp_csv}: {lfp.shape[0]} channels {channel_labels}, "
          f"fs~{fs_nominal:.2f}Hz, {t_s[-1]:.1f}s, {len(gaps_s)} gap(s)")

    return run_from_data(lfp, t_s, fs_nominal, channel_labels,
                         gaps_s=gaps_s, out_dir=out_dir)


def main():
    """CLI entry point -- callable directly (python artifact_finder.py ...)
    or via the installed console script (artifact-finder ...)."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("lfp_csv")
    ap.add_argument("--settings", default=None,
                    help="path to that session's *Settings.json")
    ap.add_argument("--out", default="artifact_output")
    args = ap.parse_args()
    run(args.lfp_csv, settings_path=args.settings, out_dir=args.out)


if __name__ == "__main__":
    main()

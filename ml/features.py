"""
Feature extraction from 5G monitoring time-series.

Input  : a DataFrame slice covering one time window (WINDOW_SIZE_S seconds),
         with columns [timestamp, metric, teid, slice, container, value].
Output : a flat dict of scalar features ready for the classifier.
"""

import numpy as np
import pandas as pd
from scipy import stats

from config import (
    THROUGHPUT_LOW_THRESHOLD_BPS,
    LATENCY_SPIKE_MULTIPLIER,
    CPU_HIGH_THRESHOLD,
    CPU_CONTAINERS,
    METRIC_THROUGHPUT,
    METRIC_LATENCY,
    METRIC_CPU,
    MIN_POINTS,
)


def _slope(values: np.ndarray) -> float:
    """Linear regression slope normalised by window length."""
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=float)
    slope, _, _, _, _ = stats.linregress(x, values)
    return float(slope)


def _safe_stats(arr: np.ndarray, prefix: str) -> dict:
    """Return mean/std/min/max/p95/p99/slope for an array."""
    if len(arr) < MIN_POINTS:
        return {f"{prefix}_{k}": 0.0 for k in ("mean", "std", "min", "max", "p95", "p99", "slope")}
    return {
        f"{prefix}_mean":  float(np.mean(arr)),
        f"{prefix}_std":   float(np.std(arr)),
        f"{prefix}_min":   float(np.min(arr)),
        f"{prefix}_max":   float(np.max(arr)),
        f"{prefix}_p95":   float(np.percentile(arr, 95)),
        f"{prefix}_p99":   float(np.percentile(arr, 99)),
        f"{prefix}_slope": _slope(arr),
    }


# ---------------------------------------------------------------------------
# Per-metric extractors
# ---------------------------------------------------------------------------

def _throughput_features(df: pd.DataFrame) -> dict:
    """
    Features derived from gtp_throughput_bitrate_bps.

    Aggregated stats + blast-radius indicators (how many TEIDs are affected).
    """
    feats = {}
    tput_df = df[df["metric"] == METRIC_THROUGHPUT]

    if tput_df.empty:
        return {k: 0.0 for k in _throughput_feature_names()}

    # Global (all TEIDs merged)
    all_values = tput_df["value"].values
    feats.update(_safe_stats(all_values, "tput_global"))
    feats["tput_below_threshold_frac"] = float(
        np.mean(all_values < THROUGHPUT_LOW_THRESHOLD_BPS)
    )

    # Per-TEID stats → then aggregate across TEIDs
    teid_means, teid_mins, teid_stds = [], [], []
    n_low_tput = 0

    for teid, grp in tput_df.groupby("teid"):
        v = grp["value"].values
        if len(v) < MIN_POINTS:
            continue
        m = float(np.mean(v))
        teid_means.append(m)
        teid_mins.append(float(np.min(v)))
        teid_stds.append(float(np.std(v)))
        if m < THROUGHPUT_LOW_THRESHOLD_BPS:
            n_low_tput += 1

    n_teids = max(len(teid_means), 1)
    feats["n_active_teids"]      = float(n_teids)
    feats["n_teids_low_tput"]    = float(n_low_tput)
    feats["blast_radius_tput"]   = float(n_low_tput) / n_teids  # 0=isolated, 1=all TEIDs

    feats["tput_variance_across_teids"] = float(np.var(teid_means)) if teid_means else 0.0
    feats["tput_min_across_teids"]      = float(np.min(teid_mins))  if teid_mins  else 0.0

    # Slice-level features
    slice_means = {}
    for sl, grp in tput_df.groupby("slice"):
        v = grp["value"].values
        if len(v) >= MIN_POINTS:
            slice_means[sl] = float(np.mean(v))

    slice_vals = list(slice_means.values())
    if len(slice_vals) >= 2:
        feats["slice_tput_ratio"]     = slice_vals[0] / max(slice_vals[1], 1.0)
        feats["slice_asymmetry"]      = float(np.std(slice_vals) / max(np.mean(slice_vals), 1.0))
        feats["n_slices_low_tput"]    = float(sum(v < THROUGHPUT_LOW_THRESHOLD_BPS for v in slice_vals))
    else:
        feats["slice_tput_ratio"]     = 1.0
        feats["slice_asymmetry"]      = 0.0
        feats["n_slices_low_tput"]    = float(sum(v < THROUGHPUT_LOW_THRESHOLD_BPS for v in slice_vals))

    return feats


def _latency_features(df: pd.DataFrame) -> dict:
    """
    Features derived from gtp_latency_us.
    """
    feats = {}
    lat_df = df[df["metric"] == METRIC_LATENCY]

    if lat_df.empty:
        return {k: 0.0 for k in _latency_feature_names()}

    all_values = lat_df["value"].values
    feats.update(_safe_stats(all_values, "lat_global"))

    window_mean = float(np.mean(all_values)) if len(all_values) else 1.0
    spike_thresh = window_mean * LATENCY_SPIKE_MULTIPLIER
    feats["lat_spike_count"]        = float(np.sum(all_values > spike_thresh))
    feats["lat_spike_frac"]         = float(np.mean(all_values > spike_thresh))

    # Per-TEID
    teid_means, teid_p95s = [], []
    n_high_lat = 0

    for teid, grp in lat_df.groupby("teid"):
        v = grp["value"].values
        if len(v) < MIN_POINTS:
            continue
        m = float(np.mean(v))
        teid_means.append(m)
        teid_p95s.append(float(np.percentile(v, 95)))
        if m > spike_thresh:
            n_high_lat += 1

    n_teids = max(len(teid_means), 1)
    feats["n_teids_high_latency"]    = float(n_high_lat)
    feats["blast_radius_latency"]    = float(n_high_lat) / n_teids
    feats["lat_variance_across_teids"] = float(np.var(teid_means)) if teid_means else 0.0

    return feats


def _cpu_features(df: pd.DataFrame) -> dict:
    """
    CPU usage rate features per component (UPF, gNB, AMF, SMF).
    cAdvisor provides cumulative counters; the DataFrame should already contain
    derived per-second rates (rate(container_cpu_usage_seconds_total[1s])).
    """
    feats = {}
    cpu_df = df[df["metric"] == METRIC_CPU]

    for component, substring in CPU_CONTAINERS.items():
        mask = cpu_df["container"].str.contains(substring, case=False, na=False)
        sub = cpu_df[mask]["value"].values
        if len(sub) >= MIN_POINTS:
            feats[f"cpu_{component}_mean"] = float(np.mean(sub))
            feats[f"cpu_{component}_max"]  = float(np.max(sub))
            feats[f"cpu_{component}_high_frac"] = float(np.mean(sub > CPU_HIGH_THRESHOLD))
        else:
            feats[f"cpu_{component}_mean"] = 0.0
            feats[f"cpu_{component}_max"]  = 0.0
            feats[f"cpu_{component}_high_frac"] = 0.0

    # Combined infra pressure score (mean of all component maxes)
    max_vals = [feats[f"cpu_{c}_max"] for c in CPU_CONTAINERS]
    feats["cpu_infra_pressure"] = float(np.mean(max_vals))

    return feats


def _derived_features(tput_feats: dict, lat_feats: dict, cpu_feats: dict) -> dict:
    """
    High-level derived features that combine multiple signals.

    These encode causal hypotheses directly as features:
    - infra_vs_radio: high CPU + degraded perf → infra cause; low CPU → radio/rf cause
    - uniform_degradation: all TEIDs affected equally → core/backhaul cause
    - isolated_degradation: only 1-2 TEIDs → RAN/radio cause
    """
    feats = {}

    tput_drop   = tput_feats.get("tput_below_threshold_frac", 0.0)
    lat_spike   = lat_feats.get("lat_spike_frac", 0.0)
    cpu_pressure = cpu_feats.get("cpu_infra_pressure", 0.0)
    blast_tput  = tput_feats.get("blast_radius_tput", 0.0)
    blast_lat   = lat_feats.get("blast_radius_latency", 0.0)

    degradation_score = max(tput_drop, lat_spike)

    # High score → infrastructure (CPU) is likely the cause
    feats["infra_cause_score"] = float(cpu_pressure * degradation_score)

    # High score → degradation is isolated to few TEIDs (radio/slice cause)
    feats["isolated_cause_score"] = float(degradation_score * (1.0 - max(blast_tput, blast_lat)))

    # High score → all TEIDs degraded uniformly (core/backhaul cause)
    feats["uniform_cause_score"] = float(degradation_score * max(blast_tput, blast_lat))

    # Slice asymmetry with degradation suggests slice contention
    slice_asym = tput_feats.get("slice_asymmetry", 0.0)
    feats["slice_contention_score"] = float(slice_asym * tput_drop)

    return feats


# ---------------------------------------------------------------------------
# Placeholder name lists (used for zero-filling when data is absent)
# ---------------------------------------------------------------------------

def _throughput_feature_names():
    base = [f"tput_global_{k}" for k in ("mean", "std", "min", "max", "p95", "p99", "slope")]
    return base + [
        "tput_below_threshold_frac",
        "n_active_teids", "n_teids_low_tput", "blast_radius_tput",
        "tput_variance_across_teids", "tput_min_across_teids",
        "slice_tput_ratio", "slice_asymmetry", "n_slices_low_tput",
    ]


def _latency_feature_names():
    base = [f"lat_global_{k}" for k in ("mean", "std", "min", "max", "p95", "p99", "slope")]
    return base + [
        "lat_spike_count", "lat_spike_frac",
        "n_teids_high_latency", "blast_radius_latency", "lat_variance_across_teids",
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract(window_df: pd.DataFrame) -> dict:
    """
    Extract all features from a single time window DataFrame.

    Expected columns:
        timestamp  : unix epoch (float)
        metric     : metric name string
        teid       : TEID hex string (may be empty for CPU metrics)
        slice      : slice label (may be empty)
        container  : container name (for CPU metrics)
        value      : float

    Returns a flat dict of scalar features.
    """
    tput_feats = _throughput_features(window_df)
    lat_feats  = _latency_features(window_df)
    cpu_feats  = _cpu_features(window_df)
    deriv_feats = _derived_features(tput_feats, lat_feats, cpu_feats)

    return {**tput_feats, **lat_feats, **cpu_feats, **deriv_feats}

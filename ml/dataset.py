"""
Build a labeled feature dataset from one or more 5g_ansible run directories.

Expected run directory layout (from develop-latency-paper playbook output):
  <run_dir>/
    metrics.csv       ← Prometheus time-series (long format)
    timeline.jsonl    ← scenario start/stop events
    qhat*.tgz         ← iperf logs (parsed optionally)

Usage:
    python dataset.py --runs results/tcp-paper-* --output dataset.parquet
"""

import argparse
import json
import os
import tarfile
import io

import numpy as np
import pandas as pd

import features as feat_module
from config import (
    SCENARIO_TO_CLASS,
    WINDOW_SIZE_S,
    SLIDE_STEP_S,
    MIN_POINTS,
)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_metrics(metrics_csv: str) -> pd.DataFrame:
    """
    Load Prometheus metrics CSV.

    Supports two common export formats:
      Long  : timestamp, metric, teid, slice, container, value
      Wide  : timestamp, <metric{labels}>, ...

    Always returns long format with columns:
        timestamp, metric, teid, slice, container, value
    """
    df = pd.read_csv(metrics_csv)
    df.columns = [c.strip() for c in df.columns]

    if "metric" in df.columns and "value" in df.columns:
        # already long format
        for col in ("teid", "slice", "container"):
            if col not in df.columns:
                df[col] = ""
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        return df.dropna(subset=["timestamp", "value"])

    # Wide format: melt into long
    id_cols = ["timestamp"]
    value_cols = [c for c in df.columns if c != "timestamp"]
    melted = df.melt(id_vars=id_cols, value_vars=value_cols, var_name="raw_metric", value_name="value")
    melted["value"] = pd.to_numeric(melted["value"], errors="coerce")
    melted["timestamp"] = pd.to_numeric(melted["timestamp"], errors="coerce")
    melted = melted.dropna(subset=["timestamp", "value"])

    # Parse metric name and labels from column name like: metricname{key="val",...}
    def _parse_label(raw, key):
        import re
        m = re.search(rf'{key}="([^"]*)"', raw)
        return m.group(1) if m else ""

    melted["metric"]    = melted["raw_metric"].str.split("{").str[0]
    melted["teid"]      = melted["raw_metric"].apply(lambda r: _parse_label(r, "teid"))
    melted["slice"]     = melted["raw_metric"].apply(lambda r: _parse_label(r, "slice"))
    melted["container"] = melted["raw_metric"].apply(lambda r: _parse_label(r, "container"))
    return melted.drop(columns=["raw_metric"])


def load_timeline(timeline_jsonl: str) -> list[dict]:
    """Load scenario timeline events from JSONL file."""
    events = []
    with open(timeline_jsonl) as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def _extract_scenario_windows(events: list[dict]) -> list[dict]:
    """
    Parse timeline events and return list of scenario windows:
    [{"scenario": str, "start_ts": float, "end_ts": float}, ...]

    Handles both explicit end events and infers end from next start.
    """
    windows = []
    starts = {}

    for ev in sorted(events, key=lambda e: e.get("ts", e.get("timestamp", 0))):
        ts = float(ev.get("ts", ev.get("timestamp", 0)))
        event_type = ev.get("event", "")
        scenario = ev.get("scenario", ev.get("name", ""))

        if "start" in event_type.lower():
            starts[scenario] = ts
        elif "end" in event_type.lower() or "stop" in event_type.lower():
            if scenario in starts:
                windows.append({
                    "scenario": scenario,
                    "start_ts": starts.pop(scenario),
                    "end_ts": ts,
                })

    # Close any unclosed windows using last known timestamp
    for scenario, start_ts in starts.items():
        windows.append({
            "scenario": scenario,
            "start_ts": start_ts,
            "end_ts":   start_ts + 300.0,  # default 300s if no end event
        })

    return windows


# ---------------------------------------------------------------------------
# Iperf feature extraction (optional enrichment)
# ---------------------------------------------------------------------------

def _load_iperf_features_from_tgz(run_dir: str) -> dict:
    """
    Extract aggregate iperf features from qhat*.tgz files in the run directory.
    Returns a dict of scalar features (averaged across all UEs).
    """
    dl_bitrates, ul_bitrates, retransmits = [], [], []

    for fname in os.listdir(run_dir):
        if not (fname.startswith("qhat") and fname.endswith(".tgz")):
            continue
        tgz_path = os.path.join(run_dir, fname)
        try:
            with tarfile.open(tgz_path) as tar:
                for member in tar.getmembers():
                    if not member.name.endswith(".json"):
                        continue
                    f = tar.extractfile(member)
                    if f is None:
                        continue
                    data = json.load(io.TextIOWrapper(f))
                    end = data.get("end", {})

                    # downlink
                    recv = end.get("sum_received", {})
                    if recv.get("bits_per_second"):
                        dl_bitrates.append(float(recv["bits_per_second"]))

                    # uplink
                    sent = end.get("sum_sent", {})
                    if sent.get("bits_per_second"):
                        ul_bitrates.append(float(sent["bits_per_second"]))
                    if sent.get("retransmits"):
                        retransmits.append(float(sent["retransmits"]))
        except Exception:
            continue

    def _agg(vals):
        if not vals:
            return 0.0, 0.0
        return float(np.mean(vals)), float(np.std(vals))

    dl_mean, dl_std = _agg(dl_bitrates)
    ul_mean, ul_std = _agg(ul_bitrates)
    ret_mean, _     = _agg(retransmits)

    return {
        "iperf_dl_bitrate_mean": dl_mean,
        "iperf_dl_bitrate_std":  dl_std,
        "iperf_ul_bitrate_mean": ul_mean,
        "iperf_ul_bitrate_std":  ul_std,
        "iperf_retransmits_mean": ret_mean,
        "iperf_dl_ul_ratio": dl_mean / max(ul_mean, 1.0),
    }


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_from_run(run_dir: str, use_iperf: bool = True) -> list[dict]:
    """
    Process a single run directory and return a list of labeled feature dicts.

    Each dict = one feature vector (one time window) with a "label" key.
    Windows are extracted from the scenario timeline and slid every SLIDE_STEP_S.
    """
    metrics_path  = os.path.join(run_dir, "metrics.csv")
    timeline_path = os.path.join(run_dir, "timeline.jsonl")

    if not os.path.exists(metrics_path):
        print(f"[SKIP] no metrics.csv in {run_dir}")
        return []
    if not os.path.exists(timeline_path):
        print(f"[SKIP] no timeline.jsonl in {run_dir}")
        return []

    metrics_df = load_metrics(metrics_path)
    events     = load_timeline(timeline_path)
    windows    = _extract_scenario_windows(events)

    iperf_feats = _load_iperf_features_from_tgz(run_dir) if use_iperf else {}

    rows = []
    for win in windows:
        scenario = win["scenario"]
        label = SCENARIO_TO_CLASS.get(scenario)
        if label is None:
            print(f"[WARN] unknown scenario '{scenario}', skipping")
            continue

        # Slide sub-windows over the scenario window
        t_start = win["start_ts"]
        t_end   = win["end_ts"]
        t       = t_start

        while t + WINDOW_SIZE_S <= t_end:
            sub = metrics_df[
                (metrics_df["timestamp"] >= t) &
                (metrics_df["timestamp"] <  t + WINDOW_SIZE_S)
            ]

            if len(sub) < MIN_POINTS:
                t += SLIDE_STEP_S
                continue

            feats = feat_module.extract(sub)
            feats.update(iperf_feats)
            feats["label"]    = label
            feats["scenario"] = scenario
            feats["t_start"]  = t
            rows.append(feats)

            t += SLIDE_STEP_S

    print(f"[OK] {run_dir}: {len(rows)} windows from {len(windows)} scenarios")
    return rows


def build_dataset(run_dirs: list[str], use_iperf: bool = True) -> pd.DataFrame:
    """Build a combined dataset from multiple run directories."""
    all_rows = []
    for rd in run_dirs:
        all_rows.extend(build_from_run(rd, use_iperf=use_iperf))

    if not all_rows:
        raise ValueError("No data found in any run directory.")

    df = pd.DataFrame(all_rows)
    df = df.fillna(0.0)
    print(f"\nDataset: {len(df)} windows, {df['label'].nunique()} classes")
    print(df["label"].value_counts().to_string())
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build labeled 5G fault dataset")
    parser.add_argument("--runs", nargs="+", required=True, help="Run directories (glob OK)")
    parser.add_argument("--output", default="ml/dataset.parquet", help="Output Parquet path")
    parser.add_argument("--no-iperf", action="store_true", help="Skip iperf log parsing")
    args = parser.parse_args()

    import glob
    dirs = []
    for pattern in args.runs:
        dirs.extend(glob.glob(pattern))
    dirs = sorted(set(dirs))

    if not dirs:
        print("No run directories found.")
        return

    df = build_dataset(dirs, use_iperf=not args.no_iperf)
    df.to_parquet(args.output, index=False)
    print(f"\nSaved to {args.output} ({len(df)} rows × {len(df.columns)} cols)")


if __name__ == "__main__":
    main()

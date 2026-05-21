"""
Real-time 5G fault inference service.

Queries live Prometheus every INFERENCE_INTERVAL seconds, extracts features
from the last WINDOW_SIZE_S seconds, and returns a classification with
per-class probabilities and top SHAP explanations.

Usage:
    python inference.py                          # starts Flask on :5000
    python inference.py --once                   # single prediction and exit
    python inference.py --prometheus http://...  # override Prometheus URL
"""

import argparse
import json
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import requests
import shap
from flask import Flask, jsonify

import features as feat_module
from config import (
    PROMETHEUS_URL,
    WINDOW_SIZE_S,
    INFERENCE_INTERVAL,
    METRIC_THROUGHPUT,
    METRIC_LATENCY,
    METRIC_CPU,
    MODEL_PATH,
    FEATURE_NAMES_PATH,
)

app = Flask(__name__)

_artifact   = None   # loaded lazily
_prometheus = PROMETHEUS_URL


# ---------------------------------------------------------------------------
# Prometheus query helpers
# ---------------------------------------------------------------------------

def _prom_range(metric_selector: str, duration_s: int, step: str = "1s") -> pd.DataFrame:
    """
    Query Prometheus range API and return a long-format DataFrame.
    Columns: timestamp, metric, teid, slice, container, value
    """
    end   = int(time.time())
    start = end - duration_s
    url   = f"{_prometheus}/api/v1/query_range"
    params = {
        "query": metric_selector,
        "start": start,
        "end":   end,
        "step":  step,
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[WARN] Prometheus query failed for '{metric_selector}': {e}")
        return pd.DataFrame()

    rows = []
    for series in data.get("data", {}).get("result", []):
        m  = series["metric"]
        for ts, val in series["values"]:
            rows.append({
                "timestamp": float(ts),
                "metric":    m.get("__name__", metric_selector.split("{")[0]),
                "teid":      m.get("teid", ""),
                "slice":     m.get("slice", ""),
                "container": m.get("container", m.get("pod", "")),
                "value":     float(val),
            })
    return pd.DataFrame(rows)


def fetch_window() -> pd.DataFrame:
    """Pull the last WINDOW_SIZE_S seconds of all relevant metrics."""
    frames = []

    frames.append(_prom_range(METRIC_THROUGHPUT, WINDOW_SIZE_S))
    frames.append(_prom_range(METRIC_LATENCY,    WINDOW_SIZE_S))

    # CPU: use rate() to convert cumulative counter to per-second rate
    cpu_query = f'rate({METRIC_CPU}[5s])'
    frames.append(_prom_range(cpu_query, WINDOW_SIZE_S, step="5s"))

    non_empty = [f for f in frames if not f.empty]
    if not non_empty:
        return pd.DataFrame()
    return pd.concat(non_empty, ignore_index=True)


# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------

def load_artifact():
    global _artifact
    if _artifact is not None:
        return _artifact

    artifact = joblib.load(MODEL_PATH)
    with open(FEATURE_NAMES_PATH) as f:
        feature_names = json.load(f)

    explainer = shap.TreeExplainer(artifact["model"])
    _artifact = {
        "model":         artifact["model"],
        "label_encoder": artifact["label_encoder"],
        "feature_names": feature_names,
        "explainer":     explainer,
    }
    return _artifact


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict(window_df: pd.DataFrame) -> dict:
    """
    Run classifier on a window DataFrame.
    Returns a dict with top predictions + SHAP explanations.
    """
    art = load_artifact()
    model   = art["model"]
    le      = art["label_encoder"]
    feat_names = art["feature_names"]
    explainer  = art["explainer"]

    raw_feats = feat_module.extract(window_df)

    # Align to trained feature order, fill missing with 0
    X_row = np.array([raw_feats.get(f, 0.0) for f in feat_names], dtype=float).reshape(1, -1)

    proba  = model.predict_proba(X_row)[0]
    top3_idx = np.argsort(proba)[::-1][:3]

    predictions = [
        {"class": le.classes_[i], "probability": round(float(proba[i]), 4)}
        for i in top3_idx
    ]

    # SHAP for the top predicted class
    top_class_idx = top3_idx[0]
    shap_vals = explainer.shap_values(X_row)
    if isinstance(shap_vals, list):
        sv = shap_vals[top_class_idx][0]
    else:
        sv = shap_vals[0]

    shap_df = pd.Series(sv, index=feat_names)
    top_shap = (
        shap_df.abs()
        .sort_values(ascending=False)
        .head(5)
    )
    explanations = [
        {
            "feature": feat,
            "shap_value": round(float(shap_df[feat]), 4),
            "feature_value": round(float(raw_feats.get(feat, 0.0)), 4),
        }
        for feat in top_shap.index
    ]

    return {
        "timestamp":    datetime.utcnow().isoformat() + "Z",
        "window_size_s": WINDOW_SIZE_S,
        "n_datapoints":  len(window_df),
        "top_cause":    predictions[0]["class"],
        "confidence":   predictions[0]["probability"],
        "predictions":  predictions,
        "explanations": explanations,
    }


# ---------------------------------------------------------------------------
# Flask endpoints
# ---------------------------------------------------------------------------

@app.route("/predict", methods=["GET"])
def predict_endpoint():
    window_df = fetch_window()
    if window_df.empty:
        return jsonify({"error": "No data from Prometheus"}), 503
    result = predict(window_df)
    return jsonify(result)


@app.route("/health", methods=["GET"])
def health():
    try:
        load_artifact()
        return jsonify({"status": "ok", "model": MODEL_PATH})
    except Exception as e:
        return jsonify({"status": "error", "detail": str(e)}), 500


@app.route("/classes", methods=["GET"])
def classes():
    art = load_artifact()
    return jsonify({"classes": list(art["label_encoder"].classes_)})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    global _prometheus

    parser = argparse.ArgumentParser(description="5G fault inference service")
    parser.add_argument("--prometheus", default=PROMETHEUS_URL, help="Prometheus base URL")
    parser.add_argument("--once", action="store_true", help="Single prediction then exit")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    _prometheus = args.prometheus

    if args.once:
        print("Fetching window from Prometheus...")
        window_df = fetch_window()
        if window_df.empty:
            print("No data returned from Prometheus.")
            return
        result = predict(window_df)
        print(json.dumps(result, indent=2))
        return

    print(f"Starting inference service on :{args.port}")
    print(f"Prometheus: {_prometheus}")
    print(f"Window: {WINDOW_SIZE_S}s  |  Model: {MODEL_PATH}")
    app.run(host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()

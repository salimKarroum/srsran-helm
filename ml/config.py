"""
Central configuration for the 5G fault classification pipeline.
Scenario classes map high-level causes to the scenario IDs from develop-latency-paper.
"""

# --- Scenario class definitions ---
# Maps each causal class to the scenario IDs that represent it.
SCENARIO_CLASSES = {
    "baseline": [
        "01_clean_near_baseline",
        "20_decomp_baseline_all_ues",
    ],
    "radio_degradation": [
        "02_near_vs_far_radio_condition",
        "12_physical_near_far_qhat02",
        "13_far_light_under_near_heavy_load",
        "21_decomp_far_ue_radio",
    ],
    "tcp_load": [
        "03_tcp_load_ramp",
        "05_far_ue_stress_with_near_load",
    ],
    "slice_contention": [
        "04_cross_slice_contention",
    ],
    "rf_interference": [
        "07_fit02_interference_near_ul_dl",
        "09_fit28_spatial_control_near_ul_dl",
        "10_fit02_bidir_interference_near_trio",
        "11_fit28_bidir_interference_near_trio",
    ],
    "upf_cpu_stress": [
        "22_decomp_upf_cpu_stress",
    ],
    "server_cpu_stress": [
        "23_decomp_iperf_server_cpu_stress",
    ],
    "mixed_traffic": [
        "06_mixed_ul_dl_near",
    ],
}

# Reverse map: scenario_id → class label
SCENARIO_TO_CLASS = {
    sid: cls
    for cls, sids in SCENARIO_CLASSES.items()
    for sid in sids
}

CLASS_NAMES = list(SCENARIO_CLASSES.keys())

# --- Prometheus ---
PROMETHEUS_URL = "http://localhost:30090"

# Metric names as scraped by the monitoring stack
METRIC_THROUGHPUT = "gtp_throughput_bitrate_bps"
METRIC_LATENCY    = "gtp_latency_us"          # microseconds; adjust if your probe uses a different name
METRIC_CPU        = "container_cpu_usage_seconds_total"

# cAdvisor container name substrings to identify components
CPU_CONTAINERS = {
    "upf":    "upf",
    "gnb":    "gnb",
    "amf":    "amf",
    "smf":    "smf",
}

# --- Feature extraction ---
WINDOW_SIZE_S    = 60    # seconds per feature window
SLIDE_STEP_S     = 30    # sliding window step for real-time inference
MIN_POINTS       = 20    # minimum data points required to compute features

# Thresholds for derived features
THROUGHPUT_LOW_THRESHOLD_BPS = 10_000_000   # 10 Mbps
LATENCY_SPIKE_MULTIPLIER     = 2.0          # spike = value > N * window_mean
CPU_HIGH_THRESHOLD           = 0.8          # fraction of 1 CPU core (rate over 1s)

# --- Model ---
MODEL_PATH = "ml/model.pkl"
FEATURE_NAMES_PATH = "ml/feature_names.json"

# XGBoost hyperparameters (defaults, override via train.py CLI)
XGB_PARAMS = {
    "n_estimators":     300,
    "max_depth":        6,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "use_label_encoder": False,
    "eval_metric":      "mlogloss",
    "random_state":     42,
}

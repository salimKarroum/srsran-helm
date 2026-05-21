"""
5G Fault Classifier — pipeline en 3 commandes

    python pipeline.py train   --data  /chemin/vers/runs/
    python pipeline.py predict --prom  http://<node>:30090
    python pipeline.py demo

Chaque "run" est un dossier contenant metrics.csv + timeline.jsonl.
"""

import argparse
import glob
import json
import os
import sys
import time

import joblib
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

MODEL_FILE = "ml/model.pkl"

# Scénario → classe de cause
SCENARIO_TO_CLASS = {
    "01_clean_near_baseline":              "normal",
    "20_decomp_baseline_all_ues":          "normal",
    "02_near_vs_far_radio_condition":      "degradation_radio",
    "12_physical_near_far_qhat02":         "degradation_radio",
    "21_decomp_far_ue_radio":             "degradation_radio",
    "03_tcp_load_ramp":                    "surcharge_trafic",
    "04_cross_slice_contention":           "contention_slice",
    "07_fit02_interference_near_ul_dl":    "interference_rf",
    "09_fit28_spatial_control_near_ul_dl": "interference_rf",
    "10_fit02_bidir_interference_near_trio":"interference_rf",
    "11_fit28_bidir_interference_near_trio":"interference_rf",
    "22_decomp_upf_cpu_stress":            "surcharge_upf",
    "23_decomp_iperf_server_cpu_stress":   "surcharge_serveur",
    "06_mixed_ul_dl_near":                 "trafic_mixte",
}


# ---------------------------------------------------------------------------
# 1. Extraction de features (8 métriques simples)
# ---------------------------------------------------------------------------

def extract_features(df: pd.DataFrame) -> dict:
    """
    Calcule 8 features depuis un DataFrame de métriques Prometheus.
    Colonnes attendues : metric, teid, slice, container, value
    """
    def mean_of(metric):
        v = df[df["metric"] == metric]["value"]
        return float(v.mean()) if len(v) else 0.0

    def std_of(metric):
        v = df[df["metric"] == metric]["value"]
        return float(v.std()) if len(v) else 0.0

    def min_of(metric):
        v = df[df["metric"] == metric]["value"]
        return float(v.min()) if len(v) else 0.0

    tput = df[df["metric"] == "gtp_throughput_bitrate_bps"]["value"]
    lat  = df[df["metric"] == "gtp_latency_us"]["value"]

    # Combien de tunnels (TEID) ont un débit < 10 Mbps
    n_teids = df[df["metric"] == "gtp_throughput_bitrate_bps"].groupby("teid")["value"].mean()
    n_low   = (n_teids < 10_000_000).sum()
    blast   = n_low / max(len(n_teids), 1)  # 0 = un seul, 1 = tous affectés

    # Asymétrie entre slices
    by_slice = df[df["metric"] == "gtp_throughput_bitrate_bps"].groupby("slice")["value"].mean()
    slice_asym = float(by_slice.std() / by_slice.mean()) if len(by_slice) >= 2 and by_slice.mean() > 0 else 0.0

    # CPU UPF et gNB (rate déjà calculé, ou 0 si absent)
    cpu_upf = df[df["container"].str.contains("upf", case=False, na=False)]["value"]
    cpu_gnb = df[df["container"].str.contains("gnb", case=False, na=False)]["value"]

    return {
        "tput_moyen_mbps":  float(tput.mean()) / 1e6 if len(tput) else 0.0,
        "tput_min_mbps":    float(tput.min())  / 1e6 if len(tput) else 0.0,
        "tput_instabilite": float(tput.std())  / 1e6 if len(tput) else 0.0,
        "latence_moy_ms":   float(lat.mean())  / 1e3 if len(lat)  else 0.0,
        "latence_p95_ms":   float(lat.quantile(0.95)) / 1e3 if len(lat) else 0.0,
        "blast_radius":     blast,
        "asym_slices":      slice_asym,
        "cpu_upf":          float(cpu_upf.mean()) if len(cpu_upf) else 0.0,
        "cpu_gnb":          float(cpu_gnb.mean()) if len(cpu_gnb) else 0.0,
    }


# ---------------------------------------------------------------------------
# 2. Chargement des données d'un run
# ---------------------------------------------------------------------------

def load_run(run_dir: str) -> list[dict]:
    """Lit metrics.csv + timeline.jsonl et retourne des lignes étiquetées."""
    metrics_path  = os.path.join(run_dir, "metrics.csv")
    timeline_path = os.path.join(run_dir, "timeline.jsonl")

    if not os.path.exists(metrics_path) or not os.path.exists(timeline_path):
        return []

    df = pd.read_csv(metrics_path)
    for col in ("teid", "slice", "container"):
        if col not in df.columns:
            df[col] = ""
    df["value"]     = pd.to_numeric(df["value"],     errors="coerce")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "value"])

    events = [json.loads(l) for l in open(timeline_path) if l.strip()]

    rows = []
    for ev in events:
        if "start" not in ev.get("event", "").lower():
            continue
        scenario = ev.get("scenario", ev.get("name", ""))
        label    = SCENARIO_TO_CLASS.get(scenario)
        if label is None:
            continue

        ts = float(ev.get("ts", ev.get("timestamp", 0)))
        window = df[(df["timestamp"] >= ts) & (df["timestamp"] < ts + 60)]
        if len(window) < 10:
            continue

        feats = extract_features(window)
        feats["label"] = label
        rows.append(feats)

    return rows


# ---------------------------------------------------------------------------
# 3. Commandes
# ---------------------------------------------------------------------------

def cmd_train(data_dir: str):
    """Lit tous les runs, entraîne RandomForest, sauvegarde le modèle."""
    print(f"Chargement des données depuis {data_dir} ...")

    all_rows = []
    for run_dir in glob.glob(os.path.join(data_dir, "*")):
        if os.path.isdir(run_dir):
            rows = load_run(run_dir)
            if rows:
                print(f"  {os.path.basename(run_dir)} : {len(rows)} fenêtres")
                all_rows.extend(rows)

    if not all_rows:
        print("Aucune donnée trouvée. Vérifie que les dossiers contiennent metrics.csv + timeline.jsonl")
        sys.exit(1)

    dataset = pd.DataFrame(all_rows).fillna(0.0)
    print(f"\nDataset : {len(dataset)} exemples")
    print(dataset["label"].value_counts().to_string())

    feature_cols = [c for c in dataset.columns if c != "label"]
    X = dataset[feature_cols]
    y = dataset["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\n--- Résultats ---")
    print(classification_report(y_test, y_pred))

    joblib.dump({"model": model, "features": feature_cols}, MODEL_FILE)
    print(f"Modèle sauvegardé dans {MODEL_FILE}")


def cmd_predict(prometheus_url: str):
    """Interroge Prometheus toutes les 30s et affiche la cause probable."""
    if not os.path.exists(MODEL_FILE):
        print(f"Modèle introuvable : {MODEL_FILE}. Lance d'abord : python pipeline.py train")
        sys.exit(1)

    artifact = joblib.load(MODEL_FILE)
    model    = artifact["model"]
    features = artifact["features"]

    print(f"Service démarré — interrogation Prometheus toutes les 30s\n")

    while True:
        end   = int(time.time())
        start = end - 60
        rows  = []

        for metric in ("gtp_throughput_bitrate_bps", "gtp_latency_us",
                        'rate(container_cpu_usage_seconds_total[5s])'):
            try:
                resp = requests.get(
                    f"{prometheus_url}/api/v1/query_range",
                    params={"query": metric, "start": start, "end": end, "step": "1s"},
                    timeout=5,
                )
                for series in resp.json().get("data", {}).get("result", []):
                    m = series["metric"]
                    for ts, val in series["values"]:
                        rows.append({
                            "timestamp": float(ts),
                            "metric":    m.get("__name__", metric.split("(")[-1].split("[")[0]),
                            "teid":      m.get("teid", ""),
                            "slice":     m.get("slice", ""),
                            "container": m.get("container", m.get("pod", "")),
                            "value":     float(val),
                        })
            except Exception:
                pass

        if rows:
            window_df = pd.DataFrame(rows)
            feats = extract_features(window_df)
            X     = pd.DataFrame([feats])[features]
            proba = model.predict_proba(X)[0]
            classes = model.classes_

            top3 = sorted(zip(classes, proba), key=lambda x: -x[1])[:3]
            print(f"[{time.strftime('%H:%M:%S')}] Cause probable :")
            for cls, p in top3:
                bar = "█" * int(p * 20)
                print(f"  {cls:<25} {bar} {p:.0%}")
            print()
        else:
            print(f"[{time.strftime('%H:%M:%S')}] Pas de données Prometheus")

        time.sleep(30)


def cmd_demo():
    """Génère des données fictives pour tester le pipeline sans vrai réseau."""
    import random
    import tempfile

    print("=== DEMO : génération de données fictives ===\n")

    scenarios = [
        ("normal",            {"tput": 45e6, "lat": 2000,  "cpu_upf": 0.1}),
        ("interference_rf",   {"tput": 12e6, "lat": 15000, "cpu_upf": 0.1}),
        ("surcharge_upf",     {"tput": 8e6,  "lat": 30000, "cpu_upf": 0.9}),
        ("degradation_radio", {"tput": 5e6,  "lat": 8000,  "cpu_upf": 0.1}),
    ]

    tmpdir = tempfile.mkdtemp()
    all_rows = []

    for label, params in scenarios:
        for _ in range(30):  # 30 fenêtres par classe
            tput = params["tput"] * random.uniform(0.8, 1.2)
            lat  = params["lat"]  * random.uniform(0.9, 1.3)
            cpu  = params["cpu_upf"] * random.uniform(0.8, 1.1)
            all_rows.append({
                "tput_moyen_mbps":  tput / 1e6,
                "tput_min_mbps":    tput * 0.6 / 1e6,
                "tput_instabilite": tput * 0.1 / 1e6,
                "latence_moy_ms":   lat / 1e3,
                "latence_p95_ms":   lat * 1.5 / 1e3,
                "blast_radius":     1.0 if label == "interference_rf" else 0.2,
                "asym_slices":      0.5 if label == "contention_slice" else 0.0,
                "cpu_upf":          cpu,
                "cpu_gnb":          0.2,
                "label":            label,
            })

    dataset = pd.DataFrame(all_rows)
    feature_cols = [c for c in dataset.columns if c != "label"]
    X_train, X_test, y_train, y_test = train_test_split(
        dataset[feature_cols], dataset["label"], test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(classification_report(y_test, y_pred))

    joblib.dump({"model": model, "features": feature_cols}, MODEL_FILE)
    print(f"Modèle demo sauvegardé dans {MODEL_FILE}")
    print("\nTu peux maintenant lancer :")
    print("  python pipeline.py predict --prom http://<ton-node>:30090")


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="5G Fault Classifier")
    sub = parser.add_subparsers(dest="cmd")

    p_train = sub.add_parser("train",   help="Entraîner depuis les runs")
    p_train.add_argument("--data", required=True, help="Dossier contenant les run directories")

    p_pred = sub.add_parser("predict",  help="Prédire en temps réel depuis Prometheus")
    p_pred.add_argument("--prom", required=True, help="URL Prometheus, ex: http://node:30090")

    sub.add_parser("demo", help="Tester avec des données fictives")

    args = parser.parse_args()

    if args.cmd == "train":
        cmd_train(args.data)
    elif args.cmd == "predict":
        cmd_predict(args.prom)
    elif args.cmd == "demo":
        cmd_demo()
    else:
        parser.print_help()

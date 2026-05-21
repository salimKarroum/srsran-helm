"""
Offline evaluation utilities.

Usage:
    # Evaluate a saved model against a held-out dataset
    python evaluate.py --model ml/model.pkl --dataset ml/dataset.parquet

    # Evaluate with a custom split ratio
    python evaluate.py --model ml/model.pkl --dataset ml/dataset.parquet --test-ratio 0.3
"""

import argparse
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from config import FEATURE_NAMES_PATH

NON_FEATURE_COLS = {"label", "scenario", "t_start"}


def load_data(dataset_path: str, feature_names_path: str, test_ratio: float):
    df = pd.read_parquet(dataset_path)
    with open(feature_names_path) as f:
        feat_names = json.load(f)

    X = df[[c for c in feat_names if c in df.columns]].astype(float).fillna(0.0)
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_ratio, stratify=y, random_state=42
    )
    return X_train, X_test, y_train, y_test, feat_names


def per_class_metrics(y_true, y_pred, classes) -> pd.DataFrame:
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    rows = []
    for cls in classes:
        m = report.get(cls, {})
        rows.append({
            "class":     cls,
            "precision": round(m.get("precision", 0), 3),
            "recall":    round(m.get("recall", 0), 3),
            "f1":        round(m.get("f1-score", 0), 3),
            "support":   int(m.get("support", 0)),
        })
    return pd.DataFrame(rows).set_index("class")


def plot_confusion_matrix(y_true, y_pred, classes, out_path: str):
    cm = confusion_matrix(y_true, y_pred, labels=classes, normalize="true")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt=".2f",
        xticklabels=classes, yticklabels=classes,
        cmap="Blues", ax=ax, vmin=0, vmax=1,
    )
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Normalised Confusion Matrix")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_shap_summary(model, X_test: pd.DataFrame, le, out_path: str):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test.values[:200])

    if isinstance(shap_values, list):
        # Multi-class: stack and take mean abs
        mean_abs = np.mean([np.abs(sv) for sv in shap_values], axis=0)
    else:
        mean_abs = np.abs(shap_values)

    importance = pd.Series(mean_abs.mean(axis=0), index=X_test.columns)
    importance = importance.sort_values(ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(10, 7))
    importance[::-1].plot(kind="barh", ax=ax, color="steelblue")
    ax.set_xlabel("Mean |SHAP value| across classes")
    ax.set_title("Top-20 Feature Importances (SHAP)")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_per_class_shap(model, X_test: pd.DataFrame, le, out_dir: str = "ml"):
    """One bar chart per class showing which features drive that class's score."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test.values[:200])

    if not isinstance(shap_values, list):
        return  # binary case, skip

    for i, cls in enumerate(le.classes_):
        sv = shap_values[i]
        importance = pd.Series(np.abs(sv).mean(axis=0), index=X_test.columns)
        importance = importance.sort_values(ascending=False).head(10)

        fig, ax = plt.subplots(figsize=(8, 5))
        importance[::-1].plot(kind="barh", ax=ax, color="coral")
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title(f"Top features for class: {cls}")
        plt.tight_layout()
        out_path = f"{out_dir}/shap_{cls}.png"
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f"Saved: {out_path}")


def evaluate(model_path: str, dataset_path: str, feature_names_path: str, test_ratio: float):
    artifact = joblib.load(model_path)
    model = artifact["model"]
    le    = artifact["label_encoder"]

    X_train, X_test, y_train, y_test, feat_names = load_data(
        dataset_path, feature_names_path, test_ratio
    )

    X_test_aligned = X_test[[c for c in feat_names if c in X_test.columns]]

    y_pred     = le.inverse_transform(model.predict(X_test_aligned.values))
    y_true     = y_test
    y_pred_idx = model.predict(X_test_aligned.values)
    y_true_idx = le.transform(y_true)

    print("\n=== Per-class metrics ===")
    metrics_df = per_class_metrics(y_true, y_pred, le.classes_)
    print(metrics_df.to_string())

    # Macro-averaged ROC-AUC
    try:
        proba = model.predict_proba(X_test_aligned.values)
        auc = roc_auc_score(y_true_idx, proba, multi_class="ovr", average="macro")
        print(f"\nMacro ROC-AUC: {auc:.4f}")
    except Exception:
        pass

    plot_confusion_matrix(y_true, y_pred, le.classes_, "ml/confusion_matrix_eval.png")
    plot_shap_summary(model, X_test_aligned, le, "ml/feature_importance_eval.png")
    plot_per_class_shap(model, X_test_aligned, le, out_dir="ml")


def main():
    parser = argparse.ArgumentParser(description="Evaluate 5G fault classifier")
    parser.add_argument("--model",         default="ml/model.pkl")
    parser.add_argument("--dataset",       default="ml/dataset.parquet")
    parser.add_argument("--feature-names", default=FEATURE_NAMES_PATH)
    parser.add_argument("--test-ratio",    type=float, default=0.2)
    args = parser.parse_args()

    evaluate(args.model, args.dataset, args.feature_names, args.test_ratio)


if __name__ == "__main__":
    main()

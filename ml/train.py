"""
Train the 5G fault classifier.

Usage:
    python train.py --dataset ml/dataset.parquet
    python train.py --dataset ml/dataset.parquet --output ml/model.pkl

Outputs:
    ml/model.pkl            XGBoost classifier
    ml/feature_names.json   Ordered feature list (needed by inference.py)
    ml/confusion_matrix.png Normalised confusion matrix figure
    ml/feature_importance.png Top-20 feature importance (SHAP)
"""

import argparse
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from config import CLASS_NAMES, MODEL_PATH, FEATURE_NAMES_PATH, XGB_PARAMS

NON_FEATURE_COLS = {"label", "scenario", "t_start"}


def load_xy(path: str):
    df = pd.read_parquet(path)
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X = df[feature_cols].astype(float).fillna(0.0)
    y_raw = df["label"].values
    return X, y_raw, feature_cols


def plot_confusion_matrix(y_true, y_pred, classes, out_path: str):
    cm = confusion_matrix(y_true, y_pred, labels=classes, normalize="true")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        xticklabels=classes,
        yticklabels=classes,
        cmap="Blues",
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Normalised Confusion Matrix — 5G Fault Classifier")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved to {out_path}")


def plot_shap_importance(model, X: pd.DataFrame, out_path: str, top_n: int = 20):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X.values[:500])  # sample for speed

    # Multi-class: shap_values is a list of arrays; take mean abs across classes
    if isinstance(shap_values, list):
        mean_abs = np.mean([np.abs(sv) for sv in shap_values], axis=0)
    else:
        mean_abs = np.abs(shap_values)

    importances = pd.Series(mean_abs.mean(axis=0), index=X.columns)
    importances = importances.sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    importances[::-1].plot(kind="barh", ax=ax, color="steelblue")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Top-{top_n} Features by SHAP Importance")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"SHAP feature importance saved to {out_path}")


def train(dataset_path: str, model_out: str, feature_names_out: str):
    print(f"Loading dataset from {dataset_path}...")
    X, y_raw, feature_cols = load_xy(dataset_path)

    le = LabelEncoder()
    le.fit(CLASS_NAMES)
    y = le.transform(y_raw)

    print(f"\nTraining XGBoost on {len(X)} samples, {len(feature_cols)} features, {len(le.classes_)} classes")

    model = XGBClassifier(
        **{k: v for k, v in XGB_PARAMS.items() if k != "random_state"},
        n_jobs=-1,
        seed=XGB_PARAMS["random_state"],
        num_class=len(le.classes_),
        objective="multi:softprob",
    )

    # Cross-validation evaluation
    print("\n--- 5-Fold Stratified Cross-Validation ---")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred_cv = cross_val_predict(model, X.values, y, cv=cv)
    y_pred_labels = le.inverse_transform(y_pred_cv)
    y_true_labels = le.inverse_transform(y)

    print(classification_report(y_true_labels, y_pred_labels, target_names=le.classes_))

    # Final model trained on full dataset
    print("\nTraining final model on full dataset...")
    model.fit(X.values, y)

    # Save model + label encoder together
    artifact = {"model": model, "label_encoder": le}
    joblib.dump(artifact, model_out)
    print(f"Model saved to {model_out}")

    # Save feature names for inference
    with open(feature_names_out, "w") as f:
        json.dump(feature_cols, f, indent=2)
    print(f"Feature names saved to {feature_names_out}")

    # Plots
    plot_confusion_matrix(y_true_labels, y_pred_labels, le.classes_, "ml/confusion_matrix.png")
    plot_shap_importance(model, X, "ml/feature_importance.png")

    return model, le, feature_cols


def main():
    parser = argparse.ArgumentParser(description="Train 5G fault classifier")
    parser.add_argument("--dataset", default="ml/dataset.parquet")
    parser.add_argument("--output", default=MODEL_PATH)
    parser.add_argument("--feature-names", default=FEATURE_NAMES_PATH)
    args = parser.parse_args()

    train(args.dataset, args.output, args.feature_names)


if __name__ == "__main__":
    main()

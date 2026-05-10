#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${NAMESPACE:-open5gs}"
CHARTS_DIR="$(cd "$(dirname "$0")/charts" && pwd)"

helm_deploy() {
  local release=$1
  local chart=$2
  shift 2
  if helm status "$release" -n "$NAMESPACE" &>/dev/null; then
    echo "[upgrade] $release"
    helm upgrade "$release" "$chart" -n "$NAMESPACE" "$@"
  else
    echo "[install] $release"
    helm install "$release" "$chart" -n "$NAMESPACE" "$@"
  fi
}

echo "==> Namespace: $NAMESPACE"
kubectl get namespace "$NAMESPACE" &>/dev/null || kubectl create namespace "$NAMESPACE"

helm_deploy srsran-gnb   "$CHARTS_DIR/srsran-gnb"      -f "$CHARTS_DIR/srsran-gnb/values-rfsim.yaml"
helm_deploy srsran-ue    "$CHARTS_DIR/srsue"
helm_deploy rl-xapp      "$CHARTS_DIR/rl-xapp"
helm_deploy telegraf     "$CHARTS_DIR/telegraf"

echo ""
echo "==> Waiting for pods to be ready..."
kubectl wait --for=condition=Ready pod -l app=srsran-gnb -n "$NAMESPACE" --timeout=120s || true
kubectl wait --for=condition=Ready pod -l app=srsran-ue  -n "$NAMESPACE" --timeout=120s || true

echo ""
echo "==> Pod status:"
kubectl get pods -n "$NAMESPACE"

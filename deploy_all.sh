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
    helm upgrade "$release" "$chart" -n "$NAMESPACE" --reset-values "$@"
  else
    echo "[install] $release"
    helm install "$release" "$chart" -n "$NAMESPACE" "$@"
  fi
}

# Print status for a pod selector; show events and tail logs if not Ready.
# Usage: check_pod_status <label-selector> <component-name> [timeout]
check_pod_status() {
  local selector=$1
  local name=$2
  local timeout=${3:-180}

  echo ""
  echo "--- $name ---"

  if ! kubectl wait --for=condition=Ready pod -l "$selector" -n "$NAMESPACE" \
       --timeout="${timeout}s" 2>/dev/null; then
    echo "[WARN] $name pod not Ready after ${timeout}s"

    local pod
    pod=$(kubectl get pod -l "$selector" -n "$NAMESPACE" \
          -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)

    if [[ -z "$pod" ]]; then
      echo "  No pod found for selector: $selector"
      return 1
    fi

    echo "  Pod: $pod"
    kubectl get pod "$pod" -n "$NAMESPACE" \
      -o custom-columns='STATUS:.status.phase,READY:.status.containerStatuses[*].ready,RESTARTS:.status.containerStatuses[*].restartCount' \
      2>/dev/null || true

    echo "  Recent events:"
    kubectl get events -n "$NAMESPACE" \
      --field-selector "involvedObject.name=$pod" \
      --sort-by='.lastTimestamp' 2>/dev/null | tail -5 || true

    echo "  Last 20 log lines:"
    kubectl logs "$pod" -n "$NAMESPACE" --tail=20 2>/dev/null || true

    return 1
  fi

  local pod
  pod=$(kubectl get pod -l "$selector" -n "$NAMESPACE" \
        -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
  echo "  [OK] $pod"
  return 0
}

echo "==> Namespace: $NAMESPACE"
kubectl get namespace "$NAMESPACE" &>/dev/null || kubectl create namespace "$NAMESPACE"

helm_deploy srsran-gnb   "$CHARTS_DIR/srsran-gnb"      -f "$CHARTS_DIR/srsran-gnb/values-rfsim.yaml"
helm_deploy srsran-ue    "$CHARTS_DIR/srsue"
helm_deploy rl-xapp      "$CHARTS_DIR/rl-xapp"
helm_deploy telegraf     "$CHARTS_DIR/telegraf"

echo ""
echo "==> Checking pod status..."
errors=0
check_pod_status "component=gnb"             "gNB"      180 || ((errors++))
check_pod_status "component=ue"              "UE"       300 || ((errors++))
check_pod_status "app.kubernetes.io/name=rl-xapp" "RL-xApp" 120 || ((errors++))
check_pod_status "app=telegraf"              "Telegraf" 120 || ((errors++))

echo ""
echo "==> Pod summary ($NAMESPACE):"
kubectl get pods -n "$NAMESPACE" -o wide

if (( errors > 0 )); then
  echo ""
  echo "[WARN] $errors component(s) not Ready — see diagnostics above"
  exit 1
fi

echo ""
echo "==> All components Ready."

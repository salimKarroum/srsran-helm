#!/bin/bash
set -e
NS=open5gs
REPO_DIR=~/srsran-helm
BRANCH=claude/check-pod-status-3pjYT

echo "=== 1. Mise a jour du code ==="
cd $REPO_DIR
git fetch origin $BRANCH
git reset --hard origin/$BRANCH

echo "=== 2. Deploiement channel-emulator ==="
helm upgrade channel-emulator ./charts/channel-emulator -n $NS
kubectl rollout restart deployment/channel-emulator -n $NS
kubectl rollout status deployment/channel-emulator -n $NS --timeout=60s

echo "=== 3. Redemarrage gNB ==="
kubectl rollout restart deployment/srsran-gnb -n $NS
kubectl rollout status deployment/srsran-gnb -n $NS --timeout=60s

echo "=== 4. Injection ZMQ pour debloquer le gNB ==="
sleep 5
EMULATOR_POD=$(kubectl get pod -n $NS --no-headers | grep channel-emulator | grep Running | awk '{print $1}')
echo "Emulator pod: $EMULATOR_POD"
kubectl exec -n $NS $EMULATOR_POD -- python3 -c "
import zmq, numpy as np, time
ctx = zmq.Context()
s = ctx.socket(zmq.REQ)
s.setsockopt(zmq.RCVTIMEO, 5000)
s.connect('tcp://10.10.3.234:2000')
s.send(np.zeros(46080, dtype=np.complex64).tobytes())
try:
    s.recv()
    print('Injection gNB OK')
except:
    print('Injection timeout (gNB deja pret)')
s.close()
"

echo "=== 5. Redemarrage des UEs ==="
kubectl rollout restart deployment/srsran-ue-srsue-embb deployment/srsran-ue-srsue-urllc deployment/srsran-ue-srsue-mmtc -n $NS
kubectl rollout status deployment/srsran-ue-srsue-embb -n $NS --timeout=60s
kubectl rollout status deployment/srsran-ue-srsue-urllc -n $NS --timeout=60s
kubectl rollout status deployment/srsran-ue-srsue-mmtc -n $NS --timeout=60s

echo "=== 6. Attente attachement UEs (60s) ==="
sleep 60

echo "=== 7. Verification ==="
echo "--- Emulateur ---"
kubectl logs -n $NS deployment/channel-emulator --tail=5

echo "--- UE eMBB ---"
EMBB_POD=$(kubectl get pod -n $NS --no-headers | grep srsue-embb | grep Running | awk '{print $1}')
kubectl logs -n $NS pod/$EMBB_POD --tail=10 | grep -E "Connected|tun_srsue|RAR|SFN|PRACH|Attach" || kubectl logs -n $NS pod/$EMBB_POD --tail=10

echo "--- IP tun_srsue eMBB ---"
kubectl exec -n $NS pod/$EMBB_POD -- ip addr show tun_srsue 2>/dev/null && echo "UE ATTACHE !" || echo "UE pas encore attache"

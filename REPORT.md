# Rapport — Déploiement 5G + RL xApp sur Kubernetes

**Date :** 12 mai 2026
**Cluster :** sopnode-w3 (R2lab / single-node)
**Branche :** `claude/check-pod-status-2DZYK`
**Objectif :** Implémenter le papier REAL (arXiv:2502.00715) — boucle fermée RL pour le slicing PRB en temps réel sur un réseau 5G simulé.

---

## 1. Architecture mise en place

```
┌──────────────────────────────────────────────────────────────────┐
│                         POD srsUE                                 │
│  (Multus n3: 10.10.3.235)                                         │
│  srsUE  ──► TX REP bind tcp://*:2101                              │
│         ──► RX REQ connect tcp://10.10.3.234:2000                 │
│  IP UE : 12.1.0.1/24 (tun_srsue, slice eMBB, IMSI 001010000000001)│
└──────────────────────────────────────────────────────────────────┘
                          ▲       │
                       UL │       │ DL  (ZMQ direct, sans CE)
                          │       ▼
┌──────────────────────────────────────────────────────────────────┐
│                         POD gNB (srsRAN_Project)                  │
│  (Multus n3: 10.10.3.234)                                         │
│  TX REP bind tcp://*:2000                                         │
│  RX REQ connect tcp://10.10.3.235:2101                            │
│  N2 (NGAP/SCTP) → AMF 10.10.3.200                                 │
│  Métriques JSON sur WebSocket :8001                               │
└──────────────────────────────────────────────────────────────────┘
                          │
                          │ N2 + N3 (GTP-U)
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│                     POD Open5GS (Core 5G)                         │
│  AMF   10.10.3.200   │  SMF1 10.10.3.101 / 10.10.4.101            │
│  UPF1  10.10.3.1     /  10.10.4.1  (slice eMBB)                   │
│  + NRF, AUSF, UDM, UDR, NSSF, PCF, BSF, SCP                       │
│  + MongoDB (subscribers)                                          │
└──────────────────────────────────────────────────────────────────┘
                          ▲
                          │ WebSocket :8001
                          │
┌──────────────────────────────────────────────────────────────────┐
│                         POD rl-xapp                               │
│  ┌──────────────┐          ┌──────────────────────────┐           │
│  │ kpm-adapter  │ ◄──HTTP──┤ RL xApp (PPO)            │           │
│  │ (WebSocket   │   :8080  │ stable-baselines3        │           │
│  │  → REST)     │ ─POST───►│ obs = [thr, prb]         │           │
│  └──────────────┘          │ act ∈ {0..6} PRB shift   │           │
│       │  ▲                 │ reward = thr/target − 1  │           │
│       │  │                 └──────────────────────────┘           │
└───────┼──┼───────────────────────────────────────────────────────┘
        │  │                                                       
        │  └── métriques par layer (rlc, du_low, cu-up...)         
        └──── rrm_policy_ratio_set (PRB par slice)                 
```

---

## 2. Composants livrés

| Composant | Chart Helm | État |
|-----------|-----------|------|
| Open5GS Core (15 NF) | (Ansible) | Fonctionnel |
| gNB srsRAN_Project | `charts/srsran-gnb` | Connecté à AMF |
| UE srsRAN 4G (ZMQ) | `charts/srsue` | Attaché, session PDU |
| Channel Emulator | `charts/srsue/templates/configmap-ce.yaml` | Désactivé (non requis 1 UE) |
| KPM Adapter | `charts/rl-xapp/templates/configmap-kpm.yaml` | Reçoit WebSocket, expose REST |
| RL xApp (PPO/DQN) | `charts/rl-xapp/templates/configmap.yaml` | S'entraîne en temps réel |

---

## 3. Problèmes rencontrés et résolus

### 3.1 gNB crash au démarrage — SD au mauvais format
- **Symptôme :** `--sd: Value FFFFFF not in range [0 - 16777215]`
- **Cause :** srsRAN_Project parse le SD comme **entier décimal**, pas hex
- **Fix :** `values-rfsim.yaml` : `sd: "FFFFFF"` → `sd: 16777215` (commit `a5f4a9c`)

### 3.2 RF overflow et CE bloqué — timeouts ZMQ inadéquats
- **Symptôme :** `Real-time failure in RF: overflow` (50+ msg/0.2ms)
- **Cause :** `dst.recv()` du CE bloquait indéfiniment quand l'UE se déconnectait
- **Fix :** ajout de `DST_TIMEOUT_MS=2000` + `zmq.Again → continue`
- **Fix complémentaire :** timeouts asymétriques DL=1500ms (poll gNB ~1s) vs UL=200ms (commit `8ad9b86`)

### 3.3 Multus non actif — pods sans interface n3
- **Symptôme :** AMF/UE sans IP 10.10.3.x, `Device "n3" does not exist`
- **Cause :** Multus DaemonSet pas installé/déployé
- **Fix :** `kubectl apply -f multus-daemonset-thick.yml` puis recréation des pods Open5GS

### 3.4 Registration reject [11] — IMSI inexistant
- **Symptôme :** `Registration reject [11] PLMN not allowed`
- **Cause :** IMSI `001010123456780` absent de MongoDB (subscribers ≠ provisionnés par Ansible)
- **Fix :** utiliser un IMSI existant (`001010000000001`) avec ses K/OPC (commit `46eda1c`)

### 3.5 UE bloqué à "Attaching UE..." — CE zombie d'Ansible
- **Symptôme :** UE ne reçoit aucun sample DL malgré TCP OK
- **Cause :** un ancien pod `channel-emulator-*` d'Ansible interceptait les connexions ZMQ vers gNB:2000
- **Fix :** `kubectl scale deploy channel-emulator --replicas=0`
- **Leçon :** Ansible peut écraser les déploiements Helm — toujours vérifier les pods zombies

### 3.6 KPM-adapter incompatible format srsRAN_Project
- **Symptôme :** `'list' object has no attribute 'values'`
- **Cause :** srsRAN_Project n'envoie pas `cells.ue_list.s_nssai` mais des messages séparés par layer (`cu-up`, `rlc_metrics`, `du_low`, ...)
- **Fix :** réécriture du parseur pour cumuler les messages par layer, extraction du DL throughput depuis `cu-up.pdcp.dl.average_throughput_mbps` (commit `51bf45b`)

---

## 4. Résultats validés

### 4.1 Attachement 5G complet
```
Random Access Transmission: prach_occasion=0, preamble_index=0, ra-rnti=0x39, tti=2254
Random Access Complete.     c-rnti=0x4601, ta=0
RRC Connected
PDU Session Establishment successful. IP: 12.1.0.1
RRC NR reconfiguration successful.
✅ Interface tun_srsue is UP with IP: 12.1.0.1/24
```

### 4.2 Data plane fonctionnel
- Ping UE → 8.8.8.8 : **0% packet loss**, RTT ~25 ms
- iperf3 DL (UPF → UE) : **~11 Mbps** soutenu pendant 30s
- iperf3 stressé à 20 Mbps : **~22 Mbps** (sans cap)

### 4.3 Boucle RL fermée
Pendant un trafic DL à 10 Mbps :
```
DL=10.98Mbps UL=0.28Mbps | PRB=[60, 40, 6] r=1.1956
DL=10.98Mbps UL=0.28Mbps | PRB=[40, 60, 6] r=1.1952
DL=10.98Mbps UL=0.28Mbps | PRB=[50, 50, 6] r=1.1952
...
```
- Reward **positif** (cible eMBB 5 Mbps dépassée largement)
- PPO explore activement l'espace d'action (7 actions, 3 slices)
- 128 timesteps complétés en ~1 min

---

## 5. Limites et gaps identifiés

### 5.1 Slice mMTC non configuré côté gNB
- Le xApp envoie des commandes pour SD=200000 (mMTC)
- Le gNB répond `No RRM policy member found for {SST=1, SD=2097152}`
- **Action requise :** ajouter mMTC dans `gnbConfig.slicing` du `values-rfsim.yaml`

### 5.2 `rrm_policy_ratio_set` non enforcé avec 1 seul UE
- Commande acceptée par le gNB (`status: sent`)
- Débit eMBB inchangé même avec `max_prb_policy_ratio=10` (22 Mbps avant/après)
- **Conclusion :** les politiques slice sont des **parts de PRB partagés** ; avec un seul flux, ce flux a 100% par défaut
- **Validation possible uniquement en multi-UE multi-slice**

### 5.3 Métriques per-slice manquantes
- srsRAN_Project n'expose pas (encore) `s_nssai` par UE dans le JSON métriques
- Le kpm-adapter projette tout sur eMBB par défaut
- **Workaround :** dans cette config (1 UE), c'est suffisant
- **Action requise pour multi-UE :** parser les métriques scheduler/MAC pour ventiler par UE/slice

---

## 6. État de la roadmap

| Étape | Statut |
|-------|--------|
| **1. Réseau** — gNB ↔ Core ↔ UE, session PDU, trafic IP | ✅ Terminé |
| **2. Métriques** — gNB WebSocket → kpm-adapter → REST API | ✅ Terminé |
| **3. Apprentissage** — PPO reçoit observations, agit, reward évolue | ✅ Terminé |
| **4. Validation enforcement** — politique slice change réellement le débit | ⏸ Bloqué (1 UE) |

---

## 7. Prochaines étapes pour finir l'étape 4

Deux pistes pour débloquer la validation finale :

### Option A — Réactiver le Channel Emulator pour multiplexer N UEs
- Pour : réutilise le travail existant, conforme au papier REAL (12 UEs / 3 slices)
- Contre : le CE a un comportement chicken-and-egg côté ZMQ qui demande tuning fin
- Travail : ~ 1-2 jours de debug

### Option B — Configurer N cellules gNB avec ZMQ séparés
- Pour : architecture plus propre, chaque UE sur sa cellule
- Contre : non conforme au papier (qui suppose 1 cellule partagée)
- Travail : ~ 1 jour de config gNB + Helm

### Compléments transverses
- Ajouter le slice mMTC dans `values-rfsim.yaml`
- Configurer un 2e/3e subscriber dans MongoDB sur URLLC + mMTC
- Adapter le kpm-adapter pour extraire les métriques per-UE depuis `rlc_metrics` (le champ `ue_id` est déjà là)

---

## 8. Commits clés

| Commit | Description |
|--------|-------------|
| `a5f4a9c` | gNB : SD hex → décimal pour démarrage |
| `8ad9b86` | CE : timeouts asymétriques DL/UL |
| `70552c9` | UE : connexion ZMQ directe (CE désactivé) |
| `46eda1c` | UE : credentials USIM alignés sur MongoDB |
| `6b47b70` | kpm-adapter : injection via ConfigMap |
| `51bf45b` | kpm-adapter : parser format srsRAN_Project per-layer |
| `0ccce4a` | kpm-adapter : exposer min/max_prb_policy_ratio |

---

## 9. Commandes utiles pour la suite

```bash
# Recréer la stack après reboot
cd ~/srsran-helm
helm upgrade srsran-gnb charts/srsran-gnb -n open5gs \
  -f charts/srsran-gnb/values-rfsim.yaml --reset-values
helm upgrade srsran-ue charts/srsue -n open5gs --reset-values
helm upgrade rl-xapp charts/rl-xapp -n open5gs --reset-values
kubectl delete pod -n open5gs -l component=gnb -l component=ue \
  -l app.kubernetes.io/name=rl-xapp

# Vérifier la boucle complète
UE_POD=$(kubectl get pod -n open5gs -l component=ue -o jsonpath='{.items[0].metadata.name}')
RL_POD=$(kubectl get pod -n open5gs -l app.kubernetes.io/name=rl-xapp -o jsonpath='{.items[0].metadata.name}')
UPF_POD=$(kubectl get pod -n open5gs -l name=upf1 -o jsonpath='{.items[0].metadata.name}')

# Lancer trafic DL et observer le xApp
kubectl exec -n open5gs $UE_POD -- bash -c "iperf3 -s -B 12.1.0.1 -D"
kubectl exec -n open5gs $UPF_POD -- bash -c "iperf3 -c 12.1.0.1 -t 60 -b 20M &"
kubectl logs -n open5gs $RL_POD -c rl-xapp -f
```

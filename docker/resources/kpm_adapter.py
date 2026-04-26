#!/usr/bin/env python3
"""
KPM Adapter — pont entre les métriques srsRAN (WebSocket JSON) et le xApp RL.

Rôle :
  1. Se connecte au WebSocket du gNB (port 8001, commande metrics_subscribe)
  2. Parse le JSON srsRAN : par cellule, par UE, avec s_nssai (slice)
  3. Agrège les métriques par slice (eMBB, URLLC, mMTC)
  4. Expose une API REST sur :8080 que le xApp PPO interroge

Format JSON srsRAN (exemple) :
{
  "timestamp": "2024-01-01T00:00:00.000",
  "cells": {
    "0": {
      "ue_list": {
        "0": {
          "pci": 1, "rnti": 17921,
          "dl_brate": 50000000,       <- bits/s downlink
          "ul_brate": 1000000,
          "pusch_snr_db": 15.2,       <- SINR uplink
          "dl_nof_ok": 10, "dl_nof_nok": 0,
          "s_nssai": {"sst": 1, "sd": "000001"}  <- slice ID
        }
      },
      "cell_metrics": {
        "dl_prb_usage": 0.45,   <- fraction 0-1 des PRBs utilisés DL
        "ul_prb_usage": 0.20
      }
    }
  }
}

API exposée :
  GET /v1/nodeb/{gnb_id}/kpm  →  { embb_dl_mbps, urllc_dl_mbps, mmtc_dl_mbps,
                                    embb_sinr_db, urllc_sinr_db, mmtc_sinr_db,
                                    embb_prb_usage, urllc_prb_usage, mmtc_prb_usage,
                                    ue_count_embb, ue_count_urllc, ue_count_mmtc }
  GET /health                 →  { status: ok }
  GET /metrics/raw            →  dernier JSON brut srsRAN
"""

import os
import json
import logging
import threading
from collections import defaultdict
from contextlib import suppress
from http.server import BaseHTTPRequestHandler, HTTPServer
from time import sleep

import websocket

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("kpm-adapter")

# ── Configuration ────────────────────────────────────────────────────────────
WS_URL       = os.getenv("WS_URL", "srsran-gnb-metrics:8001")
REST_PORT    = int(os.getenv("REST_PORT", "8080"))
TOTAL_PRB    = int(os.getenv("TOTAL_PRB", "106"))

# Mapping s_nssai → nom de slice
# Configurable via env : SLICE_MAP='{"1:000001":"embb","1:000002":"urllc","1:000003":"mmtc"}'
_default_slice_map = {"1:000001": "embb", "1:000002": "urllc", "1:000003": "mmtc"}
SLICE_MAP = json.loads(os.getenv("SLICE_MAP", json.dumps(_default_slice_map)))

# Fallback : SST seul sans SD
SST_FALLBACK = {"1": "embb", "2": "urllc", "3": "mmtc"}

# ── État partagé (thread-safe via verrou) ────────────────────────────────────
_lock       = threading.Lock()
_raw        = {}          # dernier JSON brut reçu
_kpm        = {}          # métriques agrégées par slice

# ── Parsing et agrégation ────────────────────────────────────────────────────

def _slice_name(s_nssai: dict) -> str:
    """Retourne le nom de la slice à partir de son s_nssai."""
    if not s_nssai:
        return "unknown"
    sst = str(s_nssai.get("sst", ""))
    sd  = str(s_nssai.get("sd", ""))
    key = f"{sst}:{sd}" if sd else sst
    return SLICE_MAP.get(key) or SST_FALLBACK.get(sst, "unknown")


def _aggregate(data: dict) -> dict:
    """
    Agrège les métriques srsRAN par slice.
    Retourne un dict plat consommable directement par le xApp.
    """
    # Accumulateurs par slice
    slices = defaultdict(lambda: {
        "dl_brate_sum": 0.0,   # bits/s total
        "sinr_sum":     0.0,
        "ue_count":     0,
        "prb_usage":    0.0,   # fraction 0-1, moyenne sur les cellules
        "cell_count":   0,
        "dl_ok":        0,
        "dl_nok":       0,
    })

    cells = data.get("cells", {})
    for cell in cells.values():
        # Métriques par UE
        for ue in cell.get("ue_list", {}).values():
            name = _slice_name(ue.get("s_nssai", {}))
            s = slices[name]
            s["dl_brate_sum"] += float(ue.get("dl_brate", 0))
            s["sinr_sum"]     += float(ue.get("pusch_snr_db", 0))
            s["dl_ok"]        += int(ue.get("dl_nof_ok",  0))
            s["dl_nok"]       += int(ue.get("dl_nof_nok", 0))
            s["ue_count"]     += 1

        # Métriques cellule (PRB usage partagé entre toutes les slices de la cellule)
        cm = cell.get("cell_metrics", {})
        prb_dl = float(cm.get("dl_prb_usage", 0))
        for s in slices.values():
            s["prb_usage"]  += prb_dl
            s["cell_count"] += 1

    # Construire le résultat final
    result = {}
    for slice_name in ("embb", "urllc", "mmtc"):
        s = slices[slice_name]
        ue_n = max(s["ue_count"], 1)
        cell_n = max(s["cell_count"], 1)
        dl_total = s["dl_ok"] + s["dl_nok"]

        result[f"{slice_name}_dl_mbps"]    = round(s["dl_brate_sum"] / 1e6, 3)
        result[f"{slice_name}_sinr_db"]    = round(s["sinr_sum"] / ue_n, 2)
        result[f"{slice_name}_prb_usage"]  = round(s["prb_usage"] / cell_n, 4)
        result[f"{slice_name}_ue_count"]   = s["ue_count"]
        result[f"{slice_name}_bler"]       = round(
            s["dl_nok"] / dl_total if dl_total > 0 else 0.0, 4
        )

    return result


# ── WebSocket client ─────────────────────────────────────────────────────────

def _on_open(ws: websocket.WebSocketApp):
    log.info(f"Connecté au gNB WebSocket ({WS_URL})")
    ws.send(json.dumps({"cmd": "metrics_subscribe"}))


def _on_message(_ws, message: str):
    with suppress(json.JSONDecodeError):
        data = json.loads(message)
        if "cmd" in data:   # messages de contrôle, ignorer
            return
        kpm = _aggregate(data)
        with _lock:
            _raw.clear()
            _raw.update(data)
            _kpm.clear()
            _kpm.update(kpm)
        log.debug(f"KPM mis à jour : {kpm}")


def _on_error(_ws, error):
    log.warning(f"WebSocket error : {error}")


def _on_close(_ws, *_):
    log.info("WebSocket fermé, reconnexion dans 2s...")


def _ws_loop():
    while True:
        ws = websocket.WebSocketApp(
            f"ws://{WS_URL}",
            on_open=_on_open,
            on_message=_on_message,
            on_error=_on_error,
            on_close=_on_close,
        )
        ws.run_forever()
        sleep(2)


# ── REST API ─────────────────────────────────────────────────────────────────

class KPMHandler(BaseHTTPRequestHandler):

    def log_message(self, *_):
        pass  # silence les logs HTTP

    def _send_json(self, code: int, body: dict):
        payload = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        path = self.path.rstrip("/")

        # /health
        if path == "/health":
            self._send_json(200, {"status": "ok", "ws_url": WS_URL})
            return

        # /metrics/raw
        if path == "/metrics/raw":
            with _lock:
                data = dict(_raw)
            self._send_json(200, data)
            return

        # /v1/nodeb/{gnb_id}/kpm  (le xApp peut passer n'importe quel gnb_id)
        if "/kpm" in path:
            with _lock:
                data = dict(_kpm)
            if not data:
                self._send_json(503, {"error": "no KPM data yet"})
                return
            self._send_json(200, data)
            return

        self._send_json(404, {"error": "not found"})


def _rest_loop():
    server = HTTPServer(("0.0.0.0", REST_PORT), KPMHandler)
    log.info(f"REST API démarrée sur :{REST_PORT}")
    server.serve_forever()


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info(f"KPM Adapter — WS={WS_URL}  REST=:{REST_PORT}  TOTAL_PRB={TOTAL_PRB}")

    # REST dans un thread séparé
    t = threading.Thread(target=_rest_loop, daemon=True)
    t.start()

    # WebSocket dans le thread principal (boucle infinie avec reconnexion)
    _ws_loop()

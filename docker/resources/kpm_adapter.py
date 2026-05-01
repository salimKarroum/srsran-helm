#!/usr/bin/env python3
"""
KPM Adapter — pont entre les métriques srsRAN (WebSocket JSON) et le xApp RL.

Rôle :
  1. Se connecte au WebSocket du gNB (port 8001, commande metrics_subscribe)
  2. Parse le JSON srsRAN : par cellule, par UE, avec s_nssai (slice)
  3. Agrège les métriques par slice (eMBB, URLLC, mMTC)
  4. Expose une API REST sur :8080 que le xApp PPO interroge

API exposée :
  GET /v1/nodeb/{gnb_id}/kpm  →  { embb_dl_mbps, urllc_dl_mbps, mmtc_dl_mbps, ... }
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

WS_URL    = os.getenv("WS_URL", "srsran-gnb-metrics:8001")
REST_PORT = int(os.getenv("REST_PORT", "8080"))
TOTAL_PRB = int(os.getenv("TOTAL_PRB", "106"))

_default_slice_map = {"1:000001": "embb", "1:000002": "urllc", "1:000003": "mmtc"}
SLICE_MAP = json.loads(os.getenv("SLICE_MAP", json.dumps(_default_slice_map)))
SST_FALLBACK = {"1": "embb", "2": "urllc", "3": "mmtc"}

_lock = threading.Lock()
_raw  = {}
_kpm  = {}


def _slice_name(s_nssai: dict) -> str:
    if not s_nssai:
        return "unknown"
    sst = str(s_nssai.get("sst", ""))
    sd  = str(s_nssai.get("sd", ""))
    key = f"{sst}:{sd}" if sd else sst
    return SLICE_MAP.get(key) or SST_FALLBACK.get(sst, "unknown")


def _aggregate(data: dict) -> dict:
    slices = defaultdict(lambda: {
        "dl_brate_sum": 0.0,
        "sinr_sum":     0.0,
        "ue_count":     0,
        "prb_usage":    0.0,
        "cell_count":   0,
        "dl_ok":        0,
        "dl_nok":       0,
    })

    cells = data.get("cells", {})
    for cell in cells.values():
        for ue in cell.get("ue_list", {}).values():
            name = _slice_name(ue.get("s_nssai", {}))
            s = slices[name]
            s["dl_brate_sum"] += float(ue.get("dl_brate", 0))
            s["sinr_sum"]     += float(ue.get("pusch_snr_db", 0))
            s["dl_ok"]        += int(ue.get("dl_nof_ok",  0))
            s["dl_nok"]       += int(ue.get("dl_nof_nok", 0))
            s["ue_count"]     += 1
        cm = cell.get("cell_metrics", {})
        prb_dl = float(cm.get("dl_prb_usage", 0))
        for s in slices.values():
            s["prb_usage"]  += prb_dl
            s["cell_count"] += 1

    result = {}
    for slice_name in ("embb", "urllc", "mmtc"):
        s = slices[slice_name]
        ue_n   = max(s["ue_count"], 1)
        cell_n = max(s["cell_count"], 1)
        dl_total = s["dl_ok"] + s["dl_nok"]
        result[f"{slice_name}_dl_mbps"]   = round(s["dl_brate_sum"] / 1e6, 3)
        result[f"{slice_name}_sinr_db"]   = round(s["sinr_sum"] / ue_n, 2)
        result[f"{slice_name}_prb_usage"] = round(s["prb_usage"] / cell_n, 4)
        result[f"{slice_name}_ue_count"]  = s["ue_count"]
        result[f"{slice_name}_bler"]      = round(
            s["dl_nok"] / dl_total if dl_total > 0 else 0.0, 4)
    return result


def _on_open(ws):
    log.info(f"Connecté au gNB WebSocket ({WS_URL})")
    ws.send(json.dumps({"cmd": "metrics_subscribe"}))


def _on_message(_ws, message):
    with suppress(json.JSONDecodeError):
        data = json.loads(message)
        if "cmd" in data:
            return
        kpm = _aggregate(data)
        with _lock:
            _raw.clear(); _raw.update(data)
            _kpm.clear(); _kpm.update(kpm)


def _on_error(_ws, error):
    log.warning(f"WebSocket error : {error}")


def _on_close(_ws, *_):
    log.info("WebSocket fermé, reconnexion dans 2s...")


def _ws_loop():
    while True:
        ws = websocket.WebSocketApp(
            f"ws://{WS_URL}",
            on_open=_on_open, on_message=_on_message,
            on_error=_on_error, on_close=_on_close,
        )
        ws.run_forever()
        sleep(2)


class KPMHandler(BaseHTTPRequestHandler):
    def log_message(self, *_): pass

    def _send_json(self, code, body):
        payload = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        path = self.path.rstrip("/")
        if path == "/health":
            self._send_json(200, {"status": "ok", "ws_url": WS_URL})
        elif path == "/metrics/raw":
            with _lock: data = dict(_raw)
            self._send_json(200, data)
        elif "/kpm" in path:
            with _lock: data = dict(_kpm)
            if not data:
                self._send_json(503, {"error": "no KPM data yet"})
            else:
                self._send_json(200, data)
        else:
            self._send_json(404, {"error": "not found"})


def _rest_loop():
    server = HTTPServer(("0.0.0.0", REST_PORT), KPMHandler)
    log.info(f"REST API démarrée sur :{REST_PORT}")
    server.serve_forever()


if __name__ == "__main__":
    log.info(f"KPM Adapter — WS={WS_URL}  REST=:{REST_PORT}  TOTAL_PRB={TOTAL_PRB}")
    threading.Thread(target=_rest_loop, daemon=True).start()
    _ws_loop()

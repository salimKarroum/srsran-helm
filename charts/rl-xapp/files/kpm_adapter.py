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

_default_slice_map = {"1:FFFFFF": "embb", "1:16777215": "embb", "1:100000": "urllc", "1:1048576": "urllc"}
SLICE_MAP = json.loads(os.getenv("SLICE_MAP", json.dumps(_default_slice_map)))
SST_FALLBACK = {"1": "embb", "2": "urllc", "3": "mmtc"}

_lock = threading.Lock()
_raw  = {}
_kpm  = {}
_ws_ref: list = []  # holds current WebSocketApp instance for sending commands


def _slice_name(s_nssai: dict) -> str:
    if not s_nssai:
        return "unknown"
    sst = str(s_nssai.get("sst", ""))
    sd  = str(s_nssai.get("sd", ""))
    key = f"{sst}:{sd}" if sd else sst
    return SLICE_MAP.get(key) or SST_FALLBACK.get(sst, "unknown")


def _aggregate(state: dict) -> dict:
    """
    Aggregate metrics from srsRAN_Project's per-layer JSON messages.

    state is the accumulated _raw dict, where each top-level key is a layer
    name (cu-up, rlc_metrics, du_low, executor_metrics, ...). We pull the
    relevant fields and project them onto the eMBB / URLLC / mMTC slices.

    Until per-slice metrics are exposed by srsRAN, every UE is assumed to
    belong to eMBB (the default slice). URLLC and mMTC remain zero.
    """
    cu_up_dl = state.get("cu-up", {}).get("pdcp", {}).get("dl", {}) or {}
    cu_up_ul = state.get("cu-up", {}).get("pdcp", {}).get("ul", {}) or {}
    du_low_dl = state.get("du_low", {}).get("dl", {}) or {}

    dl_mbps = float(cu_up_dl.get("average_throughput_mbps", 0.0))
    ul_mbps = float(cu_up_ul.get("average_throughput_mbps", 0.0))
    dl_latency_us = float(du_low_dl.get("average_latency_us", 0.0))

    rlc = state.get("rlc_metrics", {}) or {}
    rlc_tx_bytes = int(rlc.get("tx", {}).get("num_sdu_bytes", 0))
    rlc_rx_bytes = int(rlc.get("rx", {}).get("num_sdu_bytes", 0))
    ue_count = 1 if rlc.get("ue_id") is not None else 0

    result = {
        "embb_dl_mbps":   round(dl_mbps, 3),
        "embb_ul_mbps":   round(ul_mbps, 3),
        "embb_sinr_db":   0.0,
        "embb_prb_usage": 0.0,
        "embb_ue_count":  ue_count,
        "embb_bler":      0.0,
        "embb_dl_latency_us": round(dl_latency_us, 1),
        "embb_rlc_tx_bytes":  rlc_tx_bytes,
        "embb_rlc_rx_bytes":  rlc_rx_bytes,
    }
    for slice_name in ("urllc", "mmtc"):
        result[f"{slice_name}_dl_mbps"]   = 0.0
        result[f"{slice_name}_sinr_db"]   = 0.0
        result[f"{slice_name}_prb_usage"] = 0.0
        result[f"{slice_name}_ue_count"]  = 0
        result[f"{slice_name}_bler"]      = 0.0
    return result


def _on_open(ws):
    log.info(f"Connecté au gNB WebSocket ({WS_URL})")
    _ws_ref.clear()
    _ws_ref.append(ws)
    ws.send(json.dumps({"cmd": "metrics_subscribe"}))


def _on_message(_ws, message):
    with suppress(json.JSONDecodeError):
        data = json.loads(message)
        if "cmd" in data:
            return
        # srsRAN_Project emits one message per layer (cu-up, rlc_metrics, du_low,
        # executor_metrics, ...). Merge them into _raw instead of overwriting,
        # so _aggregate can see the latest value of every layer.
        with _lock:
            for k, v in data.items():
                _raw[k] = v
            kpm = _aggregate(_raw)
            _kpm.clear(); _kpm.update(kpm)


def _on_error(_ws, error):
    log.warning(f"WebSocket error : {error}")


def _on_close(_ws, *_):
    _ws_ref.clear()
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

    def do_POST(self):
        path = self.path.rstrip("/")
        if "/rrm/policy" not in path:
            self._send_json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            policy = json.loads(body)
        except json.JSONDecodeError:
            self._send_json(400, {"error": "invalid JSON"})
            return
        sst = int(policy.get("sst", 1))
        sd  = int(policy.get("sd", 0))
        # All three knobs are accepted independently. dedicated_ratio is the
        # historical default for the xApp; min/max let us cap a slice hard.
        ratio = int(policy.get("dedicated_ratio", 33))
        min_r = int(policy.get("min_prb_policy_ratio", 0))
        max_r = int(policy.get("max_prb_policy_ratio", 100))
        cmd = {
            "cmd": "rrm_policy_ratio_set",
            "policies": {
                "resourceType": "PRB",
                "rRMPolicyMemberList": [{"plmn": "00101", "sst": sst, "sd": sd}],
                "min_prb_policy_ratio": min_r,
                "max_prb_policy_ratio": max_r,
                "dedicated_ratio": ratio,
            }
        }
        if _ws_ref:
            try:
                _ws_ref[0].send(json.dumps(cmd))
                self._send_json(200, {"status": "sent", "cmd": cmd})
            except Exception as e:
                self._send_json(503, {"error": str(e)})
        else:
            self._send_json(503, {"error": "WebSocket not connected"})

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

#!/usr/bin/env python3
"""
Channel Emulator — ZMQ REQ/REP bridge compatible srsRAN_Project.

srsRAN_Project ZMQ driver socket types:
  tx_port → REP socket (binds, sends IQ on request)
  rx_port → REQ socket (connects, requests IQ)

Reactive bridge model (avoids startup deadlock):
  DL: UE RX REQ requests → CE fetches from gNB TX REP → CE serves UE
  UL: gNB RX REQ requests → CE fetches from UE TX REP → CE serves gNB

Topologie :
  gNB TX (REP bind :2000) ←─REQ── [DL] ──REP bind :2100─→ UE  RX (REQ :2100)
  UE  TX (REP bind :2101) ←─REQ── [UL] ──REP bind :2001─→ gNB RX (REQ :2001)
"""

import os, time, threading, logging
import numpy as np
import zmq

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("channel-emulator")

# ── Configuration ────────────────────────────────────────────────────
GNB_TX_ADDR = os.getenv("GNB_TX_ADDR",  "tcp://10.10.3.234:2000")
UE_TX_ADDR  = os.getenv("UE_TX_ADDR",   "tcp://127.0.0.1:2101")
GNB_RX_BIND = os.getenv("GNB_RX_BIND",  "tcp://0.0.0.0:2001")
UE_RX_BIND  = os.getenv("UE_RX_BIND",   "tcp://0.0.0.0:2100")

D_MIN       = float(os.getenv("DISTANCE_MIN_KM",  "0.5"))
D_MAX       = float(os.getenv("DISTANCE_MAX_KM",  "2.0"))
DOPPLER_MAX = float(os.getenv("DOPPLER_MAX_HZ",   "50"))
NF_DB       = float(os.getenv("NOISE_FIGURE_DB",   "7"))
FC_GHZ      = float(os.getenv("CARRIER_FREQ_GHZ",  "1.8"))
RESET_S     = float(os.getenv("RESET_INTERVAL_S",  "5.0"))

# Timeout waiting for source to respond (ms). On timeout, send silence.
SRC_TIMEOUT_MS = int(os.getenv("SRC_TIMEOUT_MS", "10"))

# ── Canal partagé entre DL et UL ─────────────────────────────────────────
lock   = threading.Lock()
state  = {
    "d":       np.random.uniform(D_MIN, D_MAX),
    "doppler": np.random.uniform(-DOPPLER_MAX, DOPPLER_MAX),
    "ts":      time.time(),
}

def maybe_reset():
    if time.time() - state["ts"] > RESET_S:
        with lock:
            state["d"]       = np.random.uniform(D_MIN, D_MAX)
            state["doppler"] = np.random.uniform(-DOPPLER_MAX, DOPPLER_MAX)
            state["ts"]      = time.time()
        log.info(f"Canal reset → d={state['d']:.2f} km  dop={state['doppler']:.1f} Hz")

def get_params():
    with lock:
        return state["d"], state["doppler"]

# ── Modèle de canal ──────────────────────────────────────────────────
def fspl_linear(d_km, f_ghz):
    db = 20 * np.log10(d_km) + 20 * np.log10(f_ghz) + 92.45
    return 10 ** (-db / 20.0)

def awgn_noise(n, snr_db):
    p = 10 ** (-snr_db / 10.0)
    return (np.random.randn(n) + 1j * np.random.randn(n)) * np.sqrt(p / 2)

def apply_channel(sig, d_km, dop_hz, sr=23.04e6):
    n   = len(sig)
    if n == 0:
        return sig
    a   = fspl_linear(d_km, FC_GHZ)
    out = sig * a
    # single-tap multipath
    td = np.random.randint(0, 6)
    if td > 0 and n > td:
        out[td:] += (np.random.uniform(0.3, 0.7)
                     * np.exp(1j * np.random.uniform(0, 2 * np.pi))
                     * sig[:-td])
    # Doppler
    out *= np.exp(1j * 2 * np.pi * dop_hz * np.arange(n) / sr)
    # AWGN
    snr_db = -20 * np.log10(max(a, 1e-10)) - NF_DB
    out   += awgn_noise(n, snr_db)
    return out

# ── Reactive Bridge ──────────────────────────────────────────────────
def bridge(src_addr, dst_bind, name, doppler_sign=1.0, n_silence=23040):
    """
    Reactive bridge: destination requests first, then CE fetches from source.

    dst (REP bind) serves requests from the destination (UE RX or gNB RX).
    src (REQ connect) fetches IQ from the source (gNB TX or UE TX) on demand.

    If source doesn't respond within SRC_TIMEOUT_MS, silence is sent instead.
    This allows UL/DL to bootstrap independently.
    """
    ctx = zmq.Context()

    src = ctx.socket(zmq.REQ)
    src.setsockopt(zmq.LINGER, 0)
    src.setsockopt(zmq.RCVTIMEO, SRC_TIMEOUT_MS)
    src.connect(src_addr)

    dst = ctx.socket(zmq.REP)
    dst.setsockopt(zmq.LINGER, 0)
    dst.bind(dst_bind)

    log.info(f"[{name}] REP bind {dst_bind}  REQ → {src_addr}  (reactive mode)")

    silence = np.zeros(n_silence, dtype=np.complex64).tobytes()
    src_ok  = True  # track source health for logging

    while True:
        try:
            maybe_reset()
            d, dop = get_params()

            # 1. Wait for destination to request samples
            dst.recv()

            # 2. Fetch from source (with timeout fallback to silence)
            try:
                src.send(b"")
                raw = src.recv()
                if not src_ok:
                    log.info(f"[{name}] Source {src_addr} responsive again")
                    src_ok = True
                iq = np.frombuffer(raw, dtype=np.complex64).copy()
                iq_out = apply_channel(iq, d, doppler_sign * dop).astype(np.complex64)
                payload = iq_out.tobytes()
            except zmq.Again:
                # Source not responding — send silence so destination doesn't stall
                if src_ok:
                    log.warning(f"[{name}] Source {src_addr} not responding, sending silence")
                    src_ok = False
                # Reset REQ socket state after failed recv (must create new socket)
                src.close()
                src = ctx.socket(zmq.REQ)
                src.setsockopt(zmq.LINGER, 0)
                src.setsockopt(zmq.RCVTIMEO, SRC_TIMEOUT_MS)
                src.connect(src_addr)
                payload = silence

            # 3. Respond to destination
            dst.send(payload)

        except Exception as exc:
            log.error(f"[{name}] {exc}")
            time.sleep(0.01)


if __name__ == "__main__":
    log.info("Channel emulator (reactive REQ/REP) démarré")
    log.info(f"  DL : {GNB_TX_ADDR} ─► {UE_RX_BIND}  (UE requests first)")
    log.info(f"  UL : {UE_TX_ADDR}  ─► {GNB_RX_BIND}  (gNB requests first)")
    log.info(f"  Source timeout: {SRC_TIMEOUT_MS} ms → silence fallback")

    threads = [
        threading.Thread(target=bridge,
                         args=(GNB_TX_ADDR, UE_RX_BIND, "DL",  1.0), daemon=True),
        threading.Thread(target=bridge,
                         args=(UE_TX_ADDR,  GNB_RX_BIND, "UL", -1.0), daemon=True),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

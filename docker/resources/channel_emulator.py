#!/usr/bin/env python3
"""
Channel Emulator — ZMQ REQ/REP bridge compatible srsRAN_Project.

srsRAN_Project ZMQ driver socket types:
  tx_port → REP socket (binds, envoie IQ sur demande)
  rx_port → REQ socket (connecte, demande IQ)

Ce composant doit tourner en SIDECAR dans le pod srsUE pour partager
le namespace réseau (IP multus ex. 10.10.3.235) et pouvoir se binder
sur les ports que le gNB et le srsUE attendent.

Topologie :
  gNB TX (REP bind :2000) ←─REQ── [DL thread] ──REP bind :2100─→ UE  RX (REQ :2100)
  UE  TX (REP bind :2101) ←─REQ── [UL thread] ──REP bind :2001─→ gNB RX (REQ :2001)
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

# ── Bridge REQ/REP ──────────────────────────────────────────────────
def bridge(src_addr, dst_bind, name, doppler_sign=1.0):
    """
    - REQ connecte vers src (source REP bind) : tire les IQ.
    - REP se bind à dst : sert les IQ à la destination quand elle les demande.
    """
    ctx = zmq.Context()

    req = ctx.socket(zmq.REQ)
    req.setsockopt(zmq.LINGER, 0)
    req.connect(src_addr)

    rep = ctx.socket(zmq.REP)
    rep.setsockopt(zmq.LINGER, 0)
    rep.bind(dst_bind)

    log.info(f"[{name}] REQ → {src_addr}   REP bind {dst_bind}")

    while True:
        try:
            maybe_reset()
            d, dop = get_params()

            req.send(b"")
            raw = req.recv()

            iq = np.frombuffer(raw, dtype=np.complex64).copy()
            if len(iq) == 0:
                iq = np.zeros(23040, dtype=np.complex64)
            iq_out = apply_channel(iq, d, doppler_sign * dop).astype(np.complex64)

            rep.recv()
            rep.send(iq_out.tobytes())

        except Exception as exc:
            log.error(f"[{name}] {exc}")
            time.sleep(0.01)


if __name__ == "__main__":
    log.info("Channel emulator (REQ/REP) démarré")
    log.info(f"  DL : {GNB_TX_ADDR} ─► {UE_RX_BIND}")
    log.info(f"  UL : {UE_TX_ADDR}  ─► {GNB_RX_BIND}")

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

#!/usr/bin/env python3
"""
Channel Emulator — ZMQ REQ/REP bridge compatible srsRAN_Project.

Architecture à 4 threads (2 par direction) pour découpler polling et service :

  [gnb_tx_poller]  REQ→gNB TX     → dl_buf → [ue_rx_server]  REP bind :2100
  [ue_tx_poller]   REQ→srsUE TX   → ul_buf → [gnb_rx_server] REP bind :2001

Les serveurs répondent IMMÉDIATEMENT à partir du buffer (< 1 ms),
même si le poller est bloqué en attente de la source.
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

N_SAMPLES   = int(os.getenv("N_SAMPLES",       "23040"))
D_MIN       = float(os.getenv("DISTANCE_MIN_KM",  "0.5"))
D_MAX       = float(os.getenv("DISTANCE_MAX_KM",  "2.0"))
DOPPLER_MAX = float(os.getenv("DOPPLER_MAX_HZ",   "50"))
NF_DB       = float(os.getenv("NOISE_FIGURE_DB",   "7"))
FC_GHZ      = float(os.getenv("CARRIER_FREQ_GHZ",  "1.8"))
RESET_S     = float(os.getenv("RESET_INTERVAL_S",  "5.0"))

# ── Canal partagé ────────────────────────────────────────────────────
_chan_lock = threading.Lock()
_chan = {
    "d":       np.random.uniform(D_MIN, D_MAX),
    "doppler": np.random.uniform(-DOPPLER_MAX, DOPPLER_MAX),
    "ts":      time.time(),
}

def _maybe_reset():
    if time.time() - _chan["ts"] > RESET_S:
        with _chan_lock:
            _chan["d"]       = np.random.uniform(D_MIN, D_MAX)
            _chan["doppler"] = np.random.uniform(-DOPPLER_MAX, DOPPLER_MAX)
            _chan["ts"]      = time.time()
        log.info(f"Canal reset → d={_chan['d']:.2f} km  dop={_chan['doppler']:.1f} Hz")

def _get_params():
    with _chan_lock:
        return _chan["d"], _chan["doppler"]

# ── Modèle de canal ──────────────────────────────────────────────────
def _fspl_linear(d_km, f_ghz):
    db = 20 * np.log10(d_km) + 20 * np.log10(f_ghz) + 92.45
    return 10 ** (-db / 20.0)

def _awgn(n, snr_db):
    p = 10 ** (-snr_db / 10.0)
    return (np.random.randn(n) + 1j * np.random.randn(n)) * np.sqrt(p / 2)

def _apply_channel(sig, d_km, dop_hz, sr=23.04e6):
    n   = len(sig)
    a   = _fspl_linear(d_km, FC_GHZ)
    out = sig * a
    td  = np.random.randint(0, 6)
    if td > 0 and n > td:
        out[td:] += (np.random.uniform(0.3, 0.7)
                     * np.exp(1j * np.random.uniform(0, 2 * np.pi))
                     * sig[:-td])
    out *= np.exp(1j * 2 * np.pi * dop_hz * np.arange(n) / sr)
    snr_db = -20 * np.log10(max(a, 1e-10)) - NF_DB
    out   += _awgn(n, snr_db)
    return out

# ── Buffer thread-safe ────────────────────────────────────────────────
class IQBuffer:
    """Stocke le dernier bloc IQ reçu, sert des zéros tant que rien n'est arrivé."""
    def __init__(self, n):
        self._lock = threading.Lock()
        self._buf  = np.zeros(n, dtype=np.complex64)

    def put(self, arr):
        with self._lock:
            n = min(len(arr), len(self._buf))
            self._buf[:n] = arr[:n]

    def get(self):
        with self._lock:
            return self._buf.copy()

# ── Threads ───────────────────────────────────────────────────────────
def poller(src_addr, buf, name):
    """Tire continuellement des IQ depuis la source REP, remplit le buffer."""
    ctx = zmq.Context()
    req = ctx.socket(zmq.REQ)
    req.setsockopt(zmq.LINGER, 0)
    req.connect(src_addr)
    log.info(f"[{name}/poller] REQ → {src_addr}")
    while True:
        try:
            req.send(b"")
            raw = req.recv()
            iq  = np.frombuffer(raw, dtype=np.complex64).copy() if raw else np.zeros(N_SAMPLES, dtype=np.complex64)
            if len(iq) == 0:
                iq = np.zeros(N_SAMPLES, dtype=np.complex64)
            buf.put(iq)
        except Exception as exc:
            log.error(f"[{name}/poller] {exc}")
            time.sleep(0.001)


def server(dst_bind, buf, name, doppler_sign):
    """Répond immédiatement aux requêtes de la destination depuis le buffer."""
    ctx = zmq.Context()
    rep = ctx.socket(zmq.REP)
    rep.setsockopt(zmq.LINGER, 0)
    rep.bind(dst_bind)
    log.info(f"[{name}/server] REP bind {dst_bind}")
    while True:
        try:
            _maybe_reset()
            rep.recv()                          # requête de la destination
            d, dop = _get_params()
            iq     = buf.get()                  # dernier IQ disponible (zéros si vide)
            iq_out = _apply_channel(iq, d, doppler_sign * dop).astype(np.complex64)
            rep.send(iq_out.tobytes())
        except Exception as exc:
            log.error(f"[{name}/server] {exc}")
            time.sleep(0.001)


if __name__ == "__main__":
    log.info("Channel emulator démarré (architecture buffer 4 threads)")
    log.info(f"  DL : {GNB_TX_ADDR} ─► {UE_RX_BIND}")
    log.info(f"  UL : {UE_TX_ADDR}  ─► {GNB_RX_BIND}")

    dl_buf = IQBuffer(N_SAMPLES)
    ul_buf = IQBuffer(N_SAMPLES)

    threads = [
        threading.Thread(target=poller, args=(GNB_TX_ADDR, dl_buf, "DL"), daemon=True),
        threading.Thread(target=server, args=(UE_RX_BIND,  dl_buf, "DL",  1.0), daemon=True),
        threading.Thread(target=poller, args=(UE_TX_ADDR,  ul_buf, "UL"), daemon=True),
        threading.Thread(target=server, args=(GNB_RX_BIND, ul_buf, "UL", -1.0), daemon=True),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

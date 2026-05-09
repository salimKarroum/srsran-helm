#!/usr/bin/env python3
"""
Multi-UE Channel Emulator — 3 UEs (eMBB / URLLC / mMTC) vers 1 gNB.

Architecture :
  DL (gNB→UEs) :
    - 1 poller    REQ→gNB TX REP :2000      → dl_buf partagé
    - 3 servers   REP bind :2011/:2013/:2015 → servent dl_buf avec effets canal par-UE

  UL (UEs→gNB) :
    - 3 pollers   REQ→UE{n} TX REP          → ul_buf{n} par UE
    - 1 server    REP bind :2001             → sert sum(ul_buf) au gNB

Variables d'environnement :
  GNB_TX_ADDR       adresse TX REP du gNB   (défaut tcp://10.10.3.234:2000)
  GNB_RX_BIND       bind UL server           (défaut tcp://0.0.0.0:2001)
  UE1_TX_ADDR / UE1_RX_BIND / UE1_DIST_KM / UE1_DOPPLER_HZ / UE1_NOISE_DB
  UE2_TX_ADDR / UE2_RX_BIND / UE2_DIST_KM / UE2_DOPPLER_HZ / UE2_NOISE_DB
  UE3_TX_ADDR / UE3_RX_BIND / UE3_DIST_KM / UE3_DOPPLER_HZ / UE3_NOISE_DB
"""

import os, time, threading, logging
import numpy as np
import zmq

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("multi-ue-emulator")

# ── Configuration globale ────────────────────────────────────────────────────
GNB_TX_ADDR = os.getenv("GNB_TX_ADDR", "tcp://10.10.3.234:2000")
GNB_RX_BIND = os.getenv("GNB_RX_BIND", "tcp://0.0.0.0:2001")
N_SAMPLES   = int(os.getenv("N_SAMPLES", "23040"))
FC_GHZ      = float(os.getenv("CARRIER_FREQ_GHZ", "1.8"))

# Configuration par UE : (tx_addr, rx_bind, dist_km, doppler_hz, snr_db)
UE_CONFIGS = [
    {   # UE1 — eMBB : stationnaire, proche
        "name":       "embb",
        "tx_addr":    os.getenv("UE1_TX_ADDR",    "tcp://10.10.3.235:2101"),
        "rx_bind":    os.getenv("UE1_RX_BIND",    "tcp://0.0.0.0:2011"),
        "dist_km":    float(os.getenv("UE1_DIST_KM",    "0.5")),
        "doppler_hz": float(os.getenv("UE1_DOPPLER_HZ", "0.0")),
        "snr_db":     float(os.getenv("UE1_SNR_DB",     "30.0")),
    },
    {   # UE2 — URLLC : mobile 40 km/h, distance moyenne
        "name":       "urllc",
        "tx_addr":    os.getenv("UE2_TX_ADDR",    "tcp://10.10.3.236:2103"),
        "rx_bind":    os.getenv("UE2_RX_BIND",    "tcp://0.0.0.0:2013"),
        "dist_km":    float(os.getenv("UE2_DIST_KM",    "1.0")),
        "doppler_hz": float(os.getenv("UE2_DOPPLER_HZ", "67.0")),
        "snr_db":     float(os.getenv("UE2_SNR_DB",     "20.0")),
    },
    {   # UE3 — mMTC : IoT stationnaire, loin
        "name":       "mmtc",
        "tx_addr":    os.getenv("UE3_TX_ADDR",    "tcp://10.10.3.237:2105"),
        "rx_bind":    os.getenv("UE3_RX_BIND",    "tcp://0.0.0.0:2015"),
        "dist_km":    float(os.getenv("UE3_DIST_KM",    "1.5")),
        "doppler_hz": float(os.getenv("UE3_DOPPLER_HZ", "0.0")),
        "snr_db":     float(os.getenv("UE3_SNR_DB",     "15.0")),
    },
]


# ── Buffer IQ thread-safe ────────────────────────────────────────────────────
class IQBuffer:
    """Stocke le dernier bloc IQ ; renvoie des zéros si rien n'est arrivé."""
    def __init__(self, n):
        self._cond = threading.Condition()
        self._buf  = np.zeros(n, dtype=np.complex64)
        self._seq  = 0

    def put(self, arr):
        with self._cond:
            n = min(len(arr), len(self._buf))
            self._buf[:n] = arr[:n]
            self._seq += 1
            self._cond.notify_all()

    def get(self):
        with self._cond:
            return self._buf.copy()

    def wait_new(self, since_seq, timeout=0.05):
        """Block until a frame newer than since_seq is available (or timeout)."""
        with self._cond:
            self._cond.wait_for(lambda: self._seq > since_seq, timeout=timeout)
            return self._buf.copy(), self._seq


# ── Modèle de canal ──────────────────────────────────────────────────────────
def _apply_channel(sig, dist_km, doppler_hz, snr_db, sr=23.04e6):
    """Fading mono-trajet + Doppler + AWGN sur signal ZMQ normalisé.

    Pas de FSPL absolue : les samples ZMQ sont en bande de base normalisée.
    L'atténuation relative est modélisée via l'amplitude du fading.
    snr_db est le SNR cible en dB (positif = signal > bruit).
    """
    n = len(sig)

    # Fading : amplitude décroît doucement avec la distance
    fade_amp = 1.0 / (1.0 + dist_km)   # eMBB≈0.67, URLLC≈0.5, mMTC≈0.4
    fade     = np.complex64(complex(fade_amp * 0.85, fade_amp * 0.25))
    out      = sig * fade

    # Doppler (rotation de phase linéaire)
    if abs(doppler_hz) > 0.1:
        t   = np.arange(n, dtype=np.float32) / sr
        out = out * np.exp(1j * 2 * np.pi * doppler_hz * t).astype(np.complex64)

    # AWGN calibré sur la puissance du signal après fading
    snr_lin   = 10 ** (snr_db / 10.0)
    sig_power = float(np.mean(np.abs(out) ** 2)) or 1e-10
    noise_std = np.sqrt(sig_power / (2.0 * snr_lin))
    noise     = (np.random.randn(n) + 1j * np.random.randn(n)).astype(np.complex64)
    out       = out + noise * noise_std

    return out.astype(np.complex64)


# ── Threads ───────────────────────────────────────────────────────────────────
def _make_req(ctx, addr, label, timeout_ms=2000):
    """Crée un socket REQ avec timeout et LINGER=0."""
    s = ctx.socket(zmq.REQ)
    s.setsockopt(zmq.LINGER, 0)
    s.setsockopt(zmq.RCVTIMEO, timeout_ms)
    s.setsockopt(zmq.SNDTIMEO, timeout_ms)
    s.connect(addr)
    log.info(f"[{label}] REQ → {addr}")
    return s


def dl_poller(gnb_tx_addr, dl_buf):
    """Tire des IQ depuis le TX REP du gNB → remplit dl_buf partagé.
    Reconnecte automatiquement si le gNB redémarre (timeout ZMQ REQ/REP).
    """
    ctx = zmq.Context()
    req = _make_req(ctx, gnb_tx_addr, "DL/poller")
    while True:
        try:
            req.send(bytes([0]))
            raw = req.recv()
            iq  = np.frombuffer(raw, dtype=np.complex64).copy() if raw else np.zeros(N_SAMPLES, dtype=np.complex64)
            if len(iq) == 0:
                iq = np.zeros(N_SAMPLES, dtype=np.complex64)
            dl_buf.put(iq)
        except zmq.Again:
            log.warning("[DL/poller] timeout — reconnexion gNB")
            req.close()
            time.sleep(0.5)
            req = _make_req(ctx, gnb_tx_addr, "DL/poller")
        except Exception as exc:
            log.error(f"[DL/poller] {exc}")
            req.close()
            time.sleep(0.5)
            req = _make_req(ctx, gnb_tx_addr, "DL/poller")


def ue_dl_server(ue_cfg, dl_buf):
    """Sert le signal DL (avec effets canal de cet UE) à l'UE qui requête."""
    ctx  = zmq.Context()
    rep  = ctx.socket(zmq.REP)
    rep.setsockopt(zmq.LINGER, 0)
    rep.bind(ue_cfg["rx_bind"])
    name = ue_cfg["name"]
    log.info(f"[DL/{name}] REP bind {ue_cfg['rx_bind']} "
             f"(d={ue_cfg['dist_km']}km, dop={ue_cfg['doppler_hz']}Hz, NF={ue_cfg['snr_db']}dB)")
    count    = 0
    t0       = time.time()
    last_seq = 0
    while True:
        try:
            rep.recv()
            count += 1
            if count % 2000 == 0:
                rate = count / (time.time() - t0)
                log.info(f"[DL/{name}] {count} reqs  {rate:.0f}/s")
            # Block until a new frame from gNB is available — paces UE to gNB rate
            iq, last_seq = dl_buf.wait_new(last_seq)
            iq_out = _apply_channel(iq,
                                    ue_cfg["dist_km"],
                                    ue_cfg["doppler_hz"],
                                    ue_cfg["snr_db"])
            rep.send(iq_out.tobytes())
        except Exception as exc:
            log.error(f"[DL/{name}] {exc}")
            time.sleep(0.001)


def ue_ul_poller(ue_cfg, ul_buf):
    """Tire des IQ depuis le TX REP de l'UE → remplit ul_buf de cet UE.
    Reconnecte si l'UE redémarre.
    """
    ctx  = zmq.Context()
    name = ue_cfg["name"]
    req  = _make_req(ctx, ue_cfg["tx_addr"], f"UL/{name}")
    while True:
        try:
            req.send(bytes([0]))
            raw = req.recv()
            iq  = np.frombuffer(raw, dtype=np.complex64).copy() if raw else np.zeros(N_SAMPLES, dtype=np.complex64)
            if len(iq) == 0:
                iq = np.zeros(N_SAMPLES, dtype=np.complex64)
            iq_ch = _apply_channel(iq,
                                   ue_cfg["dist_km"],
                                   -ue_cfg["doppler_hz"],
                                   ue_cfg["snr_db"])
            ul_buf.put(iq_ch)
        except zmq.Again:
            log.warning(f"[UL/{name}] timeout — reconnexion UE")
            req.close()
            time.sleep(0.5)
            req = _make_req(ctx, ue_cfg["tx_addr"], f"UL/{name}")
        except Exception as exc:
            log.error(f"[UL/{name}] {exc}")
            req.close()
            time.sleep(0.5)
            req = _make_req(ctx, ue_cfg["tx_addr"], f"UL/{name}")


SLOT_S = 10e-3  # 10ms per NR frame — matches UE DL rate (~95/s) to prevent SFN divergence

def gnb_ul_server(gnb_rx_bind, ul_bufs):
    """Combine les UL de tous les UEs (OFDM : somme) → sert au gNB."""
    ctx = zmq.Context()
    rep = ctx.socket(zmq.REP)
    rep.setsockopt(zmq.LINGER, 0)
    rep.bind(gnb_rx_bind)
    log.info(f"[UL/gNB] REP bind {gnb_rx_bind}")
    count = 0
    t0    = time.time()
    while True:
        try:
            t_slot = time.time()
            rep.recv()
            count += 1
            if count % 2000 == 0:
                rate = count / (time.time() - t0)
                log.info(f"[UL/gNB] {count} reqs  {rate:.0f}/s")
            combined = np.zeros(N_SAMPLES, dtype=np.complex64)
            for buf in ul_bufs:
                combined += buf.get()
            elapsed = time.time() - t_slot
            remaining = SLOT_S - elapsed
            if remaining > 0:
                time.sleep(remaining)
            rep.send(combined.tobytes())
        except Exception as exc:
            log.error(f"[UL/gNB] {exc}")
            time.sleep(0.001)


if __name__ == "__main__":
    log.info("=== Multi-UE Channel Emulator ===")
    log.info(f"gNB TX : {GNB_TX_ADDR}")
    log.info(f"gNB RX : {GNB_RX_BIND}")
    for cfg in UE_CONFIGS:
        log.info(f"  UE [{cfg['name']}] tx={cfg['tx_addr']} dl={cfg['rx_bind']} "
                 f"d={cfg['dist_km']}km dop={cfg['doppler_hz']}Hz NF={cfg['snr_db']}dB")

    dl_buf  = IQBuffer(N_SAMPLES)
    ul_bufs = [IQBuffer(N_SAMPLES) for _ in UE_CONFIGS]

    threads = []

    # DL poller (gNB → shared buffer)
    threads.append(threading.Thread(
        target=dl_poller, args=(GNB_TX_ADDR, dl_buf), daemon=True,
        name="dl-poller"))

    # Per-UE DL servers + UL pollers
    for i, cfg in enumerate(UE_CONFIGS):
        threads.append(threading.Thread(
            target=ue_dl_server, args=(cfg, dl_buf), daemon=True,
            name=f"dl-{cfg['name']}"))
        threads.append(threading.Thread(
            target=ue_ul_poller, args=(cfg, ul_bufs[i]), daemon=True,
            name=f"ul-{cfg['name']}"))

    # gNB UL server (combined UL → gNB)
    threads.append(threading.Thread(
        target=gnb_ul_server, args=(GNB_RX_BIND, ul_bufs), daemon=True,
        name="ul-gnb"))

    for t in threads:
        t.start()
    log.info(f"{len(threads)} threads démarrés")
    for t in threads:
        t.join()

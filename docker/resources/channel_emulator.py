#!/usr/bin/env python3
"""
Channel Emulator — conforme papier REAL (arXiv:2502.00715), section III-B.

Modèle de canal appliqué entre gNB et UEs via ZMQ :
  1. FSPL  — Free-Space Path Loss  (dépend de la distance d)
  2. Multipath — single-tap (délai + phase aléatoires)
  3. AWGN  — bruit thermique (Noise Figure configurable)
  4. Doppler — décalage fréquentiel (mobilité UE)

Distance d ∈ [DISTANCE_MIN_KM, DISTANCE_MAX_KM] tirée aléatoirement
à chaque reset (simule mobilité en environnement urbain).

Topologie ZMQ :
  gNB TX ──[canal]──▶ UE RX
  UE  TX ──[canal]──▶ gNB RX
"""

import os
import time
import logging
import numpy as np
import zmq

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("channel-emulator")

# ── Configuration ────────────────────────────────────────────────────────────
GNB_TX = os.getenv("GNB_ZMQ_TX_ADDR", "tcp://srsran-gnb:2000")
GNB_RX = os.getenv("GNB_ZMQ_RX_ADDR", "tcp://srsran-gnb:2001")
UE_TX  = os.getenv("UE_ZMQ_TX_ADDR",  "tcp://srsue:2100")
UE_RX  = os.getenv("UE_ZMQ_RX_ADDR",  "tcp://srsue:2101")

D_MIN = float(os.getenv("DISTANCE_MIN_KM", "0.5"))   # km
D_MAX = float(os.getenv("DISTANCE_MAX_KM", "2.0"))   # km
DOPPLER_MAX = float(os.getenv("DOPPLER_MAX_HZ", "50"))  # Hz
NF_DB = float(os.getenv("NOISE_FIGURE_DB", "7"))     # dB
FC_GHZ = float(os.getenv("CARRIER_FREQ_GHZ", "1.8")) # GHz (bande 3)
RESET_INTERVAL_S = float(os.getenv("RESET_INTERVAL_S", "5.0"))

# ── Modèles de canal ─────────────────────────────────────────────────────────

def fspl_linear(distance_km: float, freq_ghz: float) -> float:
    """
    Free-Space Path Loss (linéaire, facteur d'atténuation < 1).
    FSPL(dB) = 20·log10(d) + 20·log10(f) + 92.45
    où d en km, f en GHz.
    """
    fspl_db = 20 * np.log10(distance_km) + 20 * np.log10(freq_ghz) + 92.45
    return 10 ** (-fspl_db / 20.0)


def awgn_noise(n_samples: int, snr_db: float) -> np.ndarray:
    """Bruit AWGN complexe pour un SNR donné."""
    noise_power = 10 ** (-snr_db / 10.0)
    return (np.random.randn(n_samples) +
            1j * np.random.randn(n_samples)) * np.sqrt(noise_power / 2)


def apply_channel(signal: np.ndarray,
                  distance_km: float,
                  doppler_hz: float,
                  sample_rate: float = 23.04e6) -> np.ndarray:
    """
    Applique le modèle de canal complet au signal IQ :
      1. FSPL
      2. Single-tap multipath (délai 0-5 échantillons, phase aléatoire)
      3. Doppler
      4. AWGN (NF = NOISE_FIGURE_DB)
    """
    n = len(signal)

    # 1. FSPL
    atten = fspl_linear(distance_km, FC_GHZ)
    out = signal * atten

    # 2. Single-tap multipath
    tap_delay = np.random.randint(0, 6)       # délai en échantillons
    tap_phase = np.random.uniform(0, 2 * np.pi)
    tap_gain  = np.random.uniform(0.3, 0.7)   # gain du tap secondaire
    if tap_delay > 0 and n > tap_delay:
        out[tap_delay:] += tap_gain * np.exp(1j * tap_phase) * signal[:-tap_delay]

    # 3. Doppler (rotation de phase progressive)
    t = np.arange(n) / sample_rate
    out *= np.exp(1j * 2 * np.pi * doppler_hz * t)

    # 4. AWGN
    # SNR estimé depuis FSPL (référence : NF + marge de 20dB)
    snr_db = -20 * np.log10(max(atten, 1e-10)) - NF_DB
    out += awgn_noise(n, snr_db)

    return out


# ── Boucle principale ZMQ ────────────────────────────────────────────────────

def run():
    ctx = zmq.Context()

    # Réception depuis gNB (TX du gNB)
    gnb_pull = ctx.socket(zmq.PULL)
    gnb_pull.connect(GNB_TX)

    # Envoi vers UE (RX du UE) — UE bind son PULL sur UE_RX, on s'y connecte
    ue_push = ctx.socket(zmq.PUSH)
    ue_push.connect(UE_RX)

    # Réception depuis UE (TX du UE)
    ue_pull = ctx.socket(zmq.PULL)
    ue_pull.connect(UE_TX)

    # Envoi vers gNB (RX du gNB) — gNB bind son PULL sur GNB_RX, on s'y connecte
    gnb_push = ctx.socket(zmq.PUSH)
    gnb_push.connect(GNB_RX)

    log.info(f"Canal initialisé. d∈[{D_MIN},{D_MAX}]km "
             f"Doppler_max={DOPPLER_MAX}Hz NF={NF_DB}dB")

    # Paramètres canal initiaux
    d = np.random.uniform(D_MIN, D_MAX)
    doppler = np.random.uniform(-DOPPLER_MAX, DOPPLER_MAX)
    last_reset = time.time()

    poller = zmq.Poller()
    poller.register(gnb_pull, zmq.POLLIN)
    poller.register(ue_pull,  zmq.POLLIN)

    while True:
        # Reset périodique des paramètres canal (mobilité)
        now = time.time()
        if now - last_reset > RESET_INTERVAL_S:
            d = np.random.uniform(D_MIN, D_MAX)
            doppler = np.random.uniform(-DOPPLER_MAX, DOPPLER_MAX)
            last_reset = now
            log.info(f"Canal reset → d={d:.2f}km doppler={doppler:.1f}Hz")

        events = dict(poller.poll(timeout=100))

        # gNB → UE (downlink)
        if gnb_pull in events:
            raw = gnb_pull.recv()
            iq = np.frombuffer(raw, dtype=np.complex64).copy()
            iq_out = apply_channel(iq, d, doppler).astype(np.complex64)
            ue_push.send(iq_out.tobytes())

        # UE → gNB (uplink)
        if ue_pull in events:
            raw = ue_pull.recv()
            iq = np.frombuffer(raw, dtype=np.complex64).copy()
            iq_out = apply_channel(iq, d, -doppler).astype(np.complex64)
            gnb_push.send(iq_out.tobytes())


if __name__ == "__main__":
    run()

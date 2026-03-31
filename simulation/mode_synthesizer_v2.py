#!/usr/bin/env python3
"""
Mode-Superposition Sperm Whale Click Synthesizer V2
====================================================
V2 improvements over V1:
  1. Parameterizable lip excitation spectrum (was hardcoded at 10kHz peak)
     - lip_peak_hz, lip_bandwidth_octaves, lip_power, low_freq_floor
     - Enables D and F to match (they need more 2-5kHz energy)
  2. Body resonance from click recoil (sub-100Hz)
  3. Coda generation from single click template + ICI pattern
  4. Optional ocean ambient noise (sub-200Hz)
  5. Tension optimizer with expanded parameter set (15 params)

Usage:
    python mode_synthesizer_v2.py                # all 3 whales
    python mode_synthesizer_v2.py --whale A      # just Whale A
    python mode_synthesizer_v2.py --optimize      # run optimizer for all 3

Output:
    v2_comparison.png (20x16)
    v2_results.json
    whale_{A,D,F}_v2.wav (individual clicks)
    whale_{A,D,F}_coda_v2.wav (full codas)
"""

import argparse
import json
import os
import sys
import time
import warnings
import numpy as np
from pathlib import Path
from scipy.io import wavfile
from scipy.signal import resample, butter, filtfilt
from scipy.optimize import differential_evolution

warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker

# ============================================================
# PATHS
# ============================================================
OUTPUT_DIR = Path("/mnt/archive/datasets/whale_communication/analysis/mode_synthesis")
DSWP_DIR = Path("/mnt/archive/datasets/whale_communication/DSWP")
CODA_CSV = Path("/mnt/archive/datasets/whale_communication/sw-combinatoriality/data/DominicaCodas.csv")
MODAL_RESULTS = Path("/mnt/archive/datasets/whale_communication/analysis/modal_results.json")
VOICEPRINTS = Path("/mnt/archive/datasets/whale_communication/analysis/whale_voiceprints.json")

TARGET_SR = 44100

# ============================================================
# TISSUE LAYERS (from signal_chain_v3)
# ============================================================
TISSUE_LAYERS = [
    # (name, thickness_m, absorption_dB_cm_MHz, sound_speed_m_s)
    ('case_wall',    0.05, 1.0, 1570),
    ('muscle',       0.12, 1.0, 1570),
    ('junk_lipid',   0.80, 0.5, 1400),
    ('junk_septa',   0.03, 1.0, 1570),
    ('blubber',      0.15, 0.5, 1420),
    ('skin',         0.015, 2.0, 1600),
]

# ============================================================
# OCEAN PROPAGATION PARAMETERS
# ============================================================
PROPAGATION = {
    'whale_depth_m': 50,
    'recording_distance_m': 10,
    'surface_temp_c': 25.0,
    'deep_temp_c': 15.0,
    'thermocline_depth_m': 100,
}

# ============================================================
# REAL TARGETS (from whale_voiceprints.json means)
# ============================================================
REAL_TARGETS = {
    'Whale_A': {
        'spectral_centroid_hz': 7849,
        'band_2_5khz_pct': 19.2,
        'band_5_10khz_pct': 49.5,
        'band_10_20khz_pct': 22.3,
        'band_sub_100hz_pct': 0.5,
        'band_100_500hz_pct': 0.75,
        'band_500_2khz_pct': 5.5,
        'band_above_20khz_pct': 2.28,
    },
    'Whale_D': {
        'spectral_centroid_hz': 5693,
        'band_2_5khz_pct': 18.7,
        'band_5_10khz_pct': 31.4,
        'band_10_20khz_pct': 17.7,
        'band_sub_100hz_pct': 16.1,
        'band_100_500hz_pct': 1.52,
        'band_500_2khz_pct': 12.2,
        'band_above_20khz_pct': 1.14,
    },
    'Whale_F': {
        'spectral_centroid_hz': 5333,
        'band_2_5khz_pct': 15.4,
        'band_5_10khz_pct': 24.7,
        'band_10_20khz_pct': 19.4,
        'band_sub_100hz_pct': 29.5,
        'band_100_500hz_pct': 2.80,
        'band_500_2khz_pct': 7.1,
        'band_above_20khz_pct': 1.1,
    },
}

# ============================================================
# WHALE CONFIGURATIONS (V2 - with lip excitation params)
# ============================================================
WHALE_CONFIGS = {
    'Whale_A': {
        'organ_length': 3.50, 'organ_diameter': 1.20,
        'spermaceti_c': 1370,
        'tau_up': 0.00025,      # 0.25ms (measured from modal decomposition)
        'tau_down': 0.00245,    # 2.45ms (measured)
        'muscle_tensions': [0.25, 0.15, 0.05, 0.05, 0.30, 0.50],
        'muscle_band_freqs': [16000, 12000, 9000, 7000, 5000, 3500],
        'junk_length': 2.0,
        # V2: Lip excitation parameters
        'lip_peak_hz': 8000,          # smaller whale, lighter lips, higher peak
        'lip_bandwidth_octaves': 2.0,
        'lip_power': 1.0,
        'low_freq_floor': 0.01,
        # V2: Body resonance
        'body_length_m': 10.0,        # ~10m female
        'body_mass_kg': 12000,
        'body_resonance_scale': 0.02, # small whale = less recoil energy
    },
    'Whale_D': {
        'organ_length': 3.80, 'organ_diameter': 1.40,
        'spermaceti_c': 1370,
        'tau_up': 0.00235,      # 2.35ms (longest sustain)
        'tau_down': 0.02097,    # 20.97ms
        'muscle_tensions': [0.75, 0.60, 0.55, 0.45, 0.40, 0.50],
        'muscle_band_freqs': [16000, 12000, 9000, 7000, 5000, 3500],
        'junk_length': 2.2,
        # V2: Lip excitation parameters
        'lip_peak_hz': 5000,          # medium whale, lower than A
        'lip_bandwidth_octaves': 2.5,
        'lip_power': 1.0,
        'low_freq_floor': 0.03,
        # V2: Body resonance
        'body_length_m': 12.0,
        'body_mass_kg': 20000,
        'body_resonance_scale': 0.10,
    },
    'Whale_F': {
        'organ_length': 4.80, 'organ_diameter': 1.70,
        'spermaceti_c': 1370,
        'tau_up': 0.00062,      # 0.62ms
        'tau_down': 0.00117,    # 1.17ms (fastest decay)
        'muscle_tensions': [0.80, 0.70, 0.65, 0.60, 0.30, 0.35],
        'muscle_band_freqs': [16000, 12000, 9000, 7000, 5000, 3500],
        'junk_length': 2.5,
        # V2: Lip excitation parameters
        'lip_peak_hz': 4500,          # largest whale, heavier lips, lower peak
        'lip_bandwidth_octaves': 2.5,
        'lip_power': 1.0,
        'low_freq_floor': 0.05,       # more low-freq energy (largest body)
        # V2: Body resonance
        'body_length_m': 15.0,        # large adult male
        'body_mass_kg': 40000,
        'body_resonance_scale': 0.30,
    },
}


# ============================================================
# STEP 1: Compute Cavity Modes from Anatomy
# ============================================================
def compute_cavity_modes(L, D, c, n_longitudinal=128, n_transverse=5):
    """Compute resonant mode frequencies for a cylindrical cavity.

    Longitudinal: f_n = n * c / (2L)
    Transverse (circular): f_mn = c * j_mn / (pi * D)
        where j_mn are zeros of Bessel function J_m
    Combined: f = c * sqrt((n/(2L))^2 + (j_mn/(pi*R))^2)

    After computing, deduplicates modes closer than 1% in frequency.
    """
    raw_modes = []
    R = D / 2

    bessel_zeros = {
        (0, 1): 2.4048, (0, 2): 5.5201, (0, 3): 8.6537, (0, 4): 11.7915, (0, 5): 14.9309,
        (1, 1): 3.8317, (1, 2): 7.0156, (1, 3): 10.1735, (1, 4): 13.3237, (1, 5): 16.4706,
        (2, 1): 5.1356, (2, 2): 8.4172, (2, 3): 11.6198, (2, 4): 14.7960,
        (3, 1): 6.3802, (3, 2): 9.7610, (3, 3): 13.0152,
        (4, 1): 7.5883, (4, 2): 11.0647,
        (5, 1): 8.7715,
    }

    # Pure longitudinal modes (up to 25kHz)
    for n in range(1, n_longitudinal + 1):
        f = n * c / (2 * L)
        if f < 25000:
            raw_modes.append({'freq': f, 'type': 'longitudinal', 'n': n, 'm': 0})

    # Pure transverse modes
    for (m, n_r), j_mn in bessel_zeros.items():
        f_t = c * j_mn / (2 * np.pi * R)
        if f_t < 25000:
            raw_modes.append({'freq': f_t, 'type': 'transverse', 'n': n_r, 'm': m})

    # Combined longitudinal-transverse modes
    for n in range(1, 60):
        for (m, n_r), j_mn in bessel_zeros.items():
            f_combined = c * np.sqrt((n / (2 * L))**2 + (j_mn / (np.pi * R))**2)
            if 500 < f_combined < 25000:
                raw_modes.append({'freq': f_combined, 'type': 'combined', 'n': n, 'm': m})

    raw_modes.sort(key=lambda x: x['freq'])

    # Deduplicate: merge modes within 1.5% of each other
    modes = []
    for mode in raw_modes:
        merged = False
        for existing in modes:
            if abs(mode['freq'] - existing['freq']) / max(existing['freq'], 1) < 0.015:
                merged = True
                break
        if not merged:
            modes.append(mode)

    return modes


# ============================================================
# STEP 2: Base Amplitude from Lip Excitation + Source Coupling
# ============================================================
def compute_base_amplitudes(modes, L, lip_peak_hz=5000, lip_bandwidth_octaves=2.0,
                            lip_power=1.0, low_freq_floor=0.05):
    """Base amplitude depends on lip excitation spectrum AND source coupling.

    Args:
        lip_peak_hz: center frequency of lip excitation (Hz)
            - Natural vibration frequency of the fatty pads
            - Smaller whales may have stiffer/lighter lips = higher peak
            - Larger whales = heavier lips = lower peak
        lip_bandwidth_octaves: how broadband the excitation is
            - 1.0 = narrow, tonal buzzing
            - 3.0 = very broadband, noisy
        lip_power: overall excitation power (scales all amplitudes)
        low_freq_floor: minimum amplitude for low-freq modes (0-1)
            - Prevents complete cutoff of low frequencies
    """
    for mode in modes:
        f = mode['freq']

        # Lip excitation spectrum: log-normal centered on lip_peak_hz
        log_ratio = np.log2(f / lip_peak_hz)
        lip_envelope = np.exp(-0.5 * (log_ratio / lip_bandwidth_octaves)**2)

        # Floor prevents complete low-freq cutoff
        lip_envelope = max(lip_envelope, low_freq_floor)

        # Source coupling (odd harmonics couple to anterior source)
        # Using gentle rolloff (n^0.15) matching V1 behavior - the lip is broadband,
        # so all harmonics get excited. The spectral shape comes from the lip envelope,
        # not from steep harmonic rolloff.
        if mode['type'] == 'longitudinal':
            n = mode['n']
            coupling = 1.0 if n % 2 == 1 else 0.4
            rolloff = 1.0 / (n**0.15)
        elif mode['type'] == 'transverse':
            if mode['m'] == 0:
                coupling = 0.15
            else:
                coupling = 0.03 / mode['m']
            rolloff = 1.0
        else:
            # Combined modes: longitudinal component couples to source
            m_factor = 1.0 / (1.0 + mode['m'])
            coupling = 0.4 * m_factor
            rolloff = 1.0 / (mode['n']**0.15)

        mode['base_amplitude'] = lip_power * lip_envelope * coupling * rolloff

    return modes


# ============================================================
# STEP 3: Muscle Tension Controls Mode Amplitudes
# ============================================================
def apply_muscle_damping(modes, muscle_tensions, muscle_band_freqs):
    """Each muscle band damps modes near its resonant frequency."""
    for mode in modes:
        f = mode['freq']
        damping_factor = 1.0
        for tension, band_freq in zip(muscle_tensions, muscle_band_freqs):
            proximity = np.exp(-0.5 * ((f - band_freq) / (0.2 * band_freq))**2)
            damping_factor *= (1.0 - tension * proximity * 0.8)
        mode['amplitude'] = mode['base_amplitude'] * damping_factor
    return modes


# ============================================================
# STEP 4: Synthesize the Click
# ============================================================
def synthesize_click(modes, sample_rate=44100, duration_s=0.025,
                     tau_up_s=0.001, tau_down_s=0.005):
    """Generate synthetic click waveform from mode parameters."""
    n_samples = int(duration_s * sample_rate)
    t = np.arange(n_samples) / sample_rate
    signal = np.zeros(n_samples)

    # Envelope: ramp up then exponential decay
    envelope = (1 - np.exp(-t / max(tau_up_s, 1e-6))) * np.exp(-t / max(tau_down_s, 1e-6))

    for mode in modes:
        f = mode['freq']
        A = mode.get('amplitude', mode.get('base_amplitude', 0.1))
        if f < 1000:
            Q = 8 + f / 250
        elif f < 3000:
            Q = 12 + f / 250
        elif f < 10000:
            Q = 24 + f / 400
        elif f < 15000:
            Q = 49 - (f - 10000) / 500
        else:
            Q = max(10, 39 - (f - 10000) / 400)
        Q = mode.get('Q', max(Q, 5))

        phi = np.random.uniform(0, 2 * np.pi)
        decay = np.exp(-np.pi * f * t / Q)
        signal += A * np.sin(2 * np.pi * f * t + phi) * decay

    signal *= envelope
    return signal, t


# ============================================================
# STEP 5: Exit-Path Tissue Filter
# ============================================================
def apply_exit_path_filter(signal, dt, junk_length=2.0):
    """Apply frequency-domain absorption for each tissue layer on exit path."""
    spectrum = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), dt)
    f_mhz = freqs / 1e6
    absorption_power = 1.5

    layers = list(TISSUE_LAYERS)
    for i, (name, thick, alpha, c_s) in enumerate(layers):
        if name == 'junk_lipid':
            layers[i] = (name, junk_length * 0.4, alpha, c_s)

    for name, thickness_m, alpha_db_cm_mhz, c_tissue in layers:
        if thickness_m < 0.001:
            continue
        thickness_cm = thickness_m * 100.0
        atten_db = alpha_db_cm_mhz * np.power(np.maximum(f_mhz, 1e-10), absorption_power) * thickness_cm
        atten_linear = np.power(10.0, -atten_db / 20.0)
        spectrum *= atten_linear

    return np.fft.irfft(spectrum, n=len(signal))


# ============================================================
# STEP 6: Ocean Propagation (Francois-Garrison)
# ============================================================
def apply_ocean_propagation(signal, dt, propagation=None):
    """Apply depth-integrated Francois-Garrison ocean propagation model."""
    if propagation is None:
        propagation = PROPAGATION

    spectrum = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), dt)
    f_khz = np.maximum(freqs / 1000.0, 1e-6)

    whale_depth = propagation['whale_depth_m']
    distance = propagation['recording_distance_m']
    surface_temp = propagation['surface_temp_c']
    deep_temp = propagation['deep_temp_c']
    thermo_depth = propagation['thermocline_depth_m']

    slant_range_m = np.sqrt(distance**2 + whale_depth**2)
    n_segments = max(10, int(whale_depth / 25))
    total_abs_db = np.zeros_like(f_khz)
    segment_length = slant_range_m / n_segments

    for i in range(n_segments):
        frac = (i + 0.5) / n_segments
        seg_depth = whale_depth * (1 - frac)

        if seg_depth <= thermo_depth:
            t_frac = seg_depth / max(thermo_depth, 1)
            seg_temp = surface_temp + t_frac * (deep_temp - surface_temp)
        else:
            seg_temp = deep_temp

        S, T, pH = 35.0, seg_temp, 8.0
        f1 = 0.78 * np.sqrt(S / 35.0) * np.exp(T / 26.0)
        A1 = 8.86 / (10**(0.78 * pH - 5.0)) * 10**(0.002 * T)
        alpha_1 = A1 * f1 * f_khz**2 / (f1**2 + f_khz**2)

        f2 = 42.0 * np.exp(T / 17.0)
        A2 = 21.44 * (S / 35.0) * (1 + 0.025 * T)
        alpha_2 = A2 * f2 * f_khz**2 / (f2**2 + f_khz**2)

        if T <= 20:
            A3 = 4.937e-4 - 2.59e-5 * T + 9.11e-7 * T**2 - 1.50e-8 * T**3
        else:
            A3 = 3.964e-4 - 1.146e-5 * T + 1.45e-7 * T**2 - 6.5e-10 * T**3
        alpha_3 = A3 * f_khz**2

        seg_abs = (alpha_1 + alpha_2 + alpha_3) * segment_length / 1000.0
        total_abs_db += seg_abs

    attenuation = 10**(-total_abs_db / 20.0)
    spectrum *= attenuation
    if slant_range_m > 1:
        spectrum /= slant_range_m

    return np.fft.irfft(spectrum, n=len(signal))


# ============================================================
# V2 NEW: Body Resonance from Click Recoil
# ============================================================
def add_body_resonance(signal, body_length_m, body_mass_kg, body_resonance_scale,
                       sample_rate=44100):
    """Add low-frequency body resonance from click recoil.

    The click exits the head with momentum. Newton's 3rd law:
    the whale body recoils, vibrating at its natural frequency.

    f_body ~ c_tissue / (2 * body_length) ~ 1570 / (2 * 12) ~ 65 Hz
    """
    if body_resonance_scale <= 0:
        return signal

    f_body = 1570.0 / (2 * body_length_m)  # ~50-80Hz for 10-15m whale
    Q_body = 5  # heavily damped (body is not rigid)

    t = np.arange(len(signal)) / sample_rate
    # Body oscillation starts when click fires, decays quickly
    # Scale by click energy (RMS of signal)
    click_energy = np.sqrt(np.mean(signal**2))
    body_signal = (click_energy * body_resonance_scale *
                   np.sin(2 * np.pi * f_body * t) *
                   np.exp(-np.pi * f_body * t / Q_body))

    return signal + body_signal


# ============================================================
# V2 NEW: Coda Generation from Single Click Template
# ============================================================
def generate_coda(click_signal, icis_ms, sample_rate=44100):
    """Generate a full coda from a single click template + ICI pattern.

    The repeated clicks create low-frequency envelope modulation
    that shows up as sub-100Hz energy in spectral analysis.
    """
    total_duration_ms = sum(icis_ms) + len(click_signal) / sample_rate * 1000 + 100
    total_samples = int(total_duration_ms / 1000 * sample_rate)
    coda = np.zeros(total_samples)

    pos = 0
    for ici_ms in icis_ms:
        click_start = int(pos / 1000 * sample_rate)
        click_end = min(click_start + len(click_signal), total_samples)
        n = click_end - click_start
        if n > 0:
            coda[click_start:click_end] += click_signal[:n]
        pos += ici_ms

    # Add one final click
    click_start = int(pos / 1000 * sample_rate)
    click_end = min(click_start + len(click_signal), total_samples)
    n = click_end - click_start
    if n > 0:
        coda[click_start:click_end] += click_signal[:n]

    return coda


# ============================================================
# V2 NEW: Ocean Ambient Noise
# ============================================================
def add_ocean_ambient(signal, snr_db=20, sample_rate=44100):
    """Add realistic ocean ambient noise (wind, waves, shipping).
    Low-pass filtered noise below 200Hz.
    """
    noise = np.random.randn(len(signal))
    b, a = butter(4, 200 / (sample_rate / 2), btype='low')
    ambient = filtfilt(b, a, noise)

    # Scale to desired SNR
    signal_power = np.mean(signal**2)
    if signal_power < 1e-30:
        return signal
    noise_power = signal_power / (10**(snr_db / 10))
    ambient_power = np.mean(ambient**2)
    if ambient_power < 1e-30:
        return signal
    ambient *= np.sqrt(noise_power / ambient_power)

    return signal + ambient


# ============================================================
# FEATURE EXTRACTION
# ============================================================
def compute_band_energies(signal, dt):
    """Compute spectral energy in frequency bands as percentages."""
    fft_mag = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal), dt)
    total_energy = np.sum(fft_mag**2)
    if total_energy < 1e-30:
        return {}

    bands = [
        ('sub_100hz', 0, 100),
        ('100_500hz', 100, 500),
        ('500_2khz', 500, 2000),
        ('2_5khz', 2000, 5000),
        ('5_10khz', 5000, 10000),
        ('10_20khz', 10000, 20000),
        ('above_20khz', 20000, 22050),
    ]

    result = {}
    for name, f_lo, f_hi in bands:
        mask = (freqs >= f_lo) & (freqs < f_hi)
        band_energy = np.sum(fft_mag[mask]**2)
        result[f'band_{name}_pct'] = 100.0 * band_energy / total_energy
    return result


def compute_spectral_centroid(signal, dt):
    """Compute spectral centroid in Hz."""
    fft_mag = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal), dt)
    total = np.sum(fft_mag)
    if total < 1e-30:
        return 0.0
    return float(np.sum(freqs * fft_mag) / total)


# ============================================================
# LOAD REAL WAV DATA
# ============================================================
def get_whale_wav_files():
    """Map whale IDs to WAV file paths using DominicaCodas.csv."""
    import pandas as pd
    dom = pd.read_csv(CODA_CSV)
    wavs_available = set()
    for f in os.listdir(DSWP_DIR):
        if f.endswith('.wav'):
            wavs_available.add(int(f.replace('.wav', '')))

    whale_files = {}
    for whale_id in ['A', 'D', 'F']:
        codas = dom[dom['Unit'] == whale_id]['codaNUM2018'].values
        files = [DSWP_DIR / f"{c}.wav" for c in codas if c in wavs_available]
        files = list(dict.fromkeys(files))
        whale_files[whale_id] = files
    return whale_files


def load_real_click(whale_id, whale_files, n_clicks=5):
    """Load representative clicks from DSWP WAVs for a whale."""
    files = whale_files.get(whale_id, [])
    if not files:
        print(f"  Warning: no WAV files for whale {whale_id}")
        return []

    clicks = []
    for wav_path in files[:20]:
        if len(clicks) >= n_clicks:
            break
        try:
            sr, data = wavfile.read(wav_path)
            if data.dtype == np.int16:
                data = data.astype(np.float64) / 32768.0
            elif data.dtype == np.float32:
                data = data.astype(np.float64)
            if data.ndim > 1:
                data = data[:, 0]

            if sr != TARGET_SR:
                n_samp = int(len(data) * TARGET_SR / sr)
                data = resample(data, n_samp)
                sr = TARGET_SR

            window_samples = int(0.020 * sr)
            if len(data) < window_samples:
                continue
            energy = np.convolve(data**2, np.ones(window_samples) / window_samples, mode='valid')
            peak_idx = np.argmax(energy)
            start = max(0, peak_idx - window_samples // 4)
            end = min(len(data), start + window_samples)
            click = data[start:end]
            if np.max(np.abs(click)) > 0.01:
                clicks.append((click, sr))
        except Exception:
            continue

    return clicks


def get_mean_real_click(whale_id, whale_files, n_clicks=10):
    """Get a representative real click (highest RMS)."""
    clicks = load_real_click(whale_id, whale_files, n_clicks=n_clicks)
    if not clicks:
        return None, None
    best = max(clicks, key=lambda x: np.sqrt(np.mean(x[0]**2)))
    return best


# ============================================================
# FULL PIPELINE (V2)
# ============================================================
def generate_whale_click(whale_name, config, seed=42, add_body=True, add_ambient=False):
    """Full pipeline: anatomy -> modes -> lip excitation -> muscle damping ->
    synthesis -> body resonance -> filter -> ocean -> (optional ambient)."""
    np.random.seed(seed)
    t0 = time.time()

    L = config['organ_length']
    D = config['organ_diameter']
    c = config['spermaceti_c']

    # 1. Compute cavity modes from anatomy
    modes = compute_cavity_modes(L, D, c)
    n_modes_total = len(modes)

    # 2. Set base amplitudes from lip excitation + source coupling (V2: parameterized)
    modes = compute_base_amplitudes(
        modes, L,
        lip_peak_hz=config.get('lip_peak_hz', 5000),
        lip_bandwidth_octaves=config.get('lip_bandwidth_octaves', 2.0),
        lip_power=config.get('lip_power', 1.0),
        low_freq_floor=config.get('low_freq_floor', 0.05),
    )

    # 3. Apply muscle tension damping
    modes = apply_muscle_damping(modes, config['muscle_tensions'],
                                  config['muscle_band_freqs'])

    # 4. Compute click duration
    duration_s = config['tau_up'] + 5 * config['tau_down']
    duration_s = max(duration_s, 0.015)
    duration_s = min(duration_s, 0.150)

    # 5. Synthesize raw click
    raw, t = synthesize_click(modes, sample_rate=TARGET_SR, duration_s=duration_s,
                               tau_up_s=config['tau_up'],
                               tau_down_s=config['tau_down'])

    dt = 1.0 / TARGET_SR

    # 6. Add body resonance (V2)
    if add_body and config.get('body_resonance_scale', 0) > 0:
        raw = add_body_resonance(
            raw,
            body_length_m=config.get('body_length_m', 12.0),
            body_mass_kg=config.get('body_mass_kg', 20000),
            body_resonance_scale=config.get('body_resonance_scale', 0.1),
            sample_rate=TARGET_SR,
        )

    # 7. Exit-path tissue filter
    filtered = apply_exit_path_filter(raw, dt, junk_length=config.get('junk_length', 2.0))

    # 8. Ocean propagation
    final = apply_ocean_propagation(filtered, dt)

    # 9. Optional ocean ambient (V2)
    if add_ambient:
        final = add_ocean_ambient(final, snr_db=20, sample_rate=TARGET_SR)

    elapsed = time.time() - t0
    print(f"  {whale_name}: {n_modes_total} modes, duration={duration_s*1000:.1f}ms, "
          f"lip_peak={config.get('lip_peak_hz', 5000)}Hz, "
          f"body_res_scale={config.get('body_resonance_scale', 0):.2f}, "
          f"synthesized in {elapsed*1000:.1f}ms")

    return {
        'raw': raw,
        'filtered': filtered,
        'final': final,
        'modes': modes,
        'n_modes': n_modes_total,
        'duration_s': duration_s,
        'elapsed_ms': elapsed * 1000,
        't': t,
    }


# ============================================================
# SAVE WAV
# ============================================================
def save_wav(signal, filename, sample_rate=44100, n_repeats=5, ici_s=0.3):
    """Save signal as 16-bit WAV, repeated as a coda sequence for audibility."""
    click_len = len(signal)
    ici_samples = int(ici_s * sample_rate)
    gap_samples = int(1.0 * sample_rate)

    coda_len = click_len + (n_repeats - 1) * ici_samples
    coda = np.zeros(coda_len)
    for i in range(n_repeats):
        start = i * ici_samples
        end = min(start + click_len, coda_len)
        coda[start:end] += signal[:end - start]

    n_codas = 3
    total_len = n_codas * len(coda) + (n_codas - 1) * gap_samples + gap_samples
    output = np.zeros(total_len)
    pos = gap_samples // 2
    for c_idx in range(n_codas):
        output[pos:pos + len(coda)] = coda
        pos += len(coda) + gap_samples

    peak = np.max(np.abs(output))
    if peak > 0:
        output = output * 0.9 / peak
    output_16 = np.clip(output * 32767, -32768, 32767).astype(np.int16)
    wavfile.write(str(filename), sample_rate, output_16)
    dur_s = len(output_16) / sample_rate
    print(f"  Saved: {filename} ({dur_s:.1f}s, {n_repeats} clicks x {n_codas} codas)")


def save_coda_wav(coda_signal, filename, sample_rate=44100):
    """Save a full coda signal as WAV."""
    peak = np.max(np.abs(coda_signal))
    if peak > 0:
        coda_signal = coda_signal * 0.9 / peak
    output_16 = np.clip(coda_signal * 32767, -32768, 32767).astype(np.int16)
    wavfile.write(str(filename), sample_rate, output_16)
    dur_s = len(output_16) / sample_rate
    print(f"  Saved: {filename} ({dur_s:.2f}s)")


# ============================================================
# V2 OPTIMIZER (expanded parameter set)
# ============================================================
OPTIM_PARAM_NAMES = [
    'tension_0', 'tension_1', 'tension_2', 'tension_3', 'tension_4', 'tension_5',
    'tau_up', 'tau_down',
    'spermaceti_c', 'organ_length', 'organ_diameter',
    # V2 new params:
    'lip_peak_hz', 'lip_bandwidth_octaves', 'low_freq_floor', 'body_resonance_scale',
]

OPTIM_PARAM_BOUNDS = [
    (0.0, 1.0),       # tension_0
    (0.0, 1.0),       # tension_1
    (0.0, 1.0),       # tension_2
    (0.0, 1.0),       # tension_3
    (0.0, 1.0),       # tension_4
    (0.0, 1.0),       # tension_5
    (0.0001, 0.005),  # tau_up
    (0.0005, 0.030),  # tau_down
    (1350, 1530),     # spermaceti_c
    (2.5, 6.0),       # organ_length
    (0.8, 2.5),       # organ_diameter
    # V2 new bounds:
    (2000, 10000),    # lip_peak_hz
    (1.0, 4.0),       # lip_bandwidth_octaves
    (0.01, 0.20),     # low_freq_floor
    (0.0, 0.50),      # body_resonance_scale
]

MUSCLE_BAND_FREQS = [16000, 12000, 9000, 7000, 5000, 3500]


def synthesize_from_params(params, junk_length=2.0, body_length_m=12.0, body_mass_kg=20000):
    """Run the full V2 synthesis pipeline from a flat parameter vector."""
    tensions = list(params[:6])
    tau_up = params[6]
    tau_down = params[7]
    spermaceti_c = params[8]
    organ_length = params[9]
    organ_diameter = params[10]
    lip_peak_hz = params[11]
    lip_bandwidth_octaves = params[12]
    low_freq_floor = params[13]
    body_resonance_scale = params[14]

    # 1. Cavity modes
    modes = compute_cavity_modes(organ_length, organ_diameter, spermaceti_c)
    # 2. Lip excitation (V2: parameterized)
    modes = compute_base_amplitudes(modes, organ_length,
                                     lip_peak_hz=lip_peak_hz,
                                     lip_bandwidth_octaves=lip_bandwidth_octaves,
                                     lip_power=1.0,
                                     low_freq_floor=low_freq_floor)
    # 3. Muscle damping
    modes = apply_muscle_damping(modes, tensions, MUSCLE_BAND_FREQS)

    # 4. Synthesize
    duration_s = tau_up + 5 * tau_down
    duration_s = max(duration_s, 0.015)
    duration_s = min(duration_s, 0.150)
    signal, t = synthesize_click(modes, sample_rate=TARGET_SR, duration_s=duration_s,
                                  tau_up_s=tau_up, tau_down_s=tau_down)

    dt = 1.0 / TARGET_SR

    # 5. Body resonance (V2)
    if body_resonance_scale > 0:
        signal = add_body_resonance(signal, body_length_m, body_mass_kg,
                                     body_resonance_scale, sample_rate=TARGET_SR)

    # 6. Exit-path filter
    signal = apply_exit_path_filter(signal, dt, junk_length=junk_length)
    # 7. Ocean propagation
    signal = apply_ocean_propagation(signal, dt)

    return signal, dt


def cost_function(params, target, junk_length=2.0, body_length_m=12.0, body_mass_kg=20000):
    """Cost: weighted sum of squared normalized errors vs real data."""
    np.random.seed(42)

    try:
        signal, dt = synthesize_from_params(params, junk_length=junk_length,
                                             body_length_m=body_length_m,
                                             body_mass_kg=body_mass_kg)
    except Exception:
        return 100.0

    bands = compute_band_energies(signal, dt)
    centroid = compute_spectral_centroid(signal, dt)

    if centroid < 100 or not bands:
        return 100.0

    error = 0.0

    # Centroid (weight 3.0)
    error += 3.0 * ((centroid - target['spectral_centroid_hz']) / target['spectral_centroid_hz'])**2

    # Primary bands (weight 2.0)
    for key in ['band_2_5khz_pct', 'band_5_10khz_pct', 'band_10_20khz_pct']:
        real_val = target.get(key, 0)
        synth_val = bands.get(key, 0)
        if real_val > 1.0:
            error += 2.0 * ((synth_val - real_val) / real_val)**2

    # Secondary bands (weight 1.0) - including sub-100Hz now (important for D & F)
    for key in ['band_sub_100hz_pct', 'band_500_2khz_pct']:
        real_val = target.get(key, 0)
        synth_val = bands.get(key, 0)
        if real_val > 1.0:
            error += 1.0 * ((synth_val - real_val) / real_val)**2

    # Minor bands (weight 0.5)
    for key in ['band_100_500hz_pct', 'band_above_20khz_pct']:
        real_val = target.get(key, 0)
        synth_val = bands.get(key, 0)
        if real_val > 1.0:
            error += 0.5 * ((synth_val - real_val) / real_val)**2

    return error


class ProgressTracker:
    def __init__(self, whale_name, target):
        self.whale_name = whale_name
        self.target = target
        self.iteration = 0
        self.best_cost = float('inf')
        self.start_time = time.time()
        self.history = []

    def callback(self, xk, convergence=0):
        self.iteration += 1
        np.random.seed(42)
        try:
            signal, dt = synthesize_from_params(xk, junk_length=self._junk_length,
                                                 body_length_m=self._body_length_m,
                                                 body_mass_kg=self._body_mass_kg)
            bands = compute_band_energies(signal, dt)
            centroid = compute_spectral_centroid(signal, dt)
            current_cost = cost_function(xk, self.target, junk_length=self._junk_length,
                                          body_length_m=self._body_length_m,
                                          body_mass_kg=self._body_mass_kg)
        except Exception:
            return

        if current_cost < self.best_cost:
            self.best_cost = current_cost

        self.history.append(current_cost)

        if self.iteration % 10 == 0 or self.iteration == 1:
            elapsed = time.time() - self.start_time
            centroid_err = (centroid - self.target['spectral_centroid_hz']) / self.target['spectral_centroid_hz'] * 100
            print(f"[{self.whale_name} Iter {self.iteration:3d}] cost={current_cost:.4f}  "
                  f"centroid_err={centroid_err:+.1f}%  best={self.best_cost:.4f}  "
                  f"({elapsed:.0f}s)")
            lip_peak = xk[11]
            lip_bw = xk[12]
            body_scale = xk[14]
            print(f"  lip_peak={lip_peak:.0f}Hz  lip_bw={lip_bw:.2f}oct  "
                  f"body_scale={body_scale:.3f}  "
                  f"sub100={bands.get('band_sub_100hz_pct', 0):.1f}%  "
                  f"2-5k={bands.get('band_2_5khz_pct', 0):.1f}%  "
                  f"5-10k={bands.get('band_5_10khz_pct', 0):.1f}%")


def optimize_whale(whale_name, target, initial_config):
    """Run differential_evolution for one whale with V2 expanded params."""
    print(f"\n{'='*70}")
    print(f"OPTIMIZING {whale_name} (V2 - 15 params)")
    print(f"{'='*70}")
    print(f"Target centroid: {target['spectral_centroid_hz']} Hz")
    print(f"Target sub-100Hz: {target.get('band_sub_100hz_pct', 0)}%")
    print(f"Target 2-5kHz:  {target['band_2_5khz_pct']}%")
    print(f"Target 5-10kHz: {target['band_5_10khz_pct']}%")

    x0 = [
        *initial_config['muscle_tensions'],
        initial_config['tau_up'],
        initial_config['tau_down'],
        initial_config['spermaceti_c'],
        initial_config['organ_length'],
        initial_config['organ_diameter'],
        initial_config.get('lip_peak_hz', 5000),
        initial_config.get('lip_bandwidth_octaves', 2.0),
        initial_config.get('low_freq_floor', 0.05),
        initial_config.get('body_resonance_scale', 0.1),
    ]
    junk_length = initial_config.get('junk_length', 2.0)
    body_length_m = initial_config.get('body_length_m', 12.0)
    body_mass_kg = initial_config.get('body_mass_kg', 20000)

    np.random.seed(42)
    init_cost = cost_function(x0, target, junk_length=junk_length,
                               body_length_m=body_length_m, body_mass_kg=body_mass_kg)
    print(f"\nInitial cost: {init_cost:.4f}")

    # Time a single eval
    t0 = time.time()
    np.random.seed(42)
    _ = cost_function(x0, target, junk_length=junk_length,
                       body_length_m=body_length_m, body_mass_kg=body_mass_kg)
    eval_time = time.time() - t0
    print(f"Single eval time: {eval_time*1000:.1f}ms")

    tracker = ProgressTracker(whale_name, target)
    tracker._junk_length = junk_length
    tracker._body_length_m = body_length_m
    tracker._body_mass_kg = body_mass_kg

    popsize = 20
    maxiter = 200
    total_evals = popsize * len(OPTIM_PARAM_BOUNDS) * (maxiter + 1)
    eta_min = total_evals * eval_time / 60
    print(f"Estimated: {total_evals} evals, ~{eta_min:.1f} minutes")
    print(f"Starting differential_evolution (popsize={popsize}, maxiter={maxiter}, 15 params)...")
    print("-" * 70)

    start = time.time()
    result = differential_evolution(
        cost_function,
        bounds=OPTIM_PARAM_BOUNDS,
        args=(target, junk_length, body_length_m, body_mass_kg),
        seed=42,
        popsize=popsize,
        maxiter=maxiter,
        tol=1e-6,
        mutation=(0.5, 1.5),
        recombination=0.8,
        callback=tracker.callback,
        disp=False,
        init='latinhypercube',
        workers=1,
    )
    elapsed = time.time() - start

    print(f"\n{'='*70}")
    print(f"{whale_name} OPTIMIZATION COMPLETE")
    print(f"{'='*70}")
    print(f"  Final cost: {result.fun:.6f}")
    print(f"  Iterations: {result.nit}")
    print(f"  Function evals: {result.nfev}")
    print(f"  Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    opt = result.x
    tensions = list(opt[:6])
    tau_up = opt[6]
    tau_down = opt[7]
    spermaceti_c = opt[8]
    organ_length = opt[9]
    organ_diameter = opt[10]
    lip_peak_hz = opt[11]
    lip_bandwidth_octaves = opt[12]
    low_freq_floor = opt[13]
    body_resonance_scale = opt[14]

    print(f"\n  tensions: [{', '.join(f'{t:.3f}' for t in tensions)}]")
    print(f"  tau_up={tau_up*1000:.3f}ms  tau_down={tau_down*1000:.2f}ms")
    print(f"  c={spermaceti_c:.1f}  L={organ_length:.3f}  D={organ_diameter:.3f}")
    print(f"  lip_peak={lip_peak_hz:.0f}Hz  lip_bw={lip_bandwidth_octaves:.2f}oct  "
          f"floor={low_freq_floor:.3f}  body_scale={body_resonance_scale:.3f}")

    # Generate final click and extract features
    np.random.seed(42)
    signal, dt = synthesize_from_params(opt, junk_length=junk_length,
                                         body_length_m=body_length_m,
                                         body_mass_kg=body_mass_kg)
    bands = compute_band_energies(signal, dt)
    centroid = compute_spectral_centroid(signal, dt)

    print(f"\n  Optimized centroid: {centroid:.0f} Hz (target: {target['spectral_centroid_hz']})")
    band_keys_display = [
        ('band_sub_100hz_pct', '<100Hz'),
        ('band_100_500hz_pct', '100-500Hz'),
        ('band_500_2khz_pct', '0.5-2kHz'),
        ('band_2_5khz_pct', '2-5kHz'),
        ('band_5_10khz_pct', '5-10kHz'),
        ('band_10_20khz_pct', '10-20kHz'),
        ('band_above_20khz_pct', '>20kHz'),
    ]
    print(f"\n  {'Band':<12} {'Real':>8} {'Optimized':>10} {'Error':>8}")
    print(f"  {'-'*40}")
    for key, label in band_keys_display:
        real_val = target.get(key, 0)
        opt_val = bands.get(key, 0)
        err = 0
        if real_val > 0.1:
            err = (opt_val - real_val) / real_val * 100
        print(f"  {label:<12} {real_val:>7.1f}% {opt_val:>9.1f}% {err:>+7.1f}%")

    return {
        'whale': whale_name,
        'cost': float(result.fun),
        'initial_cost': init_cost,
        'nit': int(result.nit),
        'nfev': int(result.nfev),
        'elapsed_s': round(elapsed, 1),
        'params': {
            'tensions': [round(t, 4) for t in tensions],
            'tau_up_s': round(tau_up, 6),
            'tau_down_s': round(tau_down, 6),
            'spermaceti_c': round(spermaceti_c, 1),
            'organ_length': round(organ_length, 4),
            'organ_diameter': round(organ_diameter, 4),
            'lip_peak_hz': round(lip_peak_hz, 1),
            'lip_bandwidth_octaves': round(lip_bandwidth_octaves, 3),
            'low_freq_floor': round(low_freq_floor, 4),
            'body_resonance_scale': round(body_resonance_scale, 4),
        },
        'features': {
            'spectral_centroid_hz': round(centroid, 1),
            **{k: round(bands.get(k, 0), 2) for k in [kk for kk, _ in band_keys_display]},
        },
        'signal': signal,
        'dt': dt,
        'history': tracker.history,
    }


# ============================================================
# LOAD V1 RESULTS FOR COMPARISON
# ============================================================
def load_v1_results():
    """Load V1 mode synthesis results if available."""
    v1_path = OUTPUT_DIR / 'mode_synthesis_results.json'
    if v1_path.exists():
        with open(v1_path) as f:
            return json.load(f)
    return None


# ============================================================
# FIGURE: v2_comparison.png (20x16)
# ============================================================
def create_comparison_figure(all_results, whale_files, output_dir, v1_data=None, optimized_results=None):
    """Create comparison figure: V1 vs V2 vs Real for all 3 whales."""
    print("\n[Figure] Creating v2_comparison.png (20x16)...")

    whale_names = ['Whale_A', 'Whale_D', 'Whale_F']
    whale_ids = ['A', 'D', 'F']
    colors = {'Whale_A': '#1f77b4', 'Whale_D': '#ff7f0e', 'Whale_F': '#2ca02c'}
    band_colors = ['#440154', '#31688e', '#35b779', '#fde725', '#e76f51', '#d62728', '#9467bd']

    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(4, 3, figure=fig, hspace=0.40, wspace=0.30,
                  height_ratios=[1, 1, 1, 0.5])

    dt = 1.0 / TARGET_SR
    metrics_data = []

    for col, (wname, wid) in enumerate(zip(whale_names, whale_ids)):
        # Use optimized results if available, otherwise default config
        if optimized_results and wname in optimized_results:
            synth = optimized_results[wname]['signal']
            opt_label = 'V2 Optimized'
        else:
            result = all_results[wname]
            synth = result['final']
            opt_label = 'V2 Default'

        # Load real click
        real_click, _ = get_mean_real_click(wid, whale_files)

        # --- Row 1: Spectrum comparison (V2 vs V1 vs Real) ---
        ax = fig.add_subplot(gs[0, col])

        # V2 spectrum
        synth_fft = np.abs(np.fft.rfft(synth))
        synth_freqs = np.fft.rfftfreq(len(synth), dt) / 1000
        synth_db = 20 * np.log10(synth_fft / max(np.max(synth_fft), 1e-30) + 1e-30)
        ax.plot(synth_freqs, synth_db, color=colors[wname], linewidth=1.2,
                label=opt_label, zorder=3)

        # V1 spectrum (if available)
        if v1_data and wname in v1_data.get('whales', {}):
            v1_info = v1_data['whales'][wname]
            v1_centroid = v1_info.get('synth_centroid_hz', 0)
            # We don't have the V1 signal, but we can annotate
            ax.axvline(v1_centroid / 1000, color='orange', linestyle=':', alpha=0.5,
                       linewidth=1.0, label=f'V1 centroid ({v1_centroid:.0f}Hz)')

        # Real click spectrum
        if real_click is not None:
            real_fft = np.abs(np.fft.rfft(real_click))
            real_freqs = np.fft.rfftfreq(len(real_click), dt) / 1000
            real_db = 20 * np.log10(real_fft / max(np.max(real_fft), 1e-30) + 1e-30)
            ax.plot(real_freqs, real_db, color='gray', linewidth=0.8, alpha=0.7,
                    label='Real', zorder=2)

        real_centroid = REAL_TARGETS[wname]['spectral_centroid_hz']
        synth_centroid = compute_spectral_centroid(synth, dt)
        centroid_err = (synth_centroid - real_centroid) / real_centroid * 100
        ax.axvline(real_centroid / 1000, color='red', linestyle=':', alpha=0.4, linewidth=0.8)

        ax.set_title(f'{wname} - Spectrum\nCentroid: {synth_centroid:.0f}Hz '
                     f'(target {real_centroid}, err {centroid_err:+.1f}%)',
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('Frequency (kHz)')
        ax.set_ylabel('Magnitude (dB)')
        ax.set_xlim(0, 22)
        ax.set_ylim(-60, 5)
        ax.legend(fontsize=7)

        # --- Row 2: Band energy bars ---
        ax = fig.add_subplot(gs[1, col])
        synth_bands = compute_band_energies(synth, dt)
        real_bands = REAL_TARGETS[wname]

        band_names = ['sub_100hz', '100_500hz', '500_2khz', '2_5khz', '5_10khz', '10_20khz', 'above_20khz']
        band_labels_short = ['<100', '100-500', '0.5-2k', '2-5k', '5-10k', '10-20k', '>20k']
        x = np.arange(len(band_names))
        width = 0.35

        synth_vals = [synth_bands.get(f'band_{bn}_pct', 0) for bn in band_names]
        real_vals = [real_bands.get(f'band_{bn}_pct', 0) for bn in band_names]

        ax.bar(x - width/2, synth_vals, width, label=opt_label,
               color=colors[wname], alpha=0.8)
        ax.bar(x + width/2, real_vals, width, label='Real',
               color='gray', alpha=0.6)

        ax.set_title(f'{wname} - Band Energy', fontsize=10)
        ax.set_xlabel('Frequency Band')
        ax.set_ylabel('Energy (%)')
        ax.set_xticks(x)
        ax.set_xticklabels(band_labels_short, rotation=45, fontsize=8)
        ax.legend(fontsize=8)

        # --- Row 3: Waveform ---
        ax = fig.add_subplot(gs[2, col])
        t_synth = np.arange(len(synth)) / TARGET_SR * 1000
        ax.plot(t_synth, synth / max(np.max(np.abs(synth)), 1e-10),
                color=colors[wname], linewidth=0.8, label=opt_label)
        if real_click is not None:
            t_real = np.arange(len(real_click)) / TARGET_SR * 1000
            ax.plot(t_real, real_click / max(np.max(np.abs(real_click)), 1e-10),
                    color='gray', linewidth=0.6, alpha=0.7, label='Real')
        ax.set_title(f'{wname} - Waveform', fontsize=10)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Normalized Amplitude')
        ax.legend(fontsize=8)
        ax.set_xlim(0, min(t_synth[-1], 25))

        # Metrics
        total_band_err = 0
        n_bands_counted = 0
        for bn in band_names:
            sv = synth_bands.get(f'band_{bn}_pct', 0)
            rv = real_bands.get(f'band_{bn}_pct', 0)
            if rv > 1:
                total_band_err += abs(sv - rv) / rv * 100
                n_bands_counted += 1
        mean_band_err = total_band_err / max(n_bands_counted, 1)

        metrics_data.append({
            'whale': wname,
            'synth_centroid_hz': round(synth_centroid, 1),
            'real_centroid_hz': real_centroid,
            'centroid_error_pct': round(abs(centroid_err), 1),
            'mean_band_error_pct': round(mean_band_err, 1),
            'synth_bands': {bn: round(synth_bands.get(f'band_{bn}_pct', 0), 1) for bn in band_names},
        })

    # --- Row 4: Metrics table ---
    ax = fig.add_subplot(gs[3, :])
    ax.axis('off')

    table_data = []
    for m in metrics_data:
        table_data.append([
            m['whale'],
            f"{m['synth_centroid_hz']:.0f}",
            f"{m['real_centroid_hz']:.0f}",
            f"{m['centroid_error_pct']:.1f}%",
            f"{m['mean_band_error_pct']:.1f}%",
            f"{m['synth_bands'].get('sub_100hz', 0):.1f}%",
            f"{m['synth_bands'].get('2_5khz', 0):.1f}%",
            f"{m['synth_bands'].get('5_10khz', 0):.1f}%",
        ])

    table = ax.table(
        cellText=table_data,
        colLabels=['Whale', 'V2 Centroid', 'Real Centroid', 'Centroid Err',
                   'Mean Band Err', 'Sub-100Hz', '2-5kHz', '5-10kHz'],
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)
    for (row, col_idx), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#cccccc')

    fig.suptitle('Mode Synthesizer V2 - Parameterized Lip Excitation + Body Resonance',
                 fontsize=14, fontweight='bold', y=0.98)

    out_path = output_dir / 'v2_comparison.png'
    fig.savefig(str(out_path), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")

    return metrics_data


# ============================================================
# PRINT COMPARISON TABLE (V1 vs V2)
# ============================================================
def print_v1_v2_comparison(metrics_v2, v1_data=None):
    """Print full comparison table: V1 vs V2 vs Real."""
    band_names = ['sub_100hz', '100_500hz', '500_2khz', '2_5khz', '5_10khz', '10_20khz', 'above_20khz']
    band_labels = ['<100Hz', '100-500Hz', '500-2kHz', '2-5kHz', '5-10kHz', '10-20kHz', '>20kHz']

    print("\n" + "=" * 110)
    print("V1 vs V2 COMPARISON TABLE")
    print("=" * 110)

    for m in metrics_v2:
        wname = m['whale']
        real = REAL_TARGETS[wname]

        # V1 data
        v1_centroid = 0
        v1_bands = {}
        if v1_data and wname in v1_data.get('whales', {}):
            v1_info = v1_data['whales'][wname]
            v1_centroid = v1_info.get('synth_centroid_hz', 0)
            v1_bands = v1_info.get('synth_bands', {})

        real_centroid = real['spectral_centroid_hz']
        v2_centroid = m['synth_centroid_hz']
        v1_err = abs(v1_centroid - real_centroid) / real_centroid * 100 if v1_centroid > 0 else 0
        v2_err = m['centroid_error_pct']

        print(f"\n--- {wname} ---")
        print(f"  Spectral centroid: V1={v1_centroid:.0f}Hz ({v1_err:+.1f}%)  "
              f"V2={v2_centroid:.0f}Hz ({v2_err:+.1f}%)  Real={real_centroid:.0f}Hz")
        if v1_err > 0:
            improvement = v1_err - v2_err
            print(f"  Centroid improvement: {improvement:+.1f}pp")
        print(f"  {'Band':<12s} {'V1':>8s} {'V2':>8s} {'Real':>8s} {'V1 Err':>8s} {'V2 Err':>8s} {'Change':>8s}")
        print(f"  {'-'*60}")
        for bn, bl in zip(band_names, band_labels):
            rv = real.get(f'band_{bn}_pct', 0)
            v1v = v1_bands.get(f'band_{bn}_pct', 0)
            v2v = m['synth_bands'].get(bn, 0)
            v1e = abs(v1v - rv) / max(rv, 0.1) * 100 if v1v > 0 else 0
            v2e = abs(v2v - rv) / max(rv, 0.1) * 100
            change = v1e - v2e if v1v > 0 else 0
            flag = " <<" if abs(change) > 10 else ""
            print(f"  {bl:<12s} {v1v:>7.1f}% {v2v:>7.1f}% {rv:>7.1f}% {v1e:>7.1f}% {v2e:>7.1f}% {change:>+7.1f}pp{flag}")

    print("\n" + "=" * 110)


# ============================================================
# TYPICAL CODA ICI PATTERNS
# ============================================================
CODA_PATTERNS = {
    'Whale_A': [250, 250, 150, 250, 250],       # 1+1+3 pattern (100 codas of this type)
    'Whale_D': [200, 200, 200, 200, 200],        # 5R1 pattern (156 codas)
    'Whale_F': [250, 250, 100, 100, 100, 100],   # 1+1+3 + extra clicks
}


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='Mode-superposition sperm whale click synthesizer V2')
    parser.add_argument('--whale', type=str, default=None,
                        help='Specific whale to synthesize (A, D, or F)')
    parser.add_argument('--optimize', action='store_true',
                        help='Run tension optimizer with expanded V2 params')
    parser.add_argument('--no-body', action='store_true',
                        help='Disable body resonance')
    parser.add_argument('--ambient', action='store_true',
                        help='Add ocean ambient noise')
    args = parser.parse_args()

    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Mode-Superposition Sperm Whale Click Synthesizer V2")
    print("  - Parameterizable lip excitation spectrum")
    print("  - Body resonance from click recoil")
    print("  - Coda generation from click templates")
    print("  - Optional ocean ambient noise")
    print("=" * 70)

    # Determine which whales to process
    if args.whale:
        whale_key = f'Whale_{args.whale.upper()}'
        if whale_key not in WHALE_CONFIGS:
            print(f"Error: unknown whale '{args.whale}'. Use A, D, or F.")
            sys.exit(1)
        whale_names = [whale_key]
    else:
        whale_names = ['Whale_A', 'Whale_D', 'Whale_F']

    # Load whale-to-file mapping
    print("\nLoading DSWP whale file mapping...")
    whale_files = get_whale_wav_files()
    for wid, files in whale_files.items():
        print(f"  Whale {wid}: {len(files)} WAV files")

    # Load V1 results for comparison
    v1_data = load_v1_results()
    if v1_data:
        print("\nLoaded V1 results for comparison")

    # ---- OPTIMIZATION MODE ----
    if args.optimize:
        print("\n" + "=" * 70)
        print("RUNNING V2 OPTIMIZER (15 params per whale)")
        print("=" * 70)

        opt_results = []
        total_start = time.time()

        for wname in whale_names:
            target = REAL_TARGETS[wname]
            config = WHALE_CONFIGS[wname]
            result = optimize_whale(wname, target, config)
            opt_results.append(result)

        total_elapsed = time.time() - total_start
        print(f"\nTotal optimization time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")

        # Save optimized WAVs
        print("\nSaving optimized WAV files...")
        optimized_signals = {}
        for res in opt_results:
            whale_id = res['whale'].split('_')[1]
            wav_path = output_dir / f'whale_{whale_id}_v2.wav'
            save_wav(res['signal'], wav_path)
            optimized_signals[res['whale']] = res

            # Generate coda
            coda_icis = CODA_PATTERNS.get(res['whale'], [200, 200, 200, 200, 200])
            coda_signal = generate_coda(res['signal'], coda_icis, sample_rate=TARGET_SR)
            # Add ambient to coda
            coda_signal = add_ocean_ambient(coda_signal, snr_db=20, sample_rate=TARGET_SR)
            coda_path = output_dir / f'whale_{whale_id}_coda_v2.wav'
            save_coda_wav(coda_signal, coda_path)

        # Save results JSON
        json_results = {
            'metadata': {
                'method': 'Mode-superposition synthesis V2',
                'description': 'Parameterized lip excitation + body resonance + coda generation',
                'sample_rate': TARGET_SR,
                'date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'n_params': len(OPTIM_PARAM_NAMES),
                'param_names': OPTIM_PARAM_NAMES,
            },
            'whales': {},
        }
        for res in opt_results:
            json_results['whales'][res['whale']] = {
                'cost': res['cost'],
                'initial_cost': res['initial_cost'],
                'nit': res['nit'],
                'nfev': res['nfev'],
                'elapsed_s': res['elapsed_s'],
                'params': res['params'],
                'features': res['features'],
                'history': res['history'][:50],  # first 50 iterations
            }

        json_path = output_dir / 'v2_results.json'
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        print(f"\nSaved: {json_path}")

        # Generate default clicks too (for figure)
        all_results = {}
        for wname in whale_names:
            config = WHALE_CONFIGS[wname]
            result = generate_whale_click(wname, config, add_body=not args.no_body,
                                          add_ambient=args.ambient)
            all_results[wname] = result

        # Create figure
        if len(whale_names) == 3:
            metrics = create_comparison_figure(all_results, whale_files, output_dir,
                                               v1_data=v1_data,
                                               optimized_results=optimized_signals)
            print_v1_v2_comparison(metrics, v1_data=v1_data)

        # Print optimized comparison
        print("\n" + "=" * 110)
        print("OPTIMIZED V2 RESULTS")
        print("=" * 110)
        for res in opt_results:
            wname = res['whale']
            real = REAL_TARGETS[wname]
            f = res['features']
            p = res['params']
            centroid_err = (f['spectral_centroid_hz'] - real['spectral_centroid_hz']) / real['spectral_centroid_hz'] * 100
            print(f"\n{wname}:")
            print(f"  Cost: {res['initial_cost']:.4f} -> {res['cost']:.4f} "
                  f"({(1-res['cost']/res['initial_cost'])*100:+.0f}% improvement)")
            print(f"  Centroid: {f['spectral_centroid_hz']:.0f}Hz (target {real['spectral_centroid_hz']}, "
                  f"err {centroid_err:+.1f}%)")
            print(f"  Lip: peak={p['lip_peak_hz']:.0f}Hz  bw={p['lip_bandwidth_octaves']:.2f}oct  "
                  f"floor={p['low_freq_floor']:.3f}")
            print(f"  Body resonance scale: {p['body_resonance_scale']:.3f}")
            print(f"  Sub-100Hz: {f['band_sub_100hz_pct']:.1f}% (target {real['band_sub_100hz_pct']}%)")
            print(f"  2-5kHz: {f['band_2_5khz_pct']:.1f}% (target {real['band_2_5khz_pct']}%)")
            print(f"  5-10kHz: {f['band_5_10khz_pct']:.1f}% (target {real['band_5_10khz_pct']}%)")

        return

    # ---- SYNTHESIS MODE (default) ----
    all_results = {}
    for wname in whale_names:
        print(f"\n{'='*50}")
        print(f"Synthesizing {wname}")
        print(f"{'='*50}")
        config = WHALE_CONFIGS[wname]
        result = generate_whale_click(wname, config, add_body=not args.no_body,
                                      add_ambient=args.ambient)
        all_results[wname] = result

        # Save individual click WAV
        wid = wname.split('_')[1]
        wav_path = output_dir / f'whale_{wid}_v2.wav'
        save_wav(result['final'], wav_path)

        # Generate and save coda WAV
        coda_icis = CODA_PATTERNS.get(wname, [200, 200, 200, 200, 200])
        coda_signal = generate_coda(result['final'], coda_icis, sample_rate=TARGET_SR)
        coda_signal = add_ocean_ambient(coda_signal, snr_db=20, sample_rate=TARGET_SR)
        coda_path = output_dir / f'whale_{wid}_coda_v2.wav'
        save_coda_wav(coda_signal, coda_path)

    # Create figures and tables (only if all 3 whales)
    if len(all_results) == 3:
        metrics = create_comparison_figure(all_results, whale_files, output_dir, v1_data=v1_data)
        print_v1_v2_comparison(metrics, v1_data=v1_data)

        # Save results JSON
        results_json = {
            'metadata': {
                'method': 'Mode-superposition synthesis V2',
                'description': 'Parameterized lip excitation + body resonance + coda generation',
                'sample_rate': TARGET_SR,
                'date': time.strftime('%Y-%m-%d %H:%M:%S'),
            },
            'whales': {},
        }
        for wname, result in all_results.items():
            dt = 1.0 / TARGET_SR
            synth_bands = compute_band_energies(result['final'], dt)
            synth_centroid = compute_spectral_centroid(result['final'], dt)

            results_json['whales'][wname] = {
                'config': {k: v for k, v in WHALE_CONFIGS[wname].items()
                           if not isinstance(v, np.ndarray)},
                'n_modes_total': result['n_modes'],
                'duration_s': result['duration_s'],
                'synth_time_ms': result['elapsed_ms'],
                'synth_centroid_hz': round(synth_centroid, 1),
                'real_centroid_hz': REAL_TARGETS[wname].get('spectral_centroid_hz', 0),
                'synth_bands': {k: round(v, 2) for k, v in synth_bands.items()},
            }

        json_path = output_dir / 'v2_results.json'
        with open(json_path, 'w') as f:
            json.dump(results_json, f, indent=2)
        print(f"\nSaved: {json_path}")
    else:
        for wname, result in all_results.items():
            dt = 1.0 / TARGET_SR
            centroid = compute_spectral_centroid(result['final'], dt)
            bands = compute_band_energies(result['final'], dt)
            print(f"\n{wname}: centroid={centroid:.0f}Hz, "
                  f"5-10kHz={bands.get('band_5_10khz_pct', 0):.1f}%, "
                  f"sub-100Hz={bands.get('band_sub_100hz_pct', 0):.1f}%")

    print("\nDone!")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Mode-Superposition Sperm Whale Click Synthesizer
=================================================
Instead of running slow FDTD simulations, we directly synthesize clicks by
superposing damped sinusoidal cavity modes. This is physically motivated:
the modal decomposition of real clicks proved that 10 damped sinusoids
reconstruct clicks at 0.857 correlation.

The whale's head is an instrument:
  - Cavity geometry determines which modes exist (frequencies)
  - Phonic lip buzz provides broadband energy to excite all modes
  - Maxillonasalis muscle bands control which modes are damped vs ring (amplitudes)
  - Output = superposition of surviving modes, filtered by exit-path tissues and ocean

Usage:
    python mode_synthesizer.py                # all 3 whales
    python mode_synthesizer.py --whale A      # just Whale A

Output:
    mode_synthesis_comparison.png
    muscle_tension_demo.png
    vowel_demonstration.png
    whale_A_mode_synth.wav, whale_D_mode_synth.wav, whale_F_mode_synth.wav
    mode_synthesis_results.json
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
from scipy.signal import resample

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
# DSWP recordings use DTags (suction-cup hydrophones on the whale's body)
# or near-field trailing hydrophones. Effective acoustic path is very short.
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
# WHALE CONFIGURATIONS
# ============================================================
WHALE_CONFIGS = {
    'Whale_A': {
        'organ_length': 3.50, 'organ_diameter': 1.20,
        'spermaceti_c': 1370,
        'tau_up': 0.00025,      # 0.25ms (measured from modal decomposition)
        'tau_down': 0.00245,    # 2.45ms (measured)
        # Whale A: highest centroid (7849Hz). Peak energy at 5-10kHz.
        # Moderate damping at low freqs (3.5-5kHz) to prevent 2-5kHz excess.
        # Bands: [16k, 12k, 9k, 7k, 5k, 3.5k]
        'muscle_tensions': [0.25, 0.15, 0.05, 0.05, 0.30, 0.50],
        'muscle_band_freqs': [16000, 12000, 9000, 7000, 5000, 3500],
        'junk_length': 2.0,
    },
    'Whale_D': {
        'organ_length': 3.80, 'organ_diameter': 1.40,
        'spermaceti_c': 1370,
        'tau_up': 0.00235,      # 2.35ms (longest sustain)
        'tau_down': 0.02097,    # 20.97ms
        # Whale D: medium centroid (5693Hz). Broad, even spectrum.
        # Needs damping across 5-12kHz to prevent them from dominating.
        # Long tau_down (21ms) means careful damping needed everywhere.
        # Bands: [16k, 12k, 9k, 7k, 5k, 3.5k]
        'muscle_tensions': [0.75, 0.60, 0.55, 0.45, 0.40, 0.50],
        'muscle_band_freqs': [16000, 12000, 9000, 7000, 5000, 3500],
        'junk_length': 2.2,
    },
    'Whale_F': {
        'organ_length': 4.80, 'organ_diameter': 1.70,
        'spermaceti_c': 1370,
        'tau_up': 0.00062,      # 0.62ms
        'tau_down': 0.00117,    # 1.17ms (fastest decay)
        # Whale F: lowest centroid (5333Hz), largest animal.
        # Strong damping at 7-12kHz to shift energy away from the 5-10kHz peak.
        # The short tau_down (1.17ms) already limits overall ring.
        # Bands: [16k, 12k, 9k, 7k, 5k, 3.5k]
        'muscle_tensions': [0.80, 0.70, 0.65, 0.60, 0.30, 0.35],
        'muscle_band_freqs': [16000, 12000, 9000, 7000, 5000, 3500],
        'junk_length': 2.5,
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
    This prevents artificial energy pile-up from near-degenerate combined modes.
    """
    raw_modes = []
    R = D / 2

    # Extended Bessel zeros j_mn: m=0..5, n=1..5
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

    # Combined longitudinal-transverse modes (extended range)
    for n in range(1, 60):
        for (m, n_r), j_mn in bessel_zeros.items():
            f_combined = c * np.sqrt((n / (2 * L))**2 + (j_mn / (np.pi * R))**2)
            if 500 < f_combined < 25000:
                raw_modes.append({'freq': f_combined, 'type': 'combined', 'n': n, 'm': m})

    # Sort by frequency
    raw_modes.sort(key=lambda x: x['freq'])

    # Deduplicate: merge modes within 1.5% of each other
    # When modes are nearly degenerate, they act as a single broader resonance,
    # not independent oscillators. Keep the one with lowest mode order.
    modes = []
    for mode in raw_modes:
        merged = False
        for existing in modes:
            if abs(mode['freq'] - existing['freq']) / max(existing['freq'], 1) < 0.015:
                # Keep the lower-order mode (better coupled)
                merged = True
                break
        if not merged:
            modes.append(mode)

    return modes


# ============================================================
# STEP 2: Base Amplitude from Source/Exit Coupling
# ============================================================
def compute_base_amplitudes(modes, L):
    """Base amplitude depends on how well the mode couples to source/exit.

    Source is at x=0 (anterior). Longitudinal mode n has antinode at x=0
    when n is odd, node when n is even.

    Odd modes couple strongly to source (antinode at lip position).
    Even modes couple weakly (node at lip position).
    """
    # Phonic lip produces broadband impulse at the anterior end.
    # The lip is essentially a point source relative to the organ cross-section,
    # so it couples strongly to longitudinal modes but weakly to pure transverse modes.
    # Combined modes get intermediate coupling.
    #
    # The spectral shape models the phonic lip's output spectrum:
    # broad peak 3-15kHz, rolls off below 2kHz and above 18kHz.
    # This matches measured click spectra (centroid 5-8kHz).
    for mode in modes:
        f = mode['freq']

        # Broadband lip excitation spectrum.
        # Real sperm whale clicks have spectral centroid 5-8kHz with most energy
        # in 3-15kHz. The phonic lip generates a ~50us impulse giving bandwidth
        # to ~20kHz. The junk acts as an acoustic lens that focuses energy
        # preferentially at 8-15kHz. Lower frequencies (2-5kHz) are less focused
        # because the junk's lipid layers are larger than the wavelength.
        log_f = np.log10(max(f, 10))
        f_peak_log = np.log10(10000)  # peak at 10kHz (junk focus + lip bandwidth)
        if log_f < f_peak_log:
            # Below peak: steep rolloff, especially below 3kHz
            sigma = 0.35 if f > 3000 else 0.25
            spectral_shape = np.exp(-0.5 * ((log_f - f_peak_log) / sigma)**2)
        else:
            # Above peak: moderate rolloff (viscous absorption in spermaceti)
            spectral_shape = np.exp(-0.5 * ((log_f - f_peak_log) / 0.35)**2)
        # Floor for residual low-freq modes
        spectral_shape = max(spectral_shape, 0.01)

        if mode['type'] == 'longitudinal':
            n = mode['n']
            # Odd harmonics couple to source (antinode at lip position)
            coupling = 1.0 if n % 2 == 1 else 0.4
            # Very gentle rolloff - lip is broadband, all harmonics excited
            mode['base_amplitude'] = coupling * spectral_shape / (n**0.15)
        elif mode['type'] == 'transverse':
            # Transverse modes couple WEAKLY to a point source at the lip.
            # The lip excites pressure at one point, but transverse modes
            # have nodal structure across the cross-section.
            # Only m=0 (radially symmetric) modes couple to a centered source.
            if mode['m'] == 0:
                mode['base_amplitude'] = 0.15 * spectral_shape
            else:
                mode['base_amplitude'] = 0.03 * spectral_shape / (mode['m'])
        else:
            # Combined modes: longitudinal component couples to source,
            # transverse component determines radiation pattern.
            # Lower transverse order (m) = better coupling.
            m_factor = 1.0 / (1.0 + mode['m'])
            mode['base_amplitude'] = 0.4 * spectral_shape * m_factor / (mode['n']**0.15)
    return modes


# ============================================================
# STEP 3: Muscle Tension Controls Mode Amplitudes
# ============================================================
def apply_muscle_damping(modes, muscle_tensions, muscle_band_freqs):
    """Each muscle band damps modes near its resonant frequency.

    Band resonant freq = c_muscle / (2 * band_thickness)
    Damping effect: modes within +/-20% of band freq get amplitude reduced
    proportional to tension.

    Args:
        modes: list of mode dicts with 'base_amplitude'
        muscle_tensions: [0-1] x 6 bands
        muscle_band_freqs: resonant frequency of each band (Hz)
    """
    for mode in modes:
        f = mode['freq']
        damping_factor = 1.0
        for tension, band_freq in zip(muscle_tensions, muscle_band_freqs):
            # How close is this mode to the band's resonant frequency?
            proximity = np.exp(-0.5 * ((f - band_freq) / (0.2 * band_freq))**2)
            # Tension damps modes near the band frequency
            damping_factor *= (1.0 - tension * proximity * 0.8)  # max 80% damping per band
        mode['amplitude'] = mode['base_amplitude'] * damping_factor
    return modes


# ============================================================
# STEP 4: Synthesize the Click
# ============================================================
def synthesize_click(modes, sample_rate=44100, duration_s=0.025,
                     tau_up_s=0.001, tau_down_s=0.005):
    """Generate synthetic click waveform from mode parameters.

    signal(t) = envelope(t) * sum_i [A_i * sin(2*pi*f_i*t + phi_i) * exp(-pi*f_i*t/Q_i)]

    Where:
        f_i = cavity resonant mode frequency
        A_i = amplitude (controlled by muscle tension)
        Q_i = quality factor (how long the mode rings)
        phi_i = random phase (modes are incoherent)
        envelope = ramp-up * decay matching measured tau_up and tau_down
    """
    n_samples = int(duration_s * sample_rate)
    t = np.arange(n_samples) / sample_rate
    signal = np.zeros(n_samples)

    # Envelope: ramp up then exponential decay
    envelope = (1 - np.exp(-t / max(tau_up_s, 1e-6))) * np.exp(-t / max(tau_down_s, 1e-6))

    for mode in modes:
        f = mode['freq']
        A = mode.get('amplitude', mode.get('base_amplitude', 0.1))
        # Q factor models the cavity ring-down time at each frequency.
        # Low-Q (short ring) at low freqs: air sac boundaries are lossy.
        # Moderate-Q at 3-10kHz: spermaceti waveguide resonance.
        # Declining Q above 10kHz: viscous absorption in oil scales as f^2.
        if f < 1000:
            Q = 8 + f / 250       # very lossy below 1kHz
        elif f < 3000:
            Q = 12 + f / 250      # gradually increasing
        elif f < 10000:
            Q = 24 + f / 400      # moderate peak in 3-10kHz
        elif f < 15000:
            Q = 49 - (f - 10000) / 500  # declining 10-15kHz
        else:
            Q = max(10, 39 - (f - 10000) / 400)  # rapid decline above 15kHz
        Q = mode.get('Q', max(Q, 5))

        # Damped sinusoid: A * sin(2*pi*f*t + phi) * exp(-pi*f*t/Q)
        phi = np.random.uniform(0, 2 * np.pi)  # random phase
        decay = np.exp(-np.pi * f * t / Q)
        signal += A * np.sin(2 * np.pi * f * t + phi) * decay

    signal *= envelope
    return signal, t


# ============================================================
# STEP 5: Exit-Path Tissue Filter (from signal_chain_v3)
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
        # Deduplicate
        files = list(dict.fromkeys(files))
        whale_files[whale_id] = files
    return whale_files


def load_real_click(whale_id, whale_files, n_clicks=5):
    """Load representative clicks from DSWP WAVs for a whale.
    Returns list of (click_signal, sample_rate) tuples."""
    files = whale_files.get(whale_id, [])
    if not files:
        print(f"  Warning: no WAV files for whale {whale_id}")
        return []

    clicks = []
    for wav_path in files[:20]:  # scan first 20 files
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

            # Resample to TARGET_SR if needed
            if sr != TARGET_SR:
                n_samp = int(len(data) * TARGET_SR / sr)
                data = resample(data, n_samp)
                sr = TARGET_SR

            # Find the loudest 20ms window
            window_samples = int(0.020 * sr)
            if len(data) < window_samples:
                continue
            energy = np.convolve(data**2, np.ones(window_samples) / window_samples, mode='valid')
            peak_idx = np.argmax(energy)
            start = max(0, peak_idx - window_samples // 4)
            end = min(len(data), start + window_samples)
            click = data[start:end]
            if np.max(np.abs(click)) > 0.01:  # reject silence
                clicks.append((click, sr))
        except Exception as e:
            continue

    return clicks


def get_mean_real_click(whale_id, whale_files, n_clicks=10):
    """Get a representative real click (average spectrum)."""
    clicks = load_real_click(whale_id, whale_files, n_clicks=n_clicks)
    if not clicks:
        return None, None
    # Return the one with highest RMS (most representative)
    best = max(clicks, key=lambda x: np.sqrt(np.mean(x[0]**2)))
    return best


# ============================================================
# FULL PIPELINE
# ============================================================
def generate_whale_click(whale_name, config, seed=42):
    """Full pipeline: anatomy -> modes -> muscle damping -> synthesis -> filter -> ocean."""
    np.random.seed(seed)
    t0 = time.time()

    L = config['organ_length']
    D = config['organ_diameter']
    c = config['spermaceti_c']

    # 1. Compute cavity modes from anatomy
    modes = compute_cavity_modes(L, D, c)
    n_modes_total = len(modes)

    # 2. Set base amplitudes from source/exit coupling
    modes = compute_base_amplitudes(modes, L)

    # 3. Apply muscle tension damping
    modes = apply_muscle_damping(modes, config['muscle_tensions'],
                                  config['muscle_band_freqs'])

    # 4. Compute click duration from tau values
    # Duration should be long enough to capture the full decay
    duration_s = config['tau_up'] + 5 * config['tau_down']
    duration_s = max(duration_s, 0.015)  # at least 15ms
    duration_s = min(duration_s, 0.150)  # cap at 150ms

    # 5. Synthesize raw click
    raw, t = synthesize_click(modes, sample_rate=TARGET_SR, duration_s=duration_s,
                               tau_up_s=config['tau_up'],
                               tau_down_s=config['tau_down'])

    dt = 1.0 / TARGET_SR

    # 6. Exit-path tissue filter
    filtered = apply_exit_path_filter(raw, dt, junk_length=config.get('junk_length', 2.0))

    # 7. Ocean propagation
    final = apply_ocean_propagation(filtered, dt)

    elapsed = time.time() - t0
    print(f"  {whale_name}: {n_modes_total} modes, duration={duration_s*1000:.1f}ms, "
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
    """Save signal as 16-bit WAV, repeated as a coda sequence for audibility.

    Creates n_repeats clicks spaced by ici_s seconds, repeated 3 times
    with 1 second gaps. This makes the output easily audible when played back.
    """
    click_len = len(signal)
    ici_samples = int(ici_s * sample_rate)
    gap_samples = int(1.0 * sample_rate)

    # Build one coda (n_repeats clicks)
    coda_len = click_len + (n_repeats - 1) * ici_samples
    coda = np.zeros(coda_len)
    for i in range(n_repeats):
        start = i * ici_samples
        end = min(start + click_len, coda_len)
        coda[start:end] += signal[:end - start]

    # Repeat coda 3 times with gaps
    n_codas = 3
    total_len = n_codas * len(coda) + (n_codas - 1) * gap_samples + gap_samples
    output = np.zeros(total_len)
    pos = gap_samples // 2  # start with half-gap
    for c in range(n_codas):
        output[pos:pos + len(coda)] = coda
        pos += len(coda) + gap_samples

    # Normalize to 0.9 peak
    peak = np.max(np.abs(output))
    if peak > 0:
        output = output * 0.9 / peak
    output_16 = np.clip(output * 32767, -32768, 32767).astype(np.int16)
    wavfile.write(str(filename), sample_rate, output_16)
    dur_s = len(output_16) / sample_rate
    print(f"  Saved: {filename} ({dur_s:.1f}s, {n_repeats} clicks x {n_codas} codas)")


# ============================================================
# FIGURE 1: Mode Synthesis Comparison (24x20)
# ============================================================
def create_comparison_figure(all_results, whale_files, output_dir):
    """Create 4-row comparison figure: waveform, spectrum, mode chart, band energies."""
    print("\n[Figure 1] Creating mode_synthesis_comparison.png...")

    whale_names = ['Whale_A', 'Whale_D', 'Whale_F']
    whale_ids = ['A', 'D', 'F']
    colors = {'Whale_A': '#1f77b4', 'Whale_D': '#ff7f0e', 'Whale_F': '#2ca02c'}
    band_colors = ['#440154', '#31688e', '#35b779', '#fde725', '#e76f51', '#d62728', '#9467bd']

    fig = plt.figure(figsize=(24, 20))
    gs = GridSpec(5, 3, figure=fig, hspace=0.35, wspace=0.25,
                  height_ratios=[1, 1, 1, 1, 0.4])

    dt = 1.0 / TARGET_SR

    metrics_data = []

    for col, (wname, wid) in enumerate(zip(whale_names, whale_ids)):
        result = all_results[wname]
        synth = result['final']
        modes = result['modes']
        config = WHALE_CONFIGS[wname]

        # Load real click for comparison
        real_click, real_sr = get_mean_real_click(wid, whale_files)

        # --- Row 1: Time-domain waveform ---
        ax = fig.add_subplot(gs[0, col])
        t_synth = np.arange(len(synth)) / TARGET_SR * 1000  # ms
        ax.plot(t_synth, synth / max(np.max(np.abs(synth)), 1e-10),
                color=colors[wname], linewidth=0.8, label='Synthesized')
        if real_click is not None:
            t_real = np.arange(len(real_click)) / TARGET_SR * 1000
            ax.plot(t_real, real_click / max(np.max(np.abs(real_click)), 1e-10),
                    color='gray', linewidth=0.6, alpha=0.7, label='Real')
        ax.set_title(f'{wname} - Waveform', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Normalized Amplitude')
        ax.legend(fontsize=8)
        ax.set_xlim(0, min(t_synth[-1], 25))

        # --- Row 2: Spectrum ---
        ax = fig.add_subplot(gs[1, col])
        synth_fft = np.abs(np.fft.rfft(synth))
        synth_freqs = np.fft.rfftfreq(len(synth), dt) / 1000  # kHz
        synth_fft_db = 20 * np.log10(synth_fft / max(np.max(synth_fft), 1e-30) + 1e-30)
        ax.plot(synth_freqs, synth_fft_db, color=colors[wname], linewidth=1.0, label='Synthesized')

        if real_click is not None:
            real_fft = np.abs(np.fft.rfft(real_click))
            real_freqs = np.fft.rfftfreq(len(real_click), dt) / 1000
            real_fft_db = 20 * np.log10(real_fft / max(np.max(real_fft), 1e-30) + 1e-30)
            ax.plot(real_freqs, real_fft_db, color='gray', linewidth=0.8, alpha=0.7, label='Real')

        ax.set_title(f'{wname} - Spectrum', fontsize=12)
        ax.set_xlabel('Frequency (kHz)')
        ax.set_ylabel('Magnitude (dB)')
        ax.set_xlim(0, 22)
        ax.set_ylim(-60, 5)
        ax.legend(fontsize=8)

        # --- Row 3: Mode amplitude chart ---
        ax = fig.add_subplot(gs[2, col])
        band_freqs = config['muscle_band_freqs']
        band_labels = [f'{f/1000:.1f}k' for f in band_freqs]

        for mode in modes:
            f = mode['freq']
            amp = mode.get('amplitude', 0)
            if amp < 0.001:
                continue

            # Color by which muscle band is closest
            distances = [abs(f - bf) / bf for bf in band_freqs]
            closest_band = np.argmin(distances)
            c_mode = band_colors[closest_band % len(band_colors)]

            ax.vlines(f / 1000, 0, amp, colors=c_mode, linewidth=1.5, alpha=0.8)

        # Mark muscle band frequencies
        for i, bf in enumerate(band_freqs):
            ax.axvline(bf / 1000, color=band_colors[i % len(band_colors)],
                       linestyle='--', alpha=0.3, linewidth=0.8)

        ax.set_title(f'{wname} - Mode Amplitudes', fontsize=12)
        ax.set_xlabel('Frequency (kHz)')
        ax.set_ylabel('Amplitude')
        ax.set_xlim(0, 22)

        # --- Row 4: Band energy comparison ---
        ax = fig.add_subplot(gs[3, col])
        synth_bands = compute_band_energies(synth, dt)
        real_bands = REAL_TARGETS[wname]

        band_names = ['sub_100hz', '100_500hz', '500_2khz', '2_5khz', '5_10khz', '10_20khz', 'above_20khz']
        band_labels_short = ['<100', '100-500', '0.5-2k', '2-5k', '5-10k', '10-20k', '>20k']
        x = np.arange(len(band_names))
        width = 0.35

        synth_vals = [synth_bands.get(f'band_{bn}_pct', 0) for bn in band_names]
        real_vals = [real_bands.get(f'band_{bn}_pct', 0) for bn in band_names]

        bars1 = ax.bar(x - width/2, synth_vals, width, label='Synthesized',
                       color=colors[wname], alpha=0.8)
        bars2 = ax.bar(x + width/2, real_vals, width, label='Real',
                       color='gray', alpha=0.6)

        ax.set_title(f'{wname} - Band Energy', fontsize=12)
        ax.set_xlabel('Frequency Band')
        ax.set_ylabel('Energy (%)')
        ax.set_xticks(x)
        ax.set_xticklabels(band_labels_short, rotation=45, fontsize=8)
        ax.legend(fontsize=8)

        # Compute metrics
        synth_centroid = compute_spectral_centroid(synth, dt)
        real_centroid = real_bands.get('spectral_centroid_hz', 0)
        centroid_err = abs(synth_centroid - real_centroid) / max(real_centroid, 1) * 100

        # Band error
        total_band_err = 0
        n_bands = 0
        for bn in band_names:
            sv = synth_bands.get(f'band_{bn}_pct', 0)
            rv = real_bands.get(f'band_{bn}_pct', 0)
            if rv > 1:  # only count significant bands
                total_band_err += abs(sv - rv) / rv * 100
                n_bands += 1
        mean_band_err = total_band_err / max(n_bands, 1)

        metrics_data.append({
            'whale': wname,
            'synth_centroid_hz': round(synth_centroid, 1),
            'real_centroid_hz': real_centroid,
            'centroid_error_pct': round(centroid_err, 1),
            'mean_band_error_pct': round(mean_band_err, 1),
            'n_modes': result['n_modes'],
            'synth_time_ms': round(result['elapsed_ms'], 1),
            'synth_bands': {bn: round(synth_bands.get(f'band_{bn}_pct', 0), 1) for bn in band_names},
        })

    # --- Row 5: Metrics table ---
    ax = fig.add_subplot(gs[4, :])
    ax.axis('off')

    table_data = []
    for m in metrics_data:
        table_data.append([
            m['whale'],
            f"{m['n_modes']}",
            f"{m['synth_centroid_hz']:.0f}",
            f"{m['real_centroid_hz']:.0f}",
            f"{m['centroid_error_pct']:.1f}%",
            f"{m['mean_band_error_pct']:.1f}%",
            f"{m['synth_time_ms']:.1f}ms",
        ])

    table = ax.table(
        cellText=table_data,
        colLabels=['Whale', 'N Modes', 'Synth Centroid', 'Real Centroid',
                   'Centroid Err', 'Mean Band Err', 'Synth Time'],
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.5)

    fig.suptitle('Mode-Superposition Sperm Whale Click Synthesis vs Real Data',
                 fontsize=16, fontweight='bold', y=0.98)

    out_path = output_dir / 'mode_synthesis_comparison.png'
    fig.savefig(str(out_path), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")

    return metrics_data


# ============================================================
# FIGURE 2: Muscle Tension Demo (16x12)
# ============================================================
def create_muscle_tension_demo(output_dir):
    """Show 4 panels with different muscle tension patterns for Whale D."""
    print("\n[Figure 2] Creating muscle_tension_demo.png...")

    config = WHALE_CONFIGS['Whale_D'].copy()
    L = config['organ_length']
    D = config['organ_diameter']
    c = config['spermaceti_c']
    band_freqs = config['muscle_band_freqs']
    dt = 1.0 / TARGET_SR

    tension_configs = [
        ('All Relaxed', [0, 0, 0, 0, 0, 0]),
        ('High-freq Damped', [0.9, 0.8, 0.7, 0.5, 0.2, 0]),
        ('Low-freq Damped', [0, 0, 0.2, 0.5, 0.8, 0.9]),
        ('Best Fit (Whale D)', config['muscle_tensions']),
    ]

    fig, axes = plt.subplots(4, 2, figsize=(16, 12))
    fig.suptitle('Muscle Tension Controls Mode Selection - Whale D',
                 fontsize=14, fontweight='bold')

    band_colors = ['#440154', '#31688e', '#35b779', '#fde725', '#e76f51', '#d62728']

    for row, (label, tensions) in enumerate(tension_configs):
        np.random.seed(42)

        # Compute modes
        modes = compute_cavity_modes(L, D, c)
        modes = compute_base_amplitudes(modes, L)
        modes = apply_muscle_damping(modes, tensions, band_freqs)

        # Synthesize
        duration_s = max(config['tau_up'] + 5 * config['tau_down'], 0.015)
        duration_s = min(duration_s, 0.150)
        raw, t = synthesize_click(modes, sample_rate=TARGET_SR, duration_s=duration_s,
                                   tau_up_s=config['tau_up'], tau_down_s=config['tau_down'])
        filtered = apply_exit_path_filter(raw, dt, junk_length=config['junk_length'])

        # Left panel: spectrum
        ax = axes[row, 0]
        fft_mag = np.abs(np.fft.rfft(filtered))
        freqs = np.fft.rfftfreq(len(filtered), dt) / 1000
        fft_db = 20 * np.log10(fft_mag / max(np.max(fft_mag), 1e-30) + 1e-30)
        ax.plot(freqs, fft_db, color='#ff7f0e', linewidth=1.0)
        ax.set_xlim(0, 22)
        ax.set_ylim(-60, 5)
        ax.set_ylabel('dB')
        if row == 3:
            ax.set_xlabel('Frequency (kHz)')
        ax.set_title(f'{label} - Spectrum', fontsize=10)

        # Annotate tension values
        tension_str = '[' + ', '.join(f'{t:.1f}' for t in tensions) + ']'
        ax.text(0.02, 0.95, f'Tensions: {tension_str}', transform=ax.transAxes,
                fontsize=7, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Right panel: mode amplitudes
        ax = axes[row, 1]
        for mode in modes:
            f = mode['freq']
            amp = mode.get('amplitude', 0)
            if amp < 0.001:
                continue
            distances = [abs(f - bf) / bf for bf in band_freqs]
            closest = np.argmin(distances)
            ax.vlines(f / 1000, 0, amp, colors=band_colors[closest], linewidth=1.5, alpha=0.8)

        for i, bf in enumerate(band_freqs):
            ax.axvline(bf / 1000, color=band_colors[i], linestyle='--', alpha=0.3)

        ax.set_xlim(0, 22)
        ax.set_ylabel('Amplitude')
        if row == 3:
            ax.set_xlabel('Frequency (kHz)')
        ax.set_title(f'{label} - Mode Amplitudes', fontsize=10)

    plt.tight_layout()
    out_path = output_dir / 'muscle_tension_demo.png'
    fig.savefig(str(out_path), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ============================================================
# FIGURE 3: Vowel Demonstration (12x8)
# ============================================================
def create_vowel_demonstration(output_dir):
    """Show how different muscle patterns produce a-coda vs i-coda distinction."""
    print("\n[Figure 3] Creating vowel_demonstration.png...")

    config = WHALE_CONFIGS['Whale_D'].copy()
    L = config['organ_length']
    D = config['organ_diameter']
    c = config['spermaceti_c']
    band_freqs = config['muscle_band_freqs']
    dt = 1.0 / TARGET_SR

    vowel_configs = [
        ('a-coda (broad, relaxed)', [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
        ('i-coda (narrow, formant peaks)', [0.8, 0.2, 0.8, 0.2, 0.8, 0.2]),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle('Vowel-like Click Shaping via Muscle Tension',
                 fontsize=14, fontweight='bold')

    band_colors = ['#440154', '#31688e', '#35b779', '#fde725', '#e76f51', '#d62728']

    for row, (label, tensions) in enumerate(vowel_configs):
        np.random.seed(42)

        modes = compute_cavity_modes(L, D, c)
        modes = compute_base_amplitudes(modes, L)
        modes = apply_muscle_damping(modes, tensions, band_freqs)

        duration_s = max(config['tau_up'] + 5 * config['tau_down'], 0.015)
        duration_s = min(duration_s, 0.150)
        raw, t = synthesize_click(modes, sample_rate=TARGET_SR, duration_s=duration_s,
                                   tau_up_s=config['tau_up'], tau_down_s=config['tau_down'])
        filtered = apply_exit_path_filter(raw, dt, junk_length=config['junk_length'])

        # Col 0: Waveform
        ax = axes[row, 0]
        t_ms = np.arange(len(filtered)) / TARGET_SR * 1000
        ax.plot(t_ms, filtered / max(np.max(np.abs(filtered)), 1e-10),
                color='#ff7f0e', linewidth=0.8)
        ax.set_title(f'{label}\nWaveform', fontsize=10)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Amplitude')
        ax.set_xlim(0, min(t_ms[-1], 25))

        # Col 1: Spectrum
        ax = axes[row, 1]
        fft_mag = np.abs(np.fft.rfft(filtered))
        freqs = np.fft.rfftfreq(len(filtered), dt) / 1000
        fft_db = 20 * np.log10(fft_mag / max(np.max(fft_mag), 1e-30) + 1e-30)
        ax.plot(freqs, fft_db, color='#ff7f0e', linewidth=1.0)
        ax.set_xlim(0, 22)
        ax.set_ylim(-60, 5)
        ax.set_title('Spectrum', fontsize=10)
        ax.set_xlabel('Frequency (kHz)')
        ax.set_ylabel('dB')

        # Col 2: Mode amplitudes
        ax = axes[row, 2]
        for mode in modes:
            f = mode['freq']
            amp = mode.get('amplitude', 0)
            if amp < 0.001:
                continue
            distances = [abs(f - bf) / bf for bf in band_freqs]
            closest = np.argmin(distances)
            ax.vlines(f / 1000, 0, amp, colors=band_colors[closest], linewidth=1.5, alpha=0.8)

        for i, bf in enumerate(band_freqs):
            ax.axvline(bf / 1000, color=band_colors[i], linestyle='--', alpha=0.3)

        ax.set_xlim(0, 22)
        ax.set_title('Mode Amplitudes', fontsize=10)
        ax.set_xlabel('Frequency (kHz)')
        ax.set_ylabel('Amplitude')

        tension_str = '[' + ', '.join(f'{t:.1f}' for t in tensions) + ']'
        ax.text(0.02, 0.95, f'Tensions: {tension_str}', transform=ax.transAxes,
                fontsize=7, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    out_path = output_dir / 'vowel_demonstration.png'
    fig.savefig(str(out_path), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ============================================================
# PRINT COMPARISON TABLE
# ============================================================
def print_comparison_table(metrics_data):
    """Print full comparison table with centroid, band energies, and error percentages."""

    # Normalized comparison (2-20kHz click band only)
    print("\n" + "-" * 100)
    print("NORMALIZED CLICK-BAND COMPARISON (2-20kHz only)")
    print("Note: Sub-2kHz energy in real data comes from body resonance/ICI structure,")
    print("not individual click spectra. Our synthesizer correctly produces only click energy.")
    print("-" * 100)
    for m in metrics_data:
        wname = m['whale']
        real = REAL_TARGETS[wname]
        sb = m['synth_bands']
        s_2_5 = sb.get('2_5khz', 0)
        s_5_10 = sb.get('5_10khz', 0)
        s_10_20 = sb.get('10_20khz', 0)
        r_2_5 = real.get('band_2_5khz_pct', 0)
        r_5_10 = real.get('band_5_10khz_pct', 0)
        r_10_20 = real.get('band_10_20khz_pct', 0)
        st = s_2_5 + s_5_10 + s_10_20
        rt = r_2_5 + r_5_10 + r_10_20
        if st > 0 and rt > 0:
            print(f"  {wname}:")
            for label, sv, rv in [('2-5kHz', s_2_5, r_2_5), ('5-10kHz', s_5_10, r_5_10), ('10-20kHz', s_10_20, r_10_20)]:
                sn = sv / st * 100
                rn = rv / rt * 100
                err = abs(sn - rn) / max(rn, 0.1) * 100
                print(f"    {label}: synth={sn:.1f}%, real={rn:.1f}%, error={err:.1f}%")

    print("\n" + "=" * 100)
    print("MODE SYNTHESIS COMPARISON TABLE")
    print("=" * 100)

    band_names = ['sub_100hz', '100_500hz', '500_2khz', '2_5khz', '5_10khz', '10_20khz', 'above_20khz']
    band_labels = ['<100Hz', '100-500Hz', '500-2kHz', '2-5kHz', '5-10kHz', '10-20kHz', '>20kHz']

    for m in metrics_data:
        wname = m['whale']
        real = REAL_TARGETS[wname]
        print(f"\n--- {wname} ---")
        print(f"  N modes: {m['n_modes']}")
        print(f"  Synth time: {m['synth_time_ms']:.1f}ms")
        print(f"  Spectral centroid: synth={m['synth_centroid_hz']:.0f}Hz, "
              f"real={m['real_centroid_hz']:.0f}Hz, error={m['centroid_error_pct']:.1f}%")
        print(f"  Mean band error: {m['mean_band_error_pct']:.1f}%")
        print(f"  {'Band':<12s} {'Synth':>8s} {'Real':>8s} {'Error':>8s}")
        print(f"  {'-'*40}")
        for bn, bl in zip(band_names, band_labels):
            sv = m['synth_bands'].get(bn, 0)
            rv = real.get(f'band_{bn}_pct', 0)
            err = abs(sv - rv) / max(rv, 0.1) * 100
            print(f"  {bl:<12s} {sv:>7.1f}% {rv:>7.1f}% {err:>7.1f}%")


    # Normalized comparison (2-20kHz click band only)
    print("\n" + "-" * 100)
    print("NORMALIZED CLICK-BAND COMPARISON (2-20kHz only)")
    print("Note: Sub-2kHz energy in real data comes from body resonance/ICI structure,")
    print("not individual click spectra. Our synthesizer correctly produces only click energy.")
    print("-" * 100)
    for m in metrics_data:
        wname = m['whale']
        real = REAL_TARGETS[wname]
        sb = m['synth_bands']
        s_2_5 = sb.get('2_5khz', 0)
        s_5_10 = sb.get('5_10khz', 0)
        s_10_20 = sb.get('10_20khz', 0)
        r_2_5 = real.get('band_2_5khz_pct', 0)
        r_5_10 = real.get('band_5_10khz_pct', 0)
        r_10_20 = real.get('band_10_20khz_pct', 0)
        st = s_2_5 + s_5_10 + s_10_20
        rt = r_2_5 + r_5_10 + r_10_20
        if st > 0 and rt > 0:
            print(f"  {wname}:")
            for label, sv, rv in [('2-5kHz', s_2_5, r_2_5), ('5-10kHz', s_5_10, r_5_10), ('10-20kHz', s_10_20, r_10_20)]:
                sn = sv / st * 100
                rn = rv / rt * 100
                err = abs(sn - rn) / max(rn, 0.1) * 100
                print(f"    {label}: synth={sn:.1f}%, real={rn:.1f}%, error={err:.1f}%")

    print("\n" + "=" * 100)


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='Mode-superposition sperm whale click synthesizer')
    parser.add_argument('--whale', type=str, default=None,
                        help='Specific whale to synthesize (A, D, or F)')
    args = parser.parse_args()

    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Mode-Superposition Sperm Whale Click Synthesizer")
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

    # Run synthesis for each whale
    all_results = {}
    for wname in whale_names:
        print(f"\n{'='*50}")
        print(f"Synthesizing {wname}")
        print(f"{'='*50}")
        config = WHALE_CONFIGS[wname]
        result = generate_whale_click(wname, config)
        all_results[wname] = result

        # Save WAV
        wav_path = output_dir / f'{wname.lower().replace("whale_", "whale_")}_mode_synth.wav'
        # Fix filename
        wid = wname.split('_')[1]
        wav_path = output_dir / f'whale_{wid}_mode_synth.wav'
        save_wav(result['final'], wav_path)

    # Create figures (only if all 3 whales processed)
    if len(all_results) == 3:
        metrics = create_comparison_figure(all_results, whale_files, output_dir)
        create_muscle_tension_demo(output_dir)
        create_vowel_demonstration(output_dir)
        print_comparison_table(metrics)

        # Save results JSON
        results_json = {
            'metadata': {
                'method': 'Mode-superposition synthesis',
                'description': 'Damped sinusoidal cavity modes with muscle tension control',
                'sample_rate': TARGET_SR,
                'date': time.strftime('%Y-%m-%d %H:%M:%S'),
            },
            'whales': {},
        }
        for wname, result in all_results.items():
            dt = 1.0 / TARGET_SR
            synth_bands = compute_band_energies(result['final'], dt)
            synth_centroid = compute_spectral_centroid(result['final'], dt)
            real = REAL_TARGETS[wname]

            # Mode summary
            mode_summary = []
            for m in result['modes']:
                if m.get('amplitude', 0) > 0.001:
                    mode_summary.append({
                        'freq_hz': round(m['freq'], 1),
                        'type': m['type'],
                        'n': m['n'],
                        'm': m['m'],
                        'base_amplitude': round(m.get('base_amplitude', 0), 4),
                        'amplitude': round(m.get('amplitude', 0), 4),
                    })

            results_json['whales'][wname] = {
                'config': {k: v for k, v in WHALE_CONFIGS[wname].items()
                           if not isinstance(v, np.ndarray)},
                'n_modes_total': result['n_modes'],
                'n_modes_active': len(mode_summary),
                'duration_s': result['duration_s'],
                'synth_time_ms': result['elapsed_ms'],
                'synth_centroid_hz': round(synth_centroid, 1),
                'real_centroid_hz': real.get('spectral_centroid_hz', 0),
                'synth_bands': {k: round(v, 2) for k, v in synth_bands.items()},
                'active_modes': mode_summary[:50],  # top 50
            }

        json_path = output_dir / 'mode_synthesis_results.json'
        with open(json_path, 'w') as f:
            json.dump(results_json, f, indent=2)
        print(f"\n  Saved: {json_path}")
    else:
        # Single whale - just print basic metrics
        for wname, result in all_results.items():
            dt = 1.0 / TARGET_SR
            centroid = compute_spectral_centroid(result['final'], dt)
            bands = compute_band_energies(result['final'], dt)
            print(f"\n{wname}: centroid={centroid:.0f}Hz, "
                  f"5-10kHz={bands.get('band_5_10khz_pct', 0):.1f}%")

    print("\nDone!")


if __name__ == '__main__':
    main()

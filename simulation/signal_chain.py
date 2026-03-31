#!/usr/bin/env python3
"""
Sperm Whale Signal Chain v2 - Mode-Targeted Source Excitation
=============================================================
Tests different source excitation patterns (impulse, chirp, multimode, noise)
to find which one makes the FDTD cavity output match real whale spectra.

The key insight: real whale clicks have dominant modes at 5-8 kHz (not 12-15 kHz).
The broadband impulse excites all cavity modes equally, but the phonic lips
may preferentially excite certain mode bands.

Usage:
    python signal_chain_v2.py                          # all whales, all sources
    python signal_chain_v2.py --whale A --source chirp  # one whale, one source
    python signal_chain_v2.py --dx 0.02                # fast mode
"""

import argparse
import json
import os
import sys
import time
import warnings
import numpy as np
from pathlib import Path
from scipy.signal import hilbert, find_peaks, resample, butter, filtfilt
from scipy.ndimage import gaussian_filter1d
from scipy.io import wavfile

warnings.filterwarnings('ignore')

# Add parent dir so we can import from sperm_whale_sim_v2
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from sperm_whale_sim_v2 import (
    WhaleHead, broadband_impulse, fdtd_2d_v2,
    MATERIALS, ABSORPTION_COEFFICIENTS,
    TISSUE_WATER, TISSUE_SPERMACETI, TISSUE_JUNK, TISSUE_BONE,
    TISSUE_AIR_SAC, TISSUE_CONNECTIVE, TISSUE_BLUBBER,
    spermaceti_sound_speed, compute_band_energies,
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker

# ============================================================
# OUTPUT DIRECTORY
# ============================================================
OUTPUT_DIR = "/mnt/archive/datasets/whale_communication/analysis/signal_chain"
DSWP_DIR = "/mnt/archive/datasets/whale_communication/DSWP"

# ============================================================
# WHALE CONFIGURATIONS - with mode-targeted parameters
# ============================================================
WHALE_CONFIGS = {
    'Whale_A': {
        'organ_length': 3.49, 'organ_diameter': 1.2,
        'skull_curvature': 1.0, 'junk_length': 2.0,
        'junk_max_diameter': 1.0,
        'case_wall_thickness': 0.05,
        'frontal_sac_width': 0.8,
        'distal_sac_width': 0.3,
        'spermaceti_temp': 35.0,
        'blubber_thickness': 0.15,
        'source_duration_us': 50,  # original broadband
        # Mode-targeted source parameters
        'mode_band': (4000, 10000),   # Hz - where A's dominant modes live
        'source_duration_us_targeted': 800,  # sustained to build resonance
        # Measured dominant modes from modal decomposition
        'measured_modes': [6400, 7400, 16200],
        'measured_ramp_up_ms': 0.25,
    },
    'Whale_D': {
        'organ_length': 3.77, 'organ_diameter': 1.4,
        'skull_curvature': 1.0, 'junk_length': 2.2,
        'junk_max_diameter': 1.1,
        'case_wall_thickness': 0.05,
        'frontal_sac_width': 0.8,
        'distal_sac_width': 0.3,
        'spermaceti_temp': 37.0,
        'blubber_thickness': 0.15,
        'source_duration_us': 50,
        'mode_band': (2000, 8000),
        'source_duration_us_targeted': 1000,  # D has longest ramp-up
        'measured_modes': [2300, 7100],
        'measured_ramp_up_ms': 2.35,
    },
    'Whale_F': {
        'organ_length': 4.80, 'organ_diameter': 1.7,
        'skull_curvature': 1.2, 'junk_length': 2.5,
        'junk_max_diameter': 1.3,
        'case_wall_thickness': 0.05,
        'frontal_sac_width': 0.8,
        'distal_sac_width': 0.3,
        'spermaceti_temp': 37.0,
        'blubber_thickness': 0.15,
        'source_duration_us': 50,
        'mode_band': (3000, 8000),
        'source_duration_us_targeted': 600,
        'measured_modes': [5100, 12600],
        'measured_ramp_up_ms': 0.62,
    },
}

# ============================================================
# EXIT-PATH TISSUE LAYERS (from v1)
# ============================================================
TISSUE_LAYERS = [
    ('case_wall',    0.05, 1.0, 1570),
    ('muscle',       0.12, 1.0, 1570),
    ('junk_lipid',   0.80, 0.5, 1400),
    ('junk_septa',   0.03, 1.0, 1570),
    ('blubber',      0.15, 0.5, 1420),
    ('skin',         0.015, 2.0, 1600),
]

# ============================================================
# OCEAN PROPAGATION PARAMETERS (from v1)
# ============================================================
PROPAGATION = {
    'whale_depth_m': 500,
    'recording_distance_m': 300,
    'surface_temp_c': 27.0,
    'deep_temp_c': 5.0,
    'thermocline_depth_m': 200,
}

# ============================================================
# REAL TARGETS (from whale_voiceprints.json)
# ============================================================
REAL_TARGETS = {
    'Whale_A': {
        'spectral_centroid_hz': 7849,
        'band_5_10khz_pct': 49.5,
        'band_10_20khz_pct': 22.3,
        'band_2_5khz_pct': 19.2,
        'ipi_ms': 5.1,
        'zero_crossing_rate': 0.403,
    },
    'Whale_D': {
        'spectral_centroid_hz': 5693,
        'band_5_10khz_pct': 31.4,
        'band_10_20khz_pct': 17.7,
        'band_2_5khz_pct': 18.7,
        'ipi_ms': 5.5,
        'zero_crossing_rate': 0.266,
    },
    'Whale_F': {
        'spectral_centroid_hz': 5333,
        'band_5_10khz_pct': 24.7,
        'band_10_20khz_pct': 19.4,
        'band_2_5khz_pct': 15.4,
        'ipi_ms': 7.0,
        'zero_crossing_rate': 0.250,
    },
}

REPRESENTATIVE_WAVS = {
    'Whale_A': [1, 2, 3],
    'Whale_D': [100, 101, 102],
    'Whale_F': [500, 501, 502],
}


# ============================================================
# SOURCE TYPES
# ============================================================
def source_impulse(dt, config, duration_override_us=None):
    """Original broadband half-sine impulse."""
    dur = duration_override_us if duration_override_us else config['source_duration_us']
    return broadband_impulse(duration_us=dur, dt=dt, amplitude=1000)


def source_ricker(dt, config, duration_override_us=None):
    """Ricker wavelet centered at mid-band frequency."""
    f_low, f_high = config['mode_band']
    f_center = (f_low + f_high) / 2.0
    # Ricker wavelet: (1 - 2*pi^2*f^2*t^2) * exp(-pi^2*f^2*t^2)
    # Duration: about 3 periods at center frequency
    dur_us = duration_override_us if duration_override_us else config.get('source_duration_us_targeted', 800)
    dur_s = dur_us * 1e-6
    n = max(int(dur_s / dt), 10)
    t = np.arange(n) * dt - dur_s / 2.0
    u = (np.pi * f_center * t) ** 2
    wavelet = 1000.0 * (1 - 2 * u) * np.exp(-u)
    return wavelet


def source_chirp(dt, config, duration_override_us=None):
    """Linear frequency chirp sweeping through the cavity's resonant band."""
    f_low, f_high = config['mode_band']
    dur_us = duration_override_us if duration_override_us else config.get('source_duration_us_targeted', 800)
    dur_s = dur_us * 1e-6
    n = max(int(dur_s / dt), 10)
    t = np.arange(n) * dt
    phase = 2 * np.pi * (f_low * t + (f_high - f_low) / (2 * dur_s) * t**2)
    signal = 1000.0 * np.sin(phase) * np.hanning(n)
    return signal


def source_multimode(dt, config, duration_override_us=None):
    """Superpose sinusoidal bursts at predicted cavity mode frequencies."""
    mode_freqs = config['measured_modes']
    dur_us = duration_override_us if duration_override_us else config.get('source_duration_us_targeted', 800)
    dur_s = dur_us * 1e-6
    n = max(int(dur_s / dt), 10)
    t = np.arange(n) * dt
    signal = np.zeros(n)
    window = np.hanning(n)
    for f in mode_freqs:
        signal += np.sin(2 * np.pi * f * t)
    signal *= window * 1000.0 / max(len(mode_freqs), 1)
    return signal


def source_noise(dt, config, duration_override_us=None):
    """Bandpass-filtered noise burst - excites all modes in the band."""
    f_low, f_high = config['mode_band']
    dur_us = duration_override_us if duration_override_us else config.get('source_duration_us_targeted', 800)
    dur_s = dur_us * 1e-6
    n = max(int(dur_s / dt), 10)
    # Need enough samples for filter to work
    if n < 20:
        n = 20
    np.random.seed(42)  # reproducible
    noise = np.random.randn(n)
    # Bandpass filter
    nyq = 0.5 / dt
    low = f_low / nyq
    high = min(f_high / nyq, 0.99)
    if low >= high or low <= 0:
        # fallback: just use windowed noise
        return 1000.0 * noise / np.max(np.abs(noise)) * np.hanning(n)
    b, a = butter(4, [low, high], btype='band')
    # Pad to avoid edge effects
    pad_len = min(3 * max(len(a), len(b)), n - 1)
    if pad_len < 1:
        pad_len = 1
    try:
        filtered = filtfilt(b, a, noise, padlen=pad_len)
    except Exception:
        filtered = noise
    mx = np.max(np.abs(filtered))
    if mx > 0:
        filtered = filtered / mx
    return 1000.0 * filtered * np.hanning(n)


SOURCE_FUNCTIONS = {
    'impulse': source_impulse,
    'ricker': source_ricker,
    'chirp': source_chirp,
    'multimode': source_multimode,
    'noise': source_noise,
}

SOURCE_TYPES = list(SOURCE_FUNCTIONS.keys())


# ============================================================
# STAGES 1-4 (from v1, with source injection parameterized)
# ============================================================
def stage1_fdtd_cavity(whale_name, config, source_type, dx=0.01, duration_ms=20,
                       duration_override_us=None, quiet=False):
    """Run 2D FDTD simulation with a specified source type."""
    if not quiet:
        print(f"\n  [Stage 1] FDTD Cavity - {source_type} source (dx={dx}m)")

    whale = WhaleHead(
        organ_length=config['organ_length'],
        organ_diameter=config['organ_diameter'],
        skull_curvature=config['skull_curvature'],
        junk_length=config['junk_length'],
        junk_max_diameter=config['junk_max_diameter'],
        case_wall_thickness=config['case_wall_thickness'],
        frontal_sac_width=config['frontal_sac_width'],
        distal_sac_width=config['distal_sac_width'],
        spermaceti_temp=config['spermaceti_temp'],
        blubber_thickness=config['blubber_thickness'],
        name=whale_name,
    )

    rho, c, tissue_map, source_pos, sensor_positions, grid_info = whale.build_grid(dx)
    Nx, Ny = grid_info["Nx"], grid_info["Ny"]

    c_max = float(np.max(c))
    dt = 0.2 * dx / c_max
    n_steps = int(duration_ms / 1000.0 / dt)

    # Generate source signal
    source_fn = SOURCE_FUNCTIONS[source_type]
    source_signal = source_fn(dt, config, duration_override_us=duration_override_us)

    if not quiet:
        print(f"    Grid: {Nx}x{Ny}, dt={dt*1e6:.2f}us, {n_steps} steps")
        print(f"    Source: {source_type}, {len(source_signal)} samples ({len(source_signal)*dt*1e6:.0f}us)")

    # Run FDTD
    t0 = time.time()
    import io
    from contextlib import redirect_stdout
    with redirect_stdout(io.StringIO()):
        sensor_data, final_pressure = fdtd_2d_v2(
            rho, c, tissue_map, source_pos, sensor_positions,
            dx, dt, n_steps, source_signal
        )
    elapsed = time.time() - t0
    if not quiet:
        print(f"    FDTD completed in {elapsed:.1f}s")

    forward_signal = sensor_data[0].astype(np.float64)
    if np.any(np.isnan(forward_signal)) or np.any(np.isinf(forward_signal)):
        forward_signal = np.nan_to_num(forward_signal, nan=0.0, posinf=0.0, neginf=0.0)

    return forward_signal, dt, grid_info


def stage2_exit_path_filter(signal, dt, junk_length=2.0, quiet=False):
    """Apply frequency-domain absorption for tissue exit path."""
    if not quiet:
        print(f"  [Stage 2] Exit-Path Tissue Filter")

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

    filtered = np.fft.irfft(spectrum, n=len(signal))
    return filtered


def stage3_ocean_propagation(signal, dt, propagation=None, quiet=False):
    """Apply Francois-Garrison ocean propagation."""
    if propagation is None:
        propagation = PROPAGATION

    if not quiet:
        print(f"  [Stage 3] Ocean Propagation")

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

    propagated = np.fft.irfft(spectrum, n=len(signal))
    return propagated


def extract_all_features(signal, dt):
    """Extract features from signal."""
    features = {}
    fft_mag = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal), dt)
    power = fft_mag**2
    total_energy = np.sum(power)

    if total_energy > 0:
        features['spectral_centroid_hz'] = float(np.sum(freqs * power) / total_energy)
        bands = [
            ('band_sub_100hz', 0, 100),
            ('band_100_500hz', 100, 500),
            ('band_500_2khz', 500, 2000),
            ('band_2_5khz_pct', 2000, 5000),
            ('band_5_10khz_pct', 5000, 10000),
            ('band_10_20khz_pct', 10000, 20000),
            ('band_above_20khz', 20000, freqs[-1] + 1),
        ]
        for key, f_low, f_high in bands:
            mask = (freqs >= f_low) & (freqs < f_high)
            features[key] = float(np.sum(power[mask]) / total_energy * 100.0)
    else:
        features['spectral_centroid_hz'] = 0.0
        for key in ['band_sub_100hz', 'band_100_500hz', 'band_500_2khz',
                     'band_2_5khz_pct', 'band_5_10khz_pct', 'band_10_20khz_pct', 'band_above_20khz']:
            features[key] = 0.0

    # IPI
    analytic = hilbert(signal.astype(np.float64))
    envelope = np.abs(analytic)
    sigma_samples = max(1, int(0.5e-3 / dt))
    envelope_smooth = gaussian_filter1d(envelope, sigma=sigma_samples)
    min_distance_samples = max(1, int(2.0e-3 / dt))
    threshold = np.max(envelope_smooth) * 0.1
    peaks, _ = find_peaks(envelope_smooth, distance=min_distance_samples, height=threshold)
    peak_times_ms = peaks * dt * 1000.0
    if len(peaks) >= 2:
        ipis = np.diff(peak_times_ms)
        features['ipi_ms'] = float(np.median(ipis))
    else:
        features['ipi_ms'] = 0.0

    features['n_peaks'] = len(peaks)
    if len(signal) > 1:
        crossings = np.sum(np.abs(np.diff(np.sign(signal))) > 0)
        features['zero_crossing_rate'] = float(crossings / (len(signal) - 1))
    else:
        features['zero_crossing_rate'] = 0.0
    return features


def load_real_wav_click(whale_name):
    """Load representative WAV and extract loudest click."""
    wav_ids = REPRESENTATIVE_WAVS.get(whale_name, [1])
    for wav_id in wav_ids:
        wav_path = os.path.join(DSWP_DIR, f"{wav_id}.wav")
        if os.path.exists(wav_path):
            try:
                sr, data = wavfile.read(wav_path)
                if data.dtype == np.int16:
                    data = data.astype(np.float64) / 32768.0
                elif data.dtype == np.float32:
                    data = data.astype(np.float64)
                if data.ndim > 1:
                    data = data[:, 0]
                window_samples = int(0.020 * sr)
                if len(data) < window_samples:
                    continue
                energy = np.convolve(data**2, np.ones(window_samples) / window_samples, mode='valid')
                peak_idx = np.argmax(energy)
                start = max(0, peak_idx - window_samples // 4)
                end = min(len(data), start + window_samples)
                return data[start:end], sr
            except Exception:
                continue
    return None, None


def _get_real_band_energies(whale_name):
    """Real band energies from voiceprints."""
    voiceprint_bands = {
        'Whale_A': {
            'band_sub_100hz': 0.50, 'band_100_500hz': 0.75, 'band_500_2khz': 5.50,
            'band_2_5khz_pct': 19.21, 'band_5_10khz_pct': 49.48,
            'band_10_20khz_pct': 22.29, 'band_above_20khz': 2.28,
        },
        'Whale_D': {
            'band_sub_100hz': 16.11, 'band_100_500hz': 1.52, 'band_500_2khz': 12.17,
            'band_2_5khz_pct': 18.74, 'band_5_10khz_pct': 31.41,
            'band_10_20khz_pct': 17.72, 'band_above_20khz': 1.14,
        },
        'Whale_F': {
            'band_sub_100hz': 29.48, 'band_100_500hz': 2.80, 'band_500_2khz': 7.11,
            'band_2_5khz_pct': 15.39, 'band_5_10khz_pct': 24.70,
            'band_10_20khz_pct': 19.41, 'band_above_20khz': 1.10,
        },
    }
    return voiceprint_bands.get(whale_name, {})


def save_synthetic_wav(signal, dt, whale_name, source_type, output_dir):
    """Resample to 44.1kHz and save as WAV."""
    target_sr = 44100
    sim_sr = 1.0 / dt
    n_target = int(len(signal) * target_sr / sim_sr)
    if n_target < 100:
        pad_factor = max(2, int(44100 * 0.020 / n_target))
        padded = np.zeros(len(signal) * pad_factor)
        padded[:len(signal)] = signal
        n_target = int(len(padded) * target_sr / sim_sr)
        resampled = resample(padded, n_target)
    else:
        resampled = resample(signal, n_target)
    mx = np.max(np.abs(resampled))
    if mx > 0:
        resampled = resampled / mx * 0.95
    wav_data = (resampled * 32767).astype(np.int16)
    letter = whale_name.split('_')[1] if '_' in whale_name else whale_name
    path = os.path.join(output_dir, f"whale_{letter}_{source_type}_synthetic.wav")
    wavfile.write(path, target_sr, wav_data)
    return path


# ============================================================
# RUN ONE CONFIGURATION
# ============================================================
def run_one(whale_name, config, source_type, dx=0.01, duration_ms=20,
            duration_override_us=None, quiet=False):
    """Run full signal chain for one whale + source combination."""
    raw, dt, grid_info = stage1_fdtd_cavity(
        whale_name, config, source_type, dx=dx, duration_ms=duration_ms,
        duration_override_us=duration_override_us, quiet=quiet
    )
    exit_sig = stage2_exit_path_filter(raw, dt, junk_length=config['junk_length'], quiet=True)
    ocean_sig = stage3_ocean_propagation(exit_sig, dt, quiet=True)
    features = extract_all_features(ocean_sig, dt)
    raw_features = extract_all_features(raw, dt)

    return {
        'signal_raw': raw,
        'signal_final': ocean_sig,
        'dt': dt,
        'features_raw': raw_features,
        'features_final': features,
    }


# ============================================================
# SOURCE COMPARISON FIGURE
# ============================================================
def create_source_comparison_figure(results_grid, output_dir):
    """
    results_grid[whale_name][source_type] = run result dict
    Rows: whales (A, D, F)
    Columns: source types (impulse, ricker, chirp, multimode, noise)
    Each cell: spectrum overlay (sim vs real centroid markers)
    Bottom row: centroid error bar chart
    """
    whale_names = list(results_grid.keys())
    src_types = SOURCE_TYPES
    n_whales = len(whale_names)
    n_sources = len(src_types)

    fig = plt.figure(figsize=(24, 20))
    # n_whales rows of spectra + 1 row for bar chart
    gs = GridSpec(n_whales + 1, n_sources, figure=fig, hspace=0.4, wspace=0.3,
                  left=0.06, right=0.97, top=0.94, bottom=0.06)

    colors_whale = {'Whale_A': '#e74c3c', 'Whale_D': '#3498db', 'Whale_F': '#2ecc71'}

    # Spectrum panels
    for wi, wname in enumerate(whale_names):
        real_centroid = REAL_TARGETS[wname]['spectral_centroid_hz']
        letter = wname.split('_')[1]

        for si, stype in enumerate(src_types):
            ax = fig.add_subplot(gs[wi, si])
            r = results_grid[wname].get(stype)
            if r is None:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                continue

            sig = r['signal_final']
            dt_val = r['dt']
            fft_mag = np.abs(np.fft.rfft(sig))
            freqs = np.fft.rfftfreq(len(sig), dt_val)

            if len(fft_mag) > 100:
                fft_smooth = gaussian_filter1d(fft_mag, sigma=max(1, len(fft_mag) // 100))
            else:
                fft_smooth = fft_mag.copy()
            mx = np.max(fft_smooth)
            if mx > 0:
                fft_smooth /= mx

            color = colors_whale.get(wname, '#666')
            ax.semilogy(freqs / 1000, fft_smooth + 1e-6, color=color, linewidth=1, label='Sim')

            sim_centroid = r['features_final']['spectral_centroid_hz']
            ax.axvline(real_centroid / 1000, color='black', linestyle='--', alpha=0.6,
                       linewidth=0.8, label=f'Real C={real_centroid:.0f}Hz')
            ax.axvline(sim_centroid / 1000, color=color, linestyle=':', alpha=0.8,
                       linewidth=0.8, label=f'Sim C={sim_centroid:.0f}Hz')

            ax.set_xlim(0, 25)
            ax.set_ylim(1e-4, 2)
            ax.tick_params(labelsize=7)

            if wi == 0:
                ax.set_title(stype.upper(), fontsize=11, fontweight='bold')
            if si == 0:
                ax.set_ylabel(f'Whale {letter}', fontsize=10, fontweight='bold')
            if wi == n_whales - 1:
                ax.set_xlabel('Freq (kHz)', fontsize=8)

            # Error annotation
            err_hz = sim_centroid - real_centroid
            err_pct = err_hz / real_centroid * 100
            ax.text(0.98, 0.95, f'Err: {err_pct:+.0f}%\n({err_hz:+.0f}Hz)',
                    transform=ax.transAxes, ha='right', va='top', fontsize=7,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

            if wi == 0 and si == 0:
                ax.legend(fontsize=6, loc='lower left')

    # Bottom row: centroid error bar chart
    ax_bar = fig.add_subplot(gs[n_whales, :])
    x = np.arange(n_sources)
    width = 0.25
    offsets = np.linspace(-width, width, n_whales)

    for wi, wname in enumerate(whale_names):
        real_c = REAL_TARGETS[wname]['spectral_centroid_hz']
        letter = wname.split('_')[1]
        errors = []
        for stype in src_types:
            r = results_grid[wname].get(stype)
            if r:
                sim_c = r['features_final']['spectral_centroid_hz']
                errors.append((sim_c - real_c) / real_c * 100)
            else:
                errors.append(0)
        color = colors_whale.get(wname, '#666')
        bars = ax_bar.bar(x + offsets[wi], errors, width * 0.9,
                          label=f'Whale {letter}', color=color, alpha=0.8)
        # Value labels
        for bar, val in zip(bars, errors):
            ypos = bar.get_height()
            ax_bar.text(bar.get_x() + bar.get_width() / 2, ypos,
                        f'{val:+.0f}%', ha='center', va='bottom' if ypos >= 0 else 'top',
                        fontsize=6)

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([s.upper() for s in src_types], fontsize=9)
    ax_bar.set_ylabel('Centroid Error (%)', fontsize=10)
    ax_bar.set_title('Spectral Centroid Error by Source Type (0% = perfect match)', fontsize=12, fontweight='bold')
    ax_bar.axhline(0, color='black', linewidth=0.5)
    ax_bar.legend(fontsize=9)
    ax_bar.grid(axis='y', alpha=0.3)

    fig.suptitle('FDTD Source Excitation Comparison - Which Source Matches Real Whale Spectra?',
                 fontsize=14, fontweight='bold')

    path = os.path.join(output_dir, "source_comparison.png")
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved: {path}")
    return path


# ============================================================
# DURATION SWEEP FIGURE
# ============================================================
def create_duration_sweep_figure(sweep_results, output_dir):
    """
    sweep_results[whale_name] = list of (duration_us, features_dict)
    """
    whale_names = list(sweep_results.keys())
    colors_whale = {'Whale_A': '#e74c3c', 'Whale_D': '#3498db', 'Whale_F': '#2ecc71'}

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Panel 1: Centroid vs duration
    ax = axes[0, 0]
    for wname in whale_names:
        letter = wname.split('_')[1]
        color = colors_whale.get(wname, '#666')
        durations = [d for d, _ in sweep_results[wname]]
        centroids = [f['spectral_centroid_hz'] for _, f in sweep_results[wname]]
        ax.plot(durations, centroids, 'o-', color=color, label=f'Whale {letter}', linewidth=2)
        real_c = REAL_TARGETS[wname]['spectral_centroid_hz']
        ax.axhline(real_c, color=color, linestyle='--', alpha=0.4, linewidth=1)
        ax.text(durations[-1] + 50, real_c, f'{letter} real', fontsize=7, color=color, va='center')
    ax.set_xlabel('Source Duration (us)', fontsize=10)
    ax.set_ylabel('Spectral Centroid (Hz)', fontsize=10)
    ax.set_title('Spectral Centroid vs Source Duration', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Panel 2: Centroid error vs duration
    ax = axes[0, 1]
    for wname in whale_names:
        letter = wname.split('_')[1]
        color = colors_whale.get(wname, '#666')
        real_c = REAL_TARGETS[wname]['spectral_centroid_hz']
        durations = [d for d, _ in sweep_results[wname]]
        errors = [abs(f['spectral_centroid_hz'] - real_c) / real_c * 100
                  for _, f in sweep_results[wname]]
        ax.plot(durations, errors, 'o-', color=color, label=f'Whale {letter}', linewidth=2)
        # Mark minimum
        min_idx = np.argmin(errors)
        ax.plot(durations[min_idx], errors[min_idx], 's', color=color, markersize=12, zorder=5)
        ax.annotate(f'{durations[min_idx]}us\n{errors[min_idx]:.0f}%',
                    (durations[min_idx], errors[min_idx]),
                    textcoords="offset points", xytext=(10, 10), fontsize=8,
                    arrowprops=dict(arrowstyle='->', color=color))
    ax.set_xlabel('Source Duration (us)', fontsize=10)
    ax.set_ylabel('|Centroid Error| (%)', fontsize=10)
    ax.set_title('Centroid Error vs Source Duration (lower = better)', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Panel 3: 5-10kHz band energy vs duration
    ax = axes[1, 0]
    for wname in whale_names:
        letter = wname.split('_')[1]
        color = colors_whale.get(wname, '#666')
        durations = [d for d, _ in sweep_results[wname]]
        band510 = [f['band_5_10khz_pct'] for _, f in sweep_results[wname]]
        ax.plot(durations, band510, 'o-', color=color, label=f'Whale {letter}', linewidth=2)
        real_b = REAL_TARGETS[wname]['band_5_10khz_pct']
        ax.axhline(real_b, color=color, linestyle='--', alpha=0.4, linewidth=1)
    ax.set_xlabel('Source Duration (us)', fontsize=10)
    ax.set_ylabel('5-10kHz Band Energy (%)', fontsize=10)
    ax.set_title('5-10kHz Band Energy vs Source Duration', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Panel 4: Summary - optimal durations
    ax = axes[1, 1]
    ax.axis('off')
    summary_lines = ['OPTIMAL SOURCE DURATIONS', '=' * 40, '']
    for wname in whale_names:
        letter = wname.split('_')[1]
        real_c = REAL_TARGETS[wname]['spectral_centroid_hz']
        durations = [d for d, _ in sweep_results[wname]]
        errors = [abs(f['spectral_centroid_hz'] - real_c) / real_c * 100
                  for _, f in sweep_results[wname]]
        min_idx = np.argmin(errors)
        best_dur = durations[min_idx]
        best_err = errors[min_idx]
        best_centroid = sweep_results[wname][min_idx][1]['spectral_centroid_hz']
        measured_ramp = WHALE_CONFIGS[wname]['measured_ramp_up_ms']
        summary_lines.append(f'Whale {letter}:')
        summary_lines.append(f'  Best duration: {best_dur} us')
        summary_lines.append(f'  Centroid: {best_centroid:.0f} Hz (real: {real_c:.0f} Hz)')
        summary_lines.append(f'  Error: {best_err:.1f}%')
        summary_lines.append(f'  Measured ramp-up: {measured_ramp} ms')
        summary_lines.append('')

    ax.text(0.1, 0.95, '\n'.join(summary_lines), transform=ax.transAxes,
            fontsize=10, family='monospace', va='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    fig.suptitle('Duration Sweep - Finding Optimal Source Duration per Whale',
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    path = os.path.join(output_dir, "duration_sweep.png")
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")
    return path


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Signal Chain v2 - Mode-Targeted Source")
    parser.add_argument("--whale", type=str, default="all",
                        help="Which whale: A, D, F, or all")
    parser.add_argument("--source", type=str, default="all",
                        help="Source type: impulse, ricker, chirp, multimode, noise, or all")
    parser.add_argument("--dx", type=float, default=0.02,
                        help="Grid spacing in meters (default: 0.02 for speed)")
    parser.add_argument("--duration", type=float, default=20.0,
                        help="Simulation duration in ms")
    parser.add_argument("--no-sweep", action='store_true',
                        help="Skip the duration sweep")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Select whales
    if args.whale.lower() == 'all':
        whale_names = list(WHALE_CONFIGS.keys())
    else:
        letter = args.whale.upper()
        whale_names = [f'Whale_{letter}']

    # Select source types
    if args.source.lower() == 'all':
        source_types = SOURCE_TYPES
    else:
        source_types = [args.source.lower()]

    print("=" * 70)
    print("SIGNAL CHAIN v2 - MODE-TARGETED SOURCE COMPARISON")
    print("=" * 70)
    print(f"  Whales: {', '.join(whale_names)}")
    print(f"  Sources: {', '.join(source_types)}")
    print(f"  dx={args.dx}m, duration={args.duration}ms")
    print(f"  Output: {OUTPUT_DIR}")
    print()

    t_start = time.time()

    # ==============================
    # PHASE 1: Source type comparison
    # ==============================
    print("\n" + "=" * 70)
    print("PHASE 1: SOURCE TYPE COMPARISON")
    print("=" * 70)

    results_grid = {}  # results_grid[whale][source] = result
    all_json = {}

    for wname in whale_names:
        config = WHALE_CONFIGS[wname]
        letter = wname.split('_')[1]
        results_grid[wname] = {}
        all_json[wname] = {}

        print(f"\n{'#' * 60}")
        print(f"# {wname} (L={config['organ_length']}m, mode_band={config['mode_band']})")
        print(f"{'#' * 60}")

        for stype in source_types:
            print(f"\n  --- {stype.upper()} ---")
            r = run_one(wname, config, stype, dx=args.dx, duration_ms=args.duration)
            results_grid[wname][stype] = r

            sim_c = r['features_final']['spectral_centroid_hz']
            real_c = REAL_TARGETS[wname]['spectral_centroid_hz']
            err = (sim_c - real_c) / real_c * 100

            print(f"    Centroid: {sim_c:.0f} Hz (real: {real_c:.0f} Hz, error: {err:+.1f}%)")
            print(f"    5-10kHz: {r['features_final']['band_5_10khz_pct']:.1f}% "
                  f"(real: {REAL_TARGETS[wname]['band_5_10khz_pct']:.1f}%)")
            print(f"    IPI: {r['features_final']['ipi_ms']:.1f} ms "
                  f"(real: {REAL_TARGETS[wname]['ipi_ms']:.1f} ms)")

            # Save WAV for this combo
            wav_path = save_synthetic_wav(r['signal_final'], r['dt'], wname, stype, OUTPUT_DIR)

            # JSON-safe features
            all_json[wname][stype] = {
                'features_raw': {k: float(v) if isinstance(v, (float, np.floating)) else v
                                 for k, v in r['features_raw'].items()},
                'features_final': {k: float(v) if isinstance(v, (float, np.floating)) else v
                                   for k, v in r['features_final'].items()},
                'centroid_error_pct': float(err),
                'wav_path': wav_path,
            }

    # ==============================
    # SUMMARY TABLE
    # ==============================
    print("\n\n" + "=" * 90)
    print("SOURCE COMPARISON SUMMARY")
    print("=" * 90)
    print(f"\n{'':>12s}", end='')
    for stype in source_types:
        print(f"  {stype:>12s}", end='')
    print(f"  {'REAL':>10s}")
    print("-" * 90)

    for wname in whale_names:
        letter = wname.split('_')[1]
        real_c = REAL_TARGETS[wname]['spectral_centroid_hz']

        # Centroid
        print(f"  {letter} Centroid", end='')
        for stype in source_types:
            r = results_grid[wname].get(stype)
            if r:
                c = r['features_final']['spectral_centroid_hz']
                print(f"  {c:>10.0f}Hz", end='')
            else:
                print(f"  {'---':>12s}", end='')
        print(f"  {real_c:>8.0f}Hz")

        # Error
        print(f"  {letter} Error  ", end='')
        for stype in source_types:
            r = results_grid[wname].get(stype)
            if r:
                c = r['features_final']['spectral_centroid_hz']
                err = (c - real_c) / real_c * 100
                print(f"  {err:>+10.0f}%", end='')
            else:
                print(f"  {'---':>12s}", end='')
        print()

        # 5-10kHz
        real_b = REAL_TARGETS[wname]['band_5_10khz_pct']
        print(f"  {letter} 5-10kHz", end='')
        for stype in source_types:
            r = results_grid[wname].get(stype)
            if r:
                b = r['features_final']['band_5_10khz_pct']
                print(f"  {b:>10.1f}%", end='')
            else:
                print(f"  {'---':>12s}", end='')
        print(f"  {real_b:>8.1f}%")
        print()

    # Find best source per whale
    print("\nBEST SOURCE PER WHALE:")
    best_sources = {}
    for wname in whale_names:
        letter = wname.split('_')[1]
        real_c = REAL_TARGETS[wname]['spectral_centroid_hz']
        best_err = float('inf')
        best_src = None
        for stype in source_types:
            r = results_grid[wname].get(stype)
            if r:
                c = r['features_final']['spectral_centroid_hz']
                err = abs(c - real_c) / real_c * 100
                if err < best_err:
                    best_err = err
                    best_src = stype
        best_sources[wname] = best_src
        print(f"  Whale {letter}: {best_src} (error: {best_err:.1f}%)")

    # ==============================
    # PHASE 2: Duration sweep for best source
    # ==============================
    sweep_results = {}
    if not args.no_sweep and len(source_types) > 1:
        print("\n\n" + "=" * 70)
        print("PHASE 2: DURATION SWEEP (best source type per whale)")
        print("=" * 70)

        durations_us = list(range(100, 2200, 200))  # 100 to 2000 in 200us steps

        for wname in whale_names:
            config = WHALE_CONFIGS[wname]
            letter = wname.split('_')[1]
            best_src = best_sources[wname]
            sweep_results[wname] = []

            print(f"\n  Whale {letter} - sweeping {best_src} duration: {durations_us[0]}-{durations_us[-1]} us")

            for dur in durations_us:
                r = run_one(wname, config, best_src, dx=args.dx, duration_ms=args.duration,
                            duration_override_us=dur, quiet=True)
                sweep_results[wname].append((dur, r['features_final']))
                c = r['features_final']['spectral_centroid_hz']
                real_c = REAL_TARGETS[wname]['spectral_centroid_hz']
                err = (c - real_c) / real_c * 100
                print(f"    {dur:5d}us -> centroid={c:8.0f}Hz (err={err:+6.1f}%), "
                      f"5-10kHz={r['features_final']['band_5_10khz_pct']:.1f}%")

            # Find optimal
            real_c = REAL_TARGETS[wname]['spectral_centroid_hz']
            errors = [abs(f['spectral_centroid_hz'] - real_c) / real_c * 100
                      for _, f in sweep_results[wname]]
            min_idx = np.argmin(errors)
            best_dur = sweep_results[wname][min_idx][0]
            print(f"    >>> OPTIMAL: {best_dur}us (error: {errors[min_idx]:.1f}%)")

    elif not args.no_sweep:
        # Single source - sweep it
        print("\n\n" + "=" * 70)
        print(f"PHASE 2: DURATION SWEEP ({source_types[0]})")
        print("=" * 70)

        durations_us = list(range(100, 2200, 200))
        for wname in whale_names:
            config = WHALE_CONFIGS[wname]
            letter = wname.split('_')[1]
            sweep_results[wname] = []

            print(f"\n  Whale {letter} - sweeping {source_types[0]} duration")
            for dur in durations_us:
                r = run_one(wname, config, source_types[0], dx=args.dx, duration_ms=args.duration,
                            duration_override_us=dur, quiet=True)
                sweep_results[wname].append((dur, r['features_final']))
                c = r['features_final']['spectral_centroid_hz']
                real_c = REAL_TARGETS[wname]['spectral_centroid_hz']
                err = (c - real_c) / real_c * 100
                print(f"    {dur:5d}us -> centroid={c:8.0f}Hz (err={err:+6.1f}%)")

    total_time = time.time() - t_start
    print(f"\n\nTotal time: {total_time:.1f}s")

    # ==============================
    # GENERATE FIGURES
    # ==============================
    print("\n[Generating figures...]")

    if len(source_types) > 1:
        try:
            create_source_comparison_figure(results_grid, OUTPUT_DIR)
        except Exception as e:
            print(f"WARNING: source comparison figure failed: {e}")
            import traceback
            traceback.print_exc()

    if sweep_results:
        try:
            create_duration_sweep_figure(sweep_results, OUTPUT_DIR)
        except Exception as e:
            print(f"WARNING: duration sweep figure failed: {e}")
            import traceback
            traceback.print_exc()

    # ==============================
    # SAVE JSON
    # ==============================
    # Add sweep results to JSON
    json_output = {
        'source_comparison': {},
        'duration_sweep': {},
        'best_sources': {wname: best_sources.get(wname, 'unknown') for wname in whale_names},
        'parameters': {
            'dx': args.dx,
            'duration_ms': args.duration,
            'source_types': source_types,
        },
    }

    for wname in whale_names:
        json_output['source_comparison'][wname] = all_json.get(wname, {})

    for wname, sweeps in sweep_results.items():
        json_output['duration_sweep'][wname] = {
            'source_type': best_sources.get(wname, source_types[0]),
            'results': [
                {
                    'duration_us': dur,
                    'spectral_centroid_hz': float(f['spectral_centroid_hz']),
                    'band_5_10khz_pct': float(f['band_5_10khz_pct']),
                    'band_2_5khz_pct': float(f['band_2_5khz_pct']),
                    'band_10_20khz_pct': float(f['band_10_20khz_pct']),
                    'ipi_ms': float(f['ipi_ms']),
                }
                for dur, f in sweeps
            ],
        }

    json_path = os.path.join(OUTPUT_DIR, "signal_chain_v2_results.json")
    with open(json_path, 'w') as fp:
        json.dump(json_output, fp, indent=2, default=str)
    print(f"\nJSON results: {json_path}")

    print(f"\nAll outputs in: {OUTPUT_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()

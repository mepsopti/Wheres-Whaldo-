#!/usr/bin/env python3
"""
Digital Waveguide Model V2 - Sperm Whale Spermaceti Cavity
with Acoustic Valve Mechanism (Huggenberger 2014)
===========================================================

V2 adds the acoustic valve physics from Huggenberger 2014:

1. Connecting Acoustic Window (CAW):
   The right nasal passage acts as an impedance barrier at a point midway
   along the spermaceti organ. When air-filled, sound stays in the cavity
   (coda mode). When collapsed, sound leaks through into the junk
   (echolocation mode). This is modeled as a second exit tap on the
   forward delay line.

2. Frontal Sac Shape (curvature controlled by maxillonasalis):
   - Contracted muscle -> compressed sac against skull amphitheatre
     -> parabolic curvature -> focused beam, higher R_frontal (~0.99)
   - Relaxed muscle -> inflated sac -> flat shape
     -> poor focusing, lower R_frontal (~0.90)

3. Muscle tension couples CAW state and frontal sac shape:
   The maxillonasalis controls BOTH simultaneously - high tension means
   collapsed nasal passage (high T_caw) AND curved frontal sac (high R_f).

Architecture:
                Distal Sac          CAW leak point        Frontal Sac
                (R_d)               (T_caw)               (R_f, curvature)
                  |                    |                       |
Junk Exit <--(1-R_d)--[ Forward Delay ]--(-T_caw)-->Junk     |
                  |    [   Line      ]                        |
                  |    [ Backward    ]                        |
Lips --------->   +    [   Line      ]   <--(R_f)------------ +
                                                          Skull backing

Click mode presets derive from Huggenberger 2014 Table 1:
  - Echolocation: high T_caw, parabolic sac, few strong pulses, ~15kHz centroid
  - Coda: low T_caw, flat sac, many decaying pulses, ~5kHz centroid
  - Creak: medium T_caw, medium curvature, rapid short bursts, ~15kHz
  - Slow click: very low T_caw, flat sac, faint pulses, ~3kHz centroid

Usage:
    python waveguide_v2.py                # Whale D all 4 click types
    python waveguide_v2.py --whale A      # just Whale A
    python waveguide_v2.py --all-codas    # coda WAVs for all 3 whales

Output:
    waveguide_v2.py (this file)
    click_types_comparison.png (20x16)
    whale_D_echolocation.wav, whale_D_coda.wav, whale_D_creak.wav, whale_D_slow.wav
    whale_{A,D,F}_coda_v3.wav
    waveguide_v2_results.json
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
from scipy.signal import resample, butter, filtfilt, iirnotch, lfilter

warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker

# ============================================================
# PATHS
# ============================================================
OUTPUT_DIR = Path("/mnt/archive/datasets/whale_communication/analysis/waveguide")
MODE_SYNTH_DIR = Path("/mnt/archive/datasets/whale_communication/analysis/mode_synthesis")
DSWP_DIR = Path("/mnt/archive/datasets/whale_communication/DSWP")
CODA_CSV = Path("/mnt/archive/datasets/whale_communication/sw-combinatoriality/data/DominicaCodas.csv")

TARGET_SR = 44100

# ============================================================
# TISSUE LAYERS (exit path through junk + blubber + skin)
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
# REAL TARGETS (from whale_voiceprints.json - coda click means)
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
# CLICK MODE PRESETS (Huggenberger 2014)
# ============================================================
# The maxillonasalis muscle controls BOTH the CAW state and the
# frontal sac shape. High muscle_tension = collapsed nasal passage
# (high T_caw) AND compressed/curved frontal sac (high R_f).
CLICK_MODES = {
    'echolocation': {
        'T_caw': 0.35,               # CAW open - energy exits early through junk
        'frontal_sac_curvature': 0.9, # parabolic mirror - focused return beam
        'R_distal': 0.95,            # tighter distal sac
        'R_frontal': 0.98,           # high R, good parabolic mirror
        'muscle_tension': 0.8,       # strong maxillonasalis contraction
        'source_duration_us': 150,   # very short impulsive burst
        'lip_freq': 4000,            # higher frequency excitation
        'lip_peak_hz': 20000,        # spectral peak tuned very high for ~15kHz centroid
        'lip_bandwidth_octaves': 1.3, # narrow bandwidth - focused beam
        'low_freq_floor': 0.001,     # suppress low frequencies very strongly
        'junk_length_scale': 0.3,    # CAW path is shorter through junk
    },
    'coda': {
        'T_caw': 0.05,               # CAW mostly closed - energy stays in cavity
        'frontal_sac_curvature': 0.3, # flat mirror - broad, unfocused
        'R_distal': 0.97,            # high R - energy recirculates
        'R_frontal': 0.92,           # lower R, flat mirror is less efficient
        'muscle_tension': 0.3,       # relaxed maxillonasalis
        'source_duration_us': 800,   # sustained buzz - more energy into cavity
        'lip_freq': 700,             # lower frequency excitation
        'lip_peak_hz': 3000,         # spectral peak for ~5kHz centroid
        'lip_bandwidth_octaves': 1.3, # narrower - energy concentrated low
        'low_freq_floor': 0.15,      # substantial low frequency floor
        'junk_length_scale': 1.0,    # full junk path from distal sac
    },
    'creak': {
        'T_caw': 0.20,               # moderate CAW opening
        'frontal_sac_curvature': 0.6, # partially curved
        'R_distal': 0.96,
        'R_frontal': 0.95,
        'muscle_tension': 0.6,
        'source_duration_us': 80,    # very short burst
        'lip_freq': 4000,            # higher frequency
        'lip_peak_hz': 20000,        # high spectral peak like echo
        'lip_bandwidth_octaves': 1.4,
        'low_freq_floor': 0.002,     # suppress low frequencies
        'junk_length_scale': 0.5,    # mixed path
    },
    'slow_click': {
        'T_caw': 0.02,               # CAW nearly sealed
        'frontal_sac_curvature': 0.2, # flat, relaxed sac
        'R_distal': 0.80,            # low R - cavity poorly reflective
        'R_frontal': 0.75,           # very low R - flat mirror, poor reflector
        'muscle_tension': 0.2,       # very relaxed
        'source_duration_us': 2000,  # very long sustained excitation
        'lip_freq': 300,             # very low frequency
        'lip_peak_hz': 1200,         # very low spectral peak for ~3kHz centroid
        'lip_bandwidth_octaves': 0.8, # extremely narrow
        'low_freq_floor': 0.40,      # very high low-freq floor
        'junk_length_scale': 1.2,    # extra tissue filtering (longer path)
    },
}

# ============================================================
# WHALE CONFIGURATIONS
# ============================================================
WHALE_CONFIGS = {
    'Whale_A': {
        'organ_length': 3.50, 'organ_diameter': 1.20,
        'spermaceti_c': 1370,
        'muscle_tensions': [0.25, 0.15, 0.05, 0.05, 0.30, 0.50],
        'muscle_band_freqs': [16000, 12000, 9000, 7000, 5000, 3500],
        'junk_length': 2.0,
        'R_distal': 0.85, 'R_frontal': 0.90,
        'wall_loss': 0.9997,
        'lip_freq': 2000, 'lip_jitter': 0.25,
        'source_duration_us': 500,
        'source_bandwidth': 'broadband',
        'lip_peak_hz': 8000,
        'lip_bandwidth_octaves': 2.0,
        'low_freq_floor': 0.01,
        'body_length_m': 10.0, 'body_mass_kg': 12000,
        'body_resonance_scale': 0.02,
        'coda_icis_ms': [400, 400, 400, 1200, 400, 400, 400, 1200],
    },
    'Whale_D': {
        'organ_length': 3.80, 'organ_diameter': 1.40,
        'spermaceti_c': 1370,
        'muscle_tensions': [0.75, 0.60, 0.55, 0.45, 0.40, 0.50],
        'muscle_band_freqs': [16000, 12000, 9000, 7000, 5000, 3500],
        'junk_length': 2.2,
        'R_distal': 0.83, 'R_frontal': 0.88,
        'wall_loss': 0.9996,
        'lip_freq': 1500, 'lip_jitter': 0.30,
        'source_duration_us': 600,
        'source_bandwidth': 'broadband',
        'lip_peak_hz': 5000,
        'lip_bandwidth_octaves': 2.5,
        'low_freq_floor': 0.03,
        'body_length_m': 12.0, 'body_mass_kg': 20000,
        'body_resonance_scale': 0.10,
        'coda_icis_ms': [350, 350, 350, 1000, 350, 350, 350, 1000],
    },
    'Whale_F': {
        'organ_length': 4.80, 'organ_diameter': 1.70,
        'spermaceti_c': 1370,
        'muscle_tensions': [0.80, 0.70, 0.65, 0.60, 0.30, 0.35],
        'muscle_band_freqs': [16000, 12000, 9000, 7000, 5000, 3500],
        'junk_length': 2.5,
        'R_distal': 0.80, 'R_frontal': 0.85,
        'wall_loss': 0.9995,
        'lip_freq': 1200, 'lip_jitter': 0.35,
        'source_duration_us': 700,
        'source_bandwidth': 'broadband',
        'lip_peak_hz': 4500,
        'lip_bandwidth_octaves': 2.5,
        'low_freq_floor': 0.05,
        'body_length_m': 15.0, 'body_mass_kg': 40000,
        'body_resonance_scale': 0.30,
        'coda_icis_ms': [300, 300, 300, 900, 300, 300, 300, 900],
    },
}


# ============================================================
# MUSCLE BAND NOTCH FILTER
# ============================================================
class MuscleBand:
    """A single muscle band modeled as a notch filter at a specific cavity position."""

    def __init__(self, position, freq, tension, sample_rate=44100, Q=10):
        self.position = position
        self.freq = freq
        self.tension = tension
        self.Q = Q
        effective_Q = Q / max(tension, 0.01)
        if freq < sample_rate / 2:
            self.b, self.a = iirnotch(freq, effective_Q, fs=sample_rate)
        else:
            self.b, self.a = np.array([1.0]), np.array([1.0])
        self.z_fwd = np.zeros(max(len(self.a), len(self.b)) - 1)
        self.z_bwd = np.zeros(max(len(self.a), len(self.b)) - 1)

    def filter_sample(self, x, direction='forward'):
        z = self.z_fwd if direction == 'forward' else self.z_bwd
        y = self.b[0] * x + z[0]
        for i in range(len(z) - 1):
            z[i] = self.b[i+1] * x - self.a[i+1] * y + z[i+1]
        z[-1] = self.b[-1] * x - self.a[-1] * y
        return y


# ============================================================
# DIGITAL WAVEGUIDE V2 (with Acoustic Valve / CAW)
# ============================================================
class SpermacetiWaveguideV2:
    """1D digital waveguide with Huggenberger 2014 acoustic valve.

    The Connecting Acoustic Window (CAW) is a second exit point along the
    cavity where the right nasal passage runs between the spermaceti organ
    and the junk. When the nasal passage is air-filled (relaxed muscle),
    it acts as an impedance barrier and sound stays in the cavity. When
    the passage is collapsed (contracted muscle), sound can leak through
    into the junk.

    The frontal sac curvature is controlled by the same maxillonasalis
    muscle: contracted = parabolic (focused, high R), relaxed = flat
    (unfocused, lower R).

    Parameters
    ----------
    organ_length_m : float
        Length of the spermaceti organ cavity.
    spermaceti_c : float
        Speed of sound in spermaceti oil (typically 1370 m/s).
    sample_rate : int
        Audio sample rate.
    R_distal : float
        Reflection coefficient at the distal air sac (anterior end).
    R_frontal_base : float
        Base reflection coefficient at the frontal sac (before curvature).
    wall_loss : float
        Per-sample amplitude retention for viscous/thermal boundary loss.
    T_caw : float
        Transmission coefficient at the CAW tap point.
        0.0 = fully sealed (coda mode), 0.35 = open (echolocation mode).
    caw_position : float
        Normalized position of CAW along cavity (0=distal, 1=frontal).
        Anatomically ~0.5 (midway, where right nasal passage runs).
    frontal_sac_curvature : float
        0.0 = flat (relaxed), 1.0 = fully parabolic (contracted).
        Controls effective R_frontal: R_eff = R_base * (0.85 + 0.15 * curvature).
    """

    def __init__(self, organ_length_m, spermaceti_c, sample_rate=44100,
                 R_distal=0.85, R_frontal_base=0.90, wall_loss=0.9997,
                 T_caw=0.0, caw_position=0.5, frontal_sac_curvature=0.5):
        self.delay_samples = max(1, int(organ_length_m / spermaceti_c * sample_rate))
        self.R_d = R_distal
        self.R_f_base = R_frontal_base
        self.wall_loss = wall_loss
        self.sample_rate = sample_rate
        self.organ_length = organ_length_m
        self.spermaceti_c = spermaceti_c

        # Acoustic valve parameters
        self.T_caw = T_caw
        self.caw_idx = max(0, min(int(caw_position * self.delay_samples),
                                  self.delay_samples - 1))
        self.frontal_sac_curvature = frontal_sac_curvature

        # Effective frontal sac reflection: parabolic mirror focuses more
        # energy back into the cavity than a flat mirror
        # curvature=0 (flat): R_eff = R_base * 0.85
        # curvature=1 (parabolic): R_eff = R_base * 1.0
        self.R_f = R_frontal_base * (0.85 + 0.15 * frontal_sac_curvature)
        # Clamp to physical limits
        self.R_f = min(self.R_f, 0.995)

        # Delay lines
        self.forward = np.zeros(self.delay_samples)
        self.backward = np.zeros(self.delay_samples)

        # Frequency-dependent wall loss via one-pole lowpass
        # Higher frequencies lose more energy per transit (viscous boundary layer)
        # Cutoff is lower for relaxed muscle (coda) = more HF damping per pass
        # Cutoff is higher for tensed muscle (echo) = less HF damping
        # This is physically correct: muscle tension stiffens the cavity walls
        self._lp_cutoff = 8000.0 + 12000.0 * frontal_sac_curvature  # 8-20kHz range
        self._lp_alpha = 1.0 - np.exp(-2.0 * np.pi * self._lp_cutoff / sample_rate)
        self._lp_state_fwd = 0.0
        self._lp_state_bwd = 0.0

        # Muscle bands
        self.muscle_bands = []

        # Computed properties
        self.one_way_time = organ_length_m / spermaceti_c
        self.round_trip_time = 2 * self.one_way_time
        self.expected_ipi = self.round_trip_time

        # Per-round-trip amplitude retention (without CAW)
        rt_retention = (R_distal * self.R_f) * (wall_loss ** (2 * self.delay_samples))

        print(f"  WaveguideV2: L={organ_length_m:.2f}m, c={spermaceti_c}m/s, "
              f"delay={self.delay_samples}samp ({self.one_way_time*1000:.2f}ms), "
              f"R_d={R_distal:.2f}, R_f_eff={self.R_f:.3f} "
              f"(base={R_frontal_base:.2f}, curv={frontal_sac_curvature:.1f}), "
              f"T_caw={T_caw:.2f} @idx={self.caw_idx}, "
              f"rt_retention={rt_retention:.3f}")

    def add_muscle_band(self, position, freq, tension):
        band = MuscleBand(position, freq, tension, self.sample_rate)
        self.muscle_bands.append(band)

    def step(self, input_sample=0.0):
        """Advance one sample. Returns total output (distal + CAW leak)."""
        # Read from ends of delay lines
        forward_out = self.forward[-1]    # arrives at frontal sac
        backward_out = self.backward[-1]  # arrives at distal sac

        # === CAW TAP: extract energy from forward wave at CAW position ===
        # This happens BEFORE the wave reaches the frontal sac
        caw_leak = self.T_caw * self.forward[self.caw_idx]
        self.forward[self.caw_idx] *= (1.0 - self.T_caw)  # energy conservation

        # === Frontal sac reflection (posterior, skull-backed) ===
        # Sign flip for pressure inversion at hard boundary
        reflected_f = -self.R_f * forward_out

        # === Distal sac reflection (anterior) ===
        reflected_d = -self.R_d * backward_out
        # Energy exits through junk
        transmitted_d = (1.0 - abs(self.R_d)) * backward_out

        # Shift delay lines (wave propagation)
        self.forward[1:] = self.forward[:-1]
        self.backward[1:] = self.backward[:-1]

        # Inject at boundaries
        self.forward[0] = reflected_d + input_sample
        self.backward[0] = reflected_f

        # Apply wall loss (frequency-independent)
        self.forward *= self.wall_loss
        self.backward *= self.wall_loss

        # Apply frequency-dependent wall loss (one-pole lowpass on each end)
        # This models viscous/thermal boundary layer losses that scale with freq
        # Each pass through the cavity applies this filter, so after N round-trips,
        # HF energy is attenuated by lp_alpha^N -- naturally creating the coda/echo
        # spectral difference since coda clicks have more round-trips
        self._lp_state_fwd += self._lp_alpha * (self.forward[-1] - self._lp_state_fwd)
        self.forward[-1] = self._lp_state_fwd
        self._lp_state_bwd += self._lp_alpha * (self.backward[-1] - self._lp_state_bwd)
        self.backward[-1] = self._lp_state_bwd

        # Apply muscle band damping
        for band in self.muscle_bands:
            idx = int(band.position * (self.delay_samples - 1))
            idx = max(0, min(idx, self.delay_samples - 1))
            self.forward[idx] = band.filter_sample(self.forward[idx], 'forward')
            self.backward[idx] = band.filter_sample(self.backward[idx], 'backward')

        # Track both output paths separately for post-processing
        self._last_distal = transmitted_d
        self._last_caw = caw_leak

        # Total output = distal transmission + CAW leak
        total_output = transmitted_d + caw_leak

        return total_output

    def run(self, source_signal, duration_samples=None, record_internal=False):
        if duration_samples is None:
            duration_samples = len(source_signal) + self.delay_samples * 20

        output = np.zeros(duration_samples)
        self.distal_output = np.zeros(duration_samples)
        self.caw_output = np.zeros(duration_samples)
        self.forward_history = []
        self.backward_history = []
        record_interval = max(1, self.delay_samples // 4)

        for i in range(duration_samples):
            inp = source_signal[i] if i < len(source_signal) else 0.0
            output[i] = self.step(inp)
            self.distal_output[i] = self._last_distal
            self.caw_output[i] = self._last_caw

            if record_internal and (i % record_interval == 0):
                self.forward_history.append(self.forward.copy())
                self.backward_history.append(self.backward.copy())

        return output

    def reset(self):
        self.forward[:] = 0.0
        self.backward[:] = 0.0
        self.forward_history = []
        self.backward_history = []
        for band in self.muscle_bands:
            band.z_fwd[:] = 0.0
            band.z_bwd[:] = 0.0


# ============================================================
# LIP BUZZ SOURCE (parameterizable)
# ============================================================
def lip_buzz_source(lip_freq=1500, duration_us=600, sample_rate=44100,
                    amplitude=1.0, jitter=0.3, seed=42, broadband=True):
    """Buzzing phonic lips source - short impulsive burst."""
    np.random.seed(seed)
    n = max(1, int(duration_us * 1e-6 * sample_rate))
    t = np.arange(n) / sample_rate

    if broadband:
        noise = np.random.randn(n)
        env = np.exp(-0.5 * ((t - t[n//2]) / (duration_us * 0.3e-6))**2)
        phase_noise = jitter * np.cumsum(np.random.randn(n)) / sample_rate * lip_freq * 2 * np.pi
        tonal = np.sin(2 * np.pi * lip_freq * t + phase_noise)
        signal = 0.6 * noise * env + 0.4 * np.maximum(tonal, 0) * env
        signal *= np.hanning(n)
    else:
        phase = 2 * np.pi * lip_freq * t
        noise = jitter * np.cumsum(np.random.randn(n)) / sample_rate * lip_freq * 2 * np.pi
        raw = np.sin(phase + noise)
        signal = np.maximum(raw, 0)
        signal *= np.hanning(n)

    return amplitude * signal


# ============================================================
# EXIT PATH FILTER (6-layer tissue absorption)
# ============================================================
def apply_exit_path_filter(signal, dt, junk_length=2.0):
    """Frequency-domain absorption for each tissue layer on exit path (distal sac path)."""
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


def apply_caw_exit_filter(signal, dt):
    """Exit filter for CAW path - much shorter, less tissue.

    The CAW path goes through:
    - Thin case wall (~0.03m) where spermaceti organ and junk are adjacent
    - Fatty muscle of right nasal passage (~0.02m)
    - Directly into the junk at a mid-point (shorter remaining junk path ~0.5m)
    - Then through blubber + skin

    Much less tissue absorption than the full distal sac path, so high
    frequencies are preserved better. This is why echolocation clicks
    (which exit primarily through CAW) have higher centroids.
    """
    spectrum = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), dt)
    f_mhz = freqs / 1e6
    absorption_power = 1.5

    # CAW path layers - much thinner/shorter than full distal path
    caw_layers = [
        ('case_wall_thin', 0.03, 1.0, 1570),    # thin case wall at CAW
        ('nasal_passage',  0.02, 0.8, 1550),     # fatty muscle of nasal passage
        ('junk_lipid_short', 0.30, 0.5, 1400),   # only ~0.3m of junk (mid-point exit)
        ('blubber',        0.15, 0.5, 1420),
        ('skin',           0.015, 2.0, 1600),
    ]

    for name, thickness_m, alpha_db_cm_mhz, c_tissue in caw_layers:
        if thickness_m < 0.001:
            continue
        thickness_cm = thickness_m * 100.0
        atten_db = alpha_db_cm_mhz * np.power(np.maximum(f_mhz, 1e-10), absorption_power) * thickness_cm
        atten_linear = np.power(10.0, -atten_db / 20.0)
        spectrum *= atten_linear

    return np.fft.irfft(spectrum, n=len(signal))


# ============================================================
# OCEAN PROPAGATION (Francois-Garrison, depth-integrated)
# ============================================================
def apply_ocean_propagation(signal, dt, propagation=None):
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
# BODY RESONANCE
# ============================================================
def add_body_resonance(signal, config, sample_rate=44100):
    body_length = config.get('body_length_m', 12.0)
    scale = config.get('body_resonance_scale', 0.1)
    if scale <= 0:
        return signal

    f_body = 1570.0 / (2 * body_length)
    Q_body = 5
    t = np.arange(len(signal)) / sample_rate
    click_energy = np.sqrt(np.mean(signal**2))
    body_signal = (click_energy * scale *
                   np.sin(2 * np.pi * f_body * t) *
                   np.exp(-np.pi * f_body * t / Q_body))
    return signal + body_signal


# ============================================================
# LIP EXCITATION SPECTRUM SHAPING
# ============================================================
def apply_lip_excitation_spectrum(signal, config, sample_rate=44100):
    """Shape waveguide output spectrum to match lip excitation envelope.

    Uses a super-Gaussian (flatter top, steeper rolloff) for more
    aggressive spectral control, matching the physical reality that
    the phonic lips produce a specific frequency range.
    """
    lip_peak = config.get('lip_peak_hz', 5000)
    lip_bw = config.get('lip_bandwidth_octaves', 2.0)
    low_floor = config.get('low_freq_floor', 0.05)

    dt = 1.0 / sample_rate
    spectrum = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), dt)

    with np.errstate(divide='ignore', invalid='ignore'):
        log_ratio = np.log2(np.maximum(freqs, 1.0) / lip_peak)
    # Super-Gaussian (order 4) for steeper rolloff than standard Gaussian
    envelope = np.exp(-0.5 * (log_ratio / lip_bw)**4)
    envelope = np.maximum(envelope, low_floor)
    envelope[0] = low_floor

    spectrum *= envelope
    return np.fft.irfft(spectrum, n=len(signal))


# ============================================================
# FEATURE EXTRACTION
# ============================================================
def compute_band_energies(signal, dt):
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
    fft_mag = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal), dt)
    total = np.sum(fft_mag)
    if total < 1e-30:
        return 0.0
    return float(np.sum(freqs * fft_mag) / total)


# ============================================================
# PULSE DETECTION
# ============================================================
def detect_pulses(signal, sample_rate, expected_ipi_s, threshold_frac=0.05):
    analytic = np.abs(signal)
    win_samples = max(3, int(0.0003 * sample_rate))
    kernel = np.ones(win_samples) / win_samples
    envelope = np.convolve(analytic, kernel, mode='same')

    peak_amp = np.max(envelope)
    if peak_amp < 1e-20:
        return []

    threshold = peak_amp * threshold_frac
    min_separation = int(0.4 * expected_ipi_s * sample_rate)
    pulses = []
    i = 0
    while i < len(envelope):
        if envelope[i] > threshold:
            region_end = min(i + min_separation, len(envelope))
            peak_idx = i + np.argmax(envelope[i:region_end])
            pulses.append((peak_idx, float(envelope[peak_idx]), f'P{len(pulses)}'))
            i = peak_idx + min_separation
        else:
            i += 1
    return pulses


# ============================================================
# LOAD REAL WAV DATA
# ============================================================
def get_whale_wav_files():
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


def get_mean_real_click(whale_id, whale_files, n_clicks=10):
    files = whale_files.get(whale_id, [])
    if not files:
        return None, None

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

    if not clicks:
        return None, None
    best = max(clicks, key=lambda x: np.sqrt(np.mean(x[0]**2)))
    return best


# ============================================================
# LOAD V1 WAVEGUIDE AND MODE SYNTH V2 CLICKS FOR COMPARISON
# ============================================================
def load_v1_waveguide_click(whale_name):
    """Load waveguide v1 WAV for comparison."""
    wid = whale_name.split('_')[1]
    wav_path = OUTPUT_DIR / f"whale_{wid}_waveguide.wav"
    if not wav_path.exists():
        return None
    try:
        sr, data = wavfile.read(wav_path)
        if data.dtype == np.int16:
            data = data.astype(np.float64) / 32768.0
        elif data.dtype == np.float32:
            data = data.astype(np.float64)
        if data.ndim > 1:
            data = data[:, 0]
        # Extract first click
        threshold = np.max(np.abs(data)) * 0.05
        start = 0
        for i in range(len(data)):
            if abs(data[i]) > threshold:
                start = max(0, i - 100)
                break
        click_samples = int(0.025 * sr)
        end = min(start + click_samples, len(data))
        click = data[start:end]
        if sr != TARGET_SR:
            n_samp = int(len(click) * TARGET_SR / sr)
            click = resample(click, n_samp)
        return click
    except Exception as e:
        print(f"  Warning: could not load v1 waveguide for {whale_name}: {e}")
        return None


def load_mode_synth_v2_click(whale_name):
    """Load mode synth v2 WAV for comparison."""
    wid = whale_name.split('_')[1]
    wav_path = MODE_SYNTH_DIR / f"whale_{wid}_v2.wav"
    if not wav_path.exists():
        return None
    try:
        sr, data = wavfile.read(wav_path)
        if data.dtype == np.int16:
            data = data.astype(np.float64) / 32768.0
        elif data.dtype == np.float32:
            data = data.astype(np.float64)
        if data.ndim > 1:
            data = data[:, 0]
        threshold = np.max(np.abs(data)) * 0.05
        start = 0
        for i in range(len(data)):
            if abs(data[i]) > threshold:
                start = max(0, i - 100)
                break
        click_samples = int(0.025 * sr)
        end = min(start + click_samples, len(data))
        click = data[start:end]
        if sr != TARGET_SR:
            n_samp = int(len(click) * TARGET_SR / sr)
            click = resample(click, n_samp)
        return click
    except Exception as e:
        print(f"  Warning: could not load mode synth v2 for {whale_name}: {e}")
        return None


# ============================================================
# CODA GENERATION
# ============================================================
def generate_coda(click_signal, icis_ms, sample_rate=44100):
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

    click_start = int(pos / 1000 * sample_rate)
    click_end = min(click_start + len(click_signal), total_samples)
    n = click_end - click_start
    if n > 0:
        coda[click_start:click_end] += click_signal[:n]

    return coda


# ============================================================
# SAVE WAV UTILITIES
# ============================================================
def save_wav(signal, filename, sample_rate=44100, n_repeats=5, ici_s=0.3):
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
    print(f"  Saved: {filename} ({dur_s:.1f}s)")


def save_coda_wav(coda_signal, filename, sample_rate=44100):
    peak = np.max(np.abs(coda_signal))
    if peak > 0:
        coda_signal = coda_signal * 0.9 / peak
    output_16 = np.clip(coda_signal * 32767, -32768, 32767).astype(np.int16)
    wavfile.write(str(filename), sample_rate, output_16)
    dur_s = len(output_16) / sample_rate
    print(f"  Saved: {filename} ({dur_s:.2f}s)")


# ============================================================
# FULL CLICK GENERATION PIPELINE
# ============================================================
def generate_click_v2(whale_name, config, click_mode=None, sample_rate=44100,
                      record_internal=False):
    """Generate a single click using waveguide V2 with acoustic valve.

    Parameters
    ----------
    whale_name : str
        e.g. 'Whale_D'
    config : dict
        Whale anatomy configuration.
    click_mode : str or None
        One of CLICK_MODES keys, or None for default (uses whale config as-is).
    """
    t0 = time.time()
    np.random.seed(42)

    mode_label = click_mode if click_mode else 'default'
    print(f"\n{'='*60}")
    print(f"Generating {whale_name} click (waveguide V2, mode={mode_label})")
    print(f"{'='*60}")

    # Merge click mode presets over whale config
    cfg = dict(config)
    if click_mode and click_mode in CLICK_MODES:
        mode_params = CLICK_MODES[click_mode]
        # Mode overrides specific parameters
        cfg['R_distal'] = mode_params['R_distal']
        cfg['source_duration_us'] = mode_params['source_duration_us']
        cfg['lip_freq'] = mode_params['lip_freq']
        cfg['lip_peak_hz'] = mode_params['lip_peak_hz']
        cfg['lip_bandwidth_octaves'] = mode_params['lip_bandwidth_octaves']
        cfg['low_freq_floor'] = mode_params.get('low_freq_floor', cfg.get('low_freq_floor', 0.05))
        # Scale junk length for CAW path (shorter path = less tissue filtering)
        junk_scale = mode_params.get('junk_length_scale', 1.0)
        cfg['junk_length'] = cfg.get('junk_length', 2.0) * junk_scale

    # Get acoustic valve params from mode or defaults
    if click_mode and click_mode in CLICK_MODES:
        mode_p = CLICK_MODES[click_mode]
        T_caw = mode_p['T_caw']
        frontal_sac_curvature = mode_p['frontal_sac_curvature']
        R_frontal_base = mode_p['R_frontal']
    else:
        T_caw = 0.05  # default: mostly closed (coda-like)
        frontal_sac_curvature = 0.3
        R_frontal_base = cfg.get('R_frontal', 0.90)

    # 1. Create waveguide V2 with acoustic valve
    wg = SpermacetiWaveguideV2(
        organ_length_m=cfg['organ_length'],
        spermaceti_c=cfg['spermaceti_c'],
        sample_rate=sample_rate,
        R_distal=cfg.get('R_distal', 0.85),
        R_frontal_base=R_frontal_base,
        wall_loss=cfg.get('wall_loss', 0.9997),
        T_caw=T_caw,
        caw_position=0.5,  # anatomically: midway where nasal passage runs
        frontal_sac_curvature=frontal_sac_curvature,
    )

    # 2. Add muscle bands
    # Scale muscle band tensions with overall muscle_tension from mode
    if click_mode and click_mode in CLICK_MODES:
        tension_scale = CLICK_MODES[click_mode]['muscle_tension']
    else:
        tension_scale = 0.5  # moderate default
    for i in range(6):
        pos = (i + 1) / 7.0
        freq = cfg['muscle_band_freqs'][i]
        base_tension = cfg['muscle_tensions'][i]
        # Scale individual band tension by overall muscle state
        effective_tension = base_tension * (0.5 + tension_scale)
        wg.add_muscle_band(pos, freq, min(effective_tension, 1.0))
    print(f"  Added {len(wg.muscle_bands)} muscle bands (tension_scale={tension_scale:.1f})")

    # 3. Generate lip buzz source
    broadband = cfg.get('source_bandwidth', 'broadband') == 'broadband'
    source = lip_buzz_source(
        lip_freq=cfg.get('lip_freq', 1500),
        duration_us=cfg.get('source_duration_us', 600),
        sample_rate=sample_rate,
        jitter=cfg.get('lip_jitter', 0.3),
        broadband=broadband,
    )
    print(f"  Source: {len(source)} samples ({len(source)/sample_rate*1e6:.0f}us, "
          f"lip_freq={cfg.get('lip_freq', 1500)}Hz)")

    # 4. Run waveguide
    raw = wg.run(source, record_internal=record_internal)
    print(f"  Waveguide output: {len(raw)} samples ({len(raw)/sample_rate*1000:.1f}ms)")

    dt = 1.0 / sample_rate

    # 5. Process two exit paths separately with their own tissue filters
    # Distal sac path: full junk length tissue filter
    distal_shaped = apply_lip_excitation_spectrum(wg.distal_output, cfg, sample_rate)
    distal_filtered = apply_exit_path_filter(distal_shaped, dt,
                                              junk_length=cfg.get('junk_length', 2.0))

    # CAW path: shorter tissue path (preserves more high frequency)
    caw_shaped = apply_lip_excitation_spectrum(wg.caw_output, cfg, sample_rate)
    caw_filtered = apply_caw_exit_filter(caw_shaped, dt)

    # Combine the two exit paths
    combined = distal_filtered + caw_filtered

    # 6. Add body resonance
    filtered = add_body_resonance(combined, cfg, sample_rate)

    # 7. Apply ocean propagation
    final = apply_ocean_propagation(filtered, dt)

    # 9. Detect pulses
    pulses = detect_pulses(raw, sample_rate, wg.expected_ipi)
    print(f"  Detected {len(pulses)} pulses:")
    for idx, amp, label in pulses[:6]:
        t_ms = idx / sample_rate * 1000
        print(f"    {label}: t={t_ms:.2f}ms, amp={amp:.6f}")

    if len(pulses) >= 2:
        measured_ipi = (pulses[1][0] - pulses[0][0]) / sample_rate * 1000
        print(f"  Measured IPI (P0-P1): {measured_ipi:.2f}ms")

    centroid = compute_spectral_centroid(final, dt)
    bands = compute_band_energies(final, dt)
    print(f"  Spectral centroid: {centroid:.0f}Hz")

    elapsed = time.time() - t0
    print(f"  Synthesis time: {elapsed*1000:.1f}ms")

    return {
        'raw': raw,
        'shaped': combined,
        'filtered': filtered,
        'final': final,
        'waveguide': wg,
        'source': source,
        'pulses': pulses,
        'elapsed_ms': elapsed * 1000,
        'centroid_hz': centroid,
        'bands': bands,
        'click_mode': mode_label,
        'T_caw': T_caw,
        'frontal_sac_curvature': frontal_sac_curvature,
        'R_f_effective': wg.R_f,
    }


# ============================================================
# FIGURE 1: CLICK TYPES COMPARISON (20x16)
# ============================================================
def create_click_types_figure(mode_results, whale_name, output_dir):
    """Create click_types_comparison.png (20x16).

    4 click types x 3 rows: time domain, spectrum, mode parameters chart.
    """
    print(f"\n[Figure] Creating click_types_comparison.png (20x16)...")

    modes = ['echolocation', 'coda', 'creak', 'slow_click']
    mode_colors = {
        'echolocation': '#e74c3c',
        'coda': '#3498db',
        'creak': '#f39c12',
        'slow_click': '#27ae60',
    }

    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35,
                  height_ratios=[1, 1, 0.6])

    dt = 1.0 / TARGET_SR

    for col, mode_name in enumerate(modes):
        result = mode_results[mode_name]
        final = result['final']
        raw = result['raw']
        pulses = result['pulses']
        color = mode_colors[mode_name]

        def norm(s):
            mx = np.max(np.abs(s))
            return s / mx if mx > 1e-20 else s

        # === Row 1: Time domain ===
        ax = fig.add_subplot(gs[0, col])
        t_ms = np.arange(len(final)) / TARGET_SR * 1000
        ax.plot(t_ms, norm(final), color=color, linewidth=0.8)

        for idx, amp, label in pulses[:6]:
            t_pulse = idx / TARGET_SR * 1000
            if t_pulse < t_ms[-1]:
                ax.axvline(t_pulse, color='red', linestyle=':', alpha=0.4, linewidth=0.7)
                ax.text(t_pulse, 1.08, label, fontsize=7, ha='center',
                        color='red', fontweight='bold',
                        transform=ax.get_xaxis_transform())

        mode_p = CLICK_MODES[mode_name]
        ax.set_title(f'{mode_name.replace("_", " ").title()}\n'
                     f'T_caw={mode_p["T_caw"]:.2f}, '
                     f'curv={mode_p["frontal_sac_curvature"]:.1f}, '
                     f'R_f_eff={result["R_f_effective"]:.3f}',
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Normalized Amp')
        ax.set_xlim(0, min(30, t_ms[-1]))

        n_pulses = len(pulses)
        centroid = result['centroid_hz']
        ax.text(0.02, 0.95, f'{n_pulses} pulses\n{centroid:.0f}Hz centroid',
                transform=ax.transAxes, fontsize=8, va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        # === Row 2: Spectrum ===
        ax = fig.add_subplot(gs[1, col])
        fft_mag = np.abs(np.fft.rfft(final))
        freqs = np.fft.rfftfreq(len(final), dt) / 1000
        fft_db = 20 * np.log10(fft_mag / max(np.max(fft_mag), 1e-30) + 1e-30)
        ax.plot(freqs, fft_db, color=color, linewidth=1.0)

        ax.axvline(centroid / 1000, color='green', linestyle='-', alpha=0.6,
                   linewidth=1.5, label=f'Centroid: {centroid:.0f}Hz')

        ax.set_title(f'Spectrum - {mode_name.replace("_", " ").title()}',
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('Frequency (kHz)')
        ax.set_ylabel('Magnitude (dB)')
        ax.set_xlim(0, 22)
        ax.set_ylim(-60, 5)
        ax.legend(fontsize=7)

    # === Row 3: Mode parameter comparison bar chart ===
    # Left half: parameter bars, Right half: centroid/pulse comparison
    ax_params = fig.add_subplot(gs[2, 0:2])
    ax_metrics = fig.add_subplot(gs[2, 2:4])

    # Parameter bars
    param_names = ['T_caw', 'frontal_sac_curvature', 'R_frontal', 'muscle_tension']
    param_labels = ['T_caw\n(nasal passage)', 'Frontal Sac\nCurvature',
                    'R_frontal', 'Muscle\nTension']
    x = np.arange(len(param_names))
    width = 0.18

    for i, mode_name in enumerate(modes):
        mp = CLICK_MODES[mode_name]
        vals = [mp['T_caw'], mp['frontal_sac_curvature'],
                mp['R_frontal'], mp['muscle_tension']]
        ax_params.bar(x + i * width - 1.5 * width, vals, width,
                      label=mode_name.replace('_', ' ').title(),
                      color=mode_colors[mode_name], alpha=0.8)

    ax_params.set_title('Acoustic Valve Parameters (Huggenberger 2014)',
                        fontsize=11, fontweight='bold')
    ax_params.set_xticks(x)
    ax_params.set_xticklabels(param_labels, fontsize=8)
    ax_params.set_ylabel('Value')
    ax_params.legend(fontsize=7, ncol=2)
    ax_params.set_ylim(0, 1.1)

    # Centroid and pulse count comparison
    centroids = [mode_results[m]['centroid_hz'] / 1000 for m in modes]
    pulse_counts = [len(mode_results[m]['pulses']) for m in modes]

    x2 = np.arange(len(modes))
    ax_metrics.bar(x2 - 0.2, centroids, 0.35,
                   color=[mode_colors[m] for m in modes], alpha=0.8,
                   label='Centroid (kHz)')
    ax_twin = ax_metrics.twinx()
    ax_twin.bar(x2 + 0.2, pulse_counts, 0.35,
                color=[mode_colors[m] for m in modes], alpha=0.4,
                edgecolor='black', linewidth=1, label='Pulse count')

    ax_metrics.set_title('Synthesized Click Characteristics',
                         fontsize=11, fontweight='bold')
    ax_metrics.set_xticks(x2)
    ax_metrics.set_xticklabels([m.replace('_', ' ').title() for m in modes],
                                fontsize=8)
    ax_metrics.set_ylabel('Spectral Centroid (kHz)')
    ax_twin.set_ylabel('Pulse Count')

    # Add literature targets as horizontal lines
    ax_metrics.axhline(15, color='red', linestyle='--', alpha=0.4, linewidth=0.8)
    ax_metrics.text(3.5, 15.3, 'Echo target 15kHz', fontsize=7, color='red', alpha=0.6)
    ax_metrics.axhline(5, color='blue', linestyle='--', alpha=0.4, linewidth=0.8)
    ax_metrics.text(3.5, 5.3, 'Coda target 5kHz', fontsize=7, color='blue', alpha=0.6)
    ax_metrics.axhline(3, color='green', linestyle='--', alpha=0.4, linewidth=0.8)
    ax_metrics.text(3.5, 3.3, 'Slow target 3kHz', fontsize=7, color='green', alpha=0.6)

    fig.suptitle(f'{whale_name} - Click Type Comparison (Waveguide V2 + Acoustic Valve)\n'
                 f'Huggenberger 2014: Connecting Acoustic Window + Frontal Sac Curvature',
                 fontsize=13, fontweight='bold', y=0.99)
    plt.savefig(output_dir / 'click_types_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir / 'click_types_comparison.png'}")


# ============================================================
# FIGURE 2: V1 vs V2 vs MODE SYNTH vs REAL COMPARISON (16x12)
# ============================================================
def create_v1_v2_comparison_figure(v2_coda_result, whale_name, whale_files, output_dir):
    """Create v1_v2_comparison.png (16x12) showing Whale D coda from
    waveguide v1, waveguide v2, mode_synth_v2, and real recording."""
    print(f"\n[Figure] Creating v1_v2_comparison.png (16x12)...")

    wid = whale_name.split('_')[1]
    dt = 1.0 / TARGET_SR

    # Load all versions
    v1_click = load_v1_waveguide_click(whale_name)
    v2_click = v2_coda_result['final']
    mode_click = load_mode_synth_v2_click(whale_name)
    real_click, _ = get_mean_real_click(wid, whale_files)

    sources = []
    if v1_click is not None:
        sources.append(('Waveguide V1', v1_click, '#e74c3c'))
    sources.append(('Waveguide V2 (acoustic valve)', v2_click, '#3498db'))
    if mode_click is not None:
        sources.append(('Mode Synth V2', mode_click, '#f39c12'))
    if real_click is not None:
        sources.append(('Real Recording (DSWP)', real_click, '#2ecc71'))

    n_sources = len(sources)
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, n_sources, figure=fig, hspace=0.45, wspace=0.35,
                  height_ratios=[1, 1, 0.8])

    def norm(s):
        mx = np.max(np.abs(s))
        return s / mx if mx > 1e-20 else s

    centroids = []
    band_data = []

    for col, (label, click, color) in enumerate(sources):
        # Row 1: Time domain
        ax = fig.add_subplot(gs[0, col])
        t_ms = np.arange(len(click)) / TARGET_SR * 1000
        ax.plot(t_ms, norm(click), color=color, linewidth=0.8)
        ax.set_title(label, fontsize=10, fontweight='bold')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Normalized Amp')
        ax.set_xlim(0, min(25, t_ms[-1]))

        # Row 2: Spectrum
        ax = fig.add_subplot(gs[1, col])
        fft_mag = np.abs(np.fft.rfft(click))
        freqs = np.fft.rfftfreq(len(click), dt) / 1000
        fft_db = 20 * np.log10(fft_mag / max(np.max(fft_mag), 1e-30) + 1e-30)
        ax.plot(freqs, fft_db, color=color, linewidth=1.0)

        centroid = compute_spectral_centroid(click, dt)
        centroids.append((label, centroid))
        bands = compute_band_energies(click, dt)
        band_data.append((label, bands, color))

        ax.axvline(centroid / 1000, color='green', linestyle='-', alpha=0.6,
                   linewidth=1.5, label=f'Centroid: {centroid:.0f}Hz')
        ax.set_xlabel('Frequency (kHz)')
        ax.set_ylabel('dB')
        ax.set_xlim(0, 22)
        ax.set_ylim(-60, 5)
        ax.legend(fontsize=7)

    # Row 3: Band energy comparison (single chart spanning all columns)
    ax = fig.add_subplot(gs[2, :])
    band_names = ['sub_100hz', '100_500hz', '500_2khz', '2_5khz',
                  '5_10khz', '10_20khz', 'above_20khz']
    band_labels = ['<100', '100-500', '0.5-2k', '2-5k', '5-10k', '10-20k', '>20k']
    x = np.arange(len(band_names))
    width = 0.8 / max(len(band_data) + 1, 1)

    for i, (label, bands, color) in enumerate(band_data):
        vals = [bands.get(f'band_{bn}_pct', 0) for bn in band_names]
        offset = (i - len(band_data) / 2) * width
        ax.bar(x + offset, vals, width, label=label, color=color, alpha=0.7)

    # Add real targets
    real_target = REAL_TARGETS.get(whale_name, {})
    if real_target:
        target_vals = [real_target.get(f'band_{bn}_pct', 0) for bn in band_names]
        offset = (len(band_data) - len(band_data) / 2) * width
        ax.bar(x + offset, target_vals, width, label='Real Target (voiceprint)',
               color='gray', alpha=0.5, edgecolor='black', linewidth=0.5)

    ax.set_title(f'{whale_name} Coda Click - Band Energy Comparison', fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(band_labels, fontsize=9)
    ax.set_ylabel('Energy (%)')
    ax.legend(fontsize=7, ncol=3)

    # Compute distance from real for each source
    if real_click is not None:
        real_centroid = compute_spectral_centroid(real_click, dt)
        real_bands_dict = compute_band_energies(real_click, dt)
        print(f"\n  Distance from real recording ({whale_name}):")
        for label, centroid in centroids:
            if label == 'Real Recording (DSWP)':
                continue
            c_err = abs(centroid - real_centroid) / max(real_centroid, 1) * 100
            print(f"    {label}: centroid error = {c_err:.1f}%")

    fig.suptitle(f'{whale_name} Coda Click - Synthesizer Comparison\n'
                 f'Waveguide V1 vs V2 (acoustic valve) vs Mode Synth V2 vs Real',
                 fontsize=13, fontweight='bold', y=0.99)
    plt.savefig(output_dir / 'v1_v2_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir / 'v1_v2_comparison.png'}")


# ============================================================
# JSON SERIALIZATION
# ============================================================
def make_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='Digital Waveguide V2 - Sperm Whale with Acoustic Valve (Huggenberger 2014)')
    parser.add_argument('--whale', type=str, default=None,
                        help='Single whale (A, D, or F). Default: D for click types.')
    parser.add_argument('--all-codas', action='store_true',
                        help='Generate coda WAVs for all 3 whales')
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("DIGITAL WAVEGUIDE V2 - ACOUSTIC VALVE MODEL (Huggenberger 2014)")
    print("=" * 70)
    print("\nKey physics:")
    print("  - Connecting Acoustic Window (CAW): 2nd exit point on cavity")
    print("  - Frontal sac curvature: parabolic (echo) vs flat (coda)")
    print("  - Maxillonasalis couples CAW state + sac shape")

    # Load real data
    print("\n[Loading] Real click data from DSWP...")
    try:
        whale_files = get_whale_wav_files()
        print(f"  Found WAV files for {len(whale_files)} whales")
    except Exception as e:
        print(f"  Warning: could not load real data: {e}")
        whale_files = {}

    # ========================================
    # PART 1: Generate all 4 click types for Whale D
    # ========================================
    target_whale = f'Whale_{args.whale.upper()}' if args.whale else 'Whale_D'
    config = WHALE_CONFIGS[target_whale]
    wid = target_whale.split('_')[1]

    print(f"\n{'#'*70}")
    print(f"# PART 1: Click Type Comparison for {target_whale}")
    print(f"{'#'*70}")

    mode_results = {}
    for mode_name in ['echolocation', 'coda', 'creak', 'slow_click']:
        result = generate_click_v2(target_whale, config, click_mode=mode_name)
        mode_results[mode_name] = result

        # Save WAV
        wav_name = f'whale_{wid}_{mode_name}.wav'
        save_wav(result['final'], OUTPUT_DIR / wav_name, sample_rate=TARGET_SR)

    # Create click types comparison figure
    create_click_types_figure(mode_results, target_whale, OUTPUT_DIR)

    # ========================================
    # PART 2: Coda WAVs for all 3 whales
    # ========================================
    print(f"\n{'#'*70}")
    print(f"# PART 2: Coda WAVs (V3 - with acoustic valve)")
    print(f"{'#'*70}")

    all_coda_results = {}
    for wname in ['Whale_A', 'Whale_D', 'Whale_F']:
        wid_c = wname.split('_')[1]
        cfg = WHALE_CONFIGS[wname]
        result = generate_click_v2(wname, cfg, click_mode='coda')
        all_coda_results[wname] = result

        # Generate coda from click template
        icis = cfg.get('coda_icis_ms', [400, 400, 400, 1200])
        coda = generate_coda(result['final'], icis, sample_rate=TARGET_SR)
        save_coda_wav(coda, OUTPUT_DIR / f'whale_{wid_c}_coda_v3.wav',
                      sample_rate=TARGET_SR)

    # ========================================
    # PART 3: V1 vs V2 comparison figure
    # ========================================
    print(f"\n{'#'*70}")
    print(f"# PART 3: V1 vs V2 Comparison ({target_whale})")
    print(f"{'#'*70}")

    create_v1_v2_comparison_figure(
        all_coda_results.get(target_whale, mode_results['coda']),
        target_whale, whale_files, OUTPUT_DIR)

    # ========================================
    # PART 4: Results JSON
    # ========================================
    results_json = {
        'model': 'waveguide_v2',
        'physics': 'Huggenberger 2014 acoustic valve + digital waveguide',
        'components': [
            '1D digital waveguide (forward/backward delay lines)',
            'Connecting Acoustic Window (CAW) - 2nd exit tap',
            'Frontal sac curvature (parabolic vs flat mirror)',
            'Maxillonasalis muscle coupling (CAW + sac shape)',
            'Muscle band notch filters (6 bands)',
            'Parameterizable lip excitation',
            'Body resonance',
            '6-layer tissue exit filter',
            'Francois-Garrison ocean propagation',
        ],
        'click_modes': {},
        'whale_codas': {},
    }

    dt = 1.0 / TARGET_SR

    for mode_name, result in mode_results.items():
        mode_p = CLICK_MODES[mode_name]
        results_json['click_modes'][mode_name] = {
            'whale': target_whale,
            'T_caw': mode_p['T_caw'],
            'frontal_sac_curvature': mode_p['frontal_sac_curvature'],
            'R_distal': mode_p['R_distal'],
            'R_frontal_base': mode_p['R_frontal'],
            'R_frontal_effective': result['R_f_effective'],
            'muscle_tension': mode_p['muscle_tension'],
            'source_duration_us': mode_p['source_duration_us'],
            'lip_freq': mode_p['lip_freq'],
            'spectral_centroid_hz': round(result['centroid_hz'], 1),
            'n_pulses': len(result['pulses']),
            'pulses': [
                {'label': p[2], 'time_ms': round(p[0] / TARGET_SR * 1000, 3),
                 'amplitude': round(p[1], 6)}
                for p in result['pulses'][:8]
            ],
            'band_energies': {k: round(v, 2) for k, v in result['bands'].items()},
            'synthesis_time_ms': round(result['elapsed_ms'], 1),
        }

    for wname, result in all_coda_results.items():
        results_json['whale_codas'][wname] = {
            'click_mode': 'coda',
            'spectral_centroid_hz': round(result['centroid_hz'], 1),
            'n_pulses': len(result['pulses']),
            'band_energies': {k: round(v, 2) for k, v in result['bands'].items()},
            'real_targets': REAL_TARGETS.get(wname, {}),
        }

    results_json = make_serializable(results_json)
    json_path = OUTPUT_DIR / 'waveguide_v2_results.json'
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"\n  Saved: {json_path}")

    # ========================================
    # PRINT TABLE 1 COMPARISON
    # ========================================
    print("\n" + "=" * 80)
    print("CLICK TYPE SYNTHESIS vs LITERATURE (Table 1, Huggenberger 2014)")
    print("=" * 80)
    print(f"{'Type':<16} {'Literature':^30} {'Synthesized':^30}")
    print(f"{'':16} {'Centroid':>10} {'Pulses':>8} {'Dir':>6}    {'Centroid':>10} {'Pulses':>8}")
    print("-" * 80)

    lit_data = {
        'echolocation': ('~15kHz', '1-2', 'High'),
        'coda':         ('~5kHz',  '3-5', 'Low'),
        'creak':        ('~15kHz', 'many', 'High'),
        'slow_click':   ('~3kHz',  'faint', 'Low'),
    }

    for mode_name in ['echolocation', 'coda', 'creak', 'slow_click']:
        lit = lit_data[mode_name]
        result = mode_results[mode_name]
        synth_centroid = f"{result['centroid_hz']/1000:.1f}kHz"
        synth_pulses = str(len(result['pulses']))
        print(f"{mode_name:<16} {lit[0]:>10} {lit[1]:>8} {lit[2]:>6}    "
              f"{synth_centroid:>10} {synth_pulses:>8}")

    print("-" * 80)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n  Target whale: {target_whale}")
    print(f"  Organ: L={config['organ_length']:.2f}m, c={config['spermaceti_c']}m/s")
    print(f"  IPI (2L/c): {2 * config['organ_length'] / config['spermaceti_c'] * 1000:.2f}ms")
    print(f"\n  Key acoustic valve effects:")
    echo_c = mode_results['echolocation']['centroid_hz']
    coda_c = mode_results['coda']['centroid_hz']
    echo_p = len(mode_results['echolocation']['pulses'])
    coda_p = len(mode_results['coda']['pulses'])
    print(f"    Echo vs Coda centroid: {echo_c:.0f}Hz vs {coda_c:.0f}Hz "
          f"(ratio: {echo_c/max(coda_c,1):.1f}x)")
    print(f"    Echo vs Coda pulses: {echo_p} vs {coda_p}")
    print(f"    CAW transmission: echo T={CLICK_MODES['echolocation']['T_caw']:.2f} "
          f"vs coda T={CLICK_MODES['coda']['T_caw']:.2f}")
    print(f"    Frontal sac R_eff: echo {mode_results['echolocation']['R_f_effective']:.3f} "
          f"vs coda {mode_results['coda']['R_f_effective']:.3f}")

    print(f"\n  Output files:")
    print(f"    waveguide_v2.py (this script)")
    print(f"    click_types_comparison.png (20x16)")
    print(f"    v1_v2_comparison.png (16x12)")
    for mode_name in ['echolocation', 'coda', 'creak', 'slow_click']:
        print(f"    whale_{target_whale.split('_')[1]}_{mode_name}.wav")
    for wid_c in ['A', 'D', 'F']:
        print(f"    whale_{wid_c}_coda_v3.wav")
    print(f"    waveguide_v2_results.json")

    print(f"\n  All output saved to: {OUTPUT_DIR}")
    print("Done.")


if __name__ == '__main__':
    main()

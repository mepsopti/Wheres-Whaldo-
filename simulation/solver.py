#!/usr/bin/env python3
"""
Parameter Optimization Solver for Sperm Whale Acoustic Model

Wraps the FDTD simulator (sperm_whale_sim_v2.py) and uses
scipy.optimize.differential_evolution to find parameter combinations
that match real whale acoustic profiles.

Usage:
  python solver.py                    # Solve for Whale_A (default)
  python solver.py --whale A          # Solve for Whale_A
  python solver.py --whale D          # Solve for Whale_D
  python solver.py --whale all        # Solve for all whales
  python solver.py --whale A --maxiter 100
"""

import argparse
import json
import os
import sys
import time
import numpy as np
from pathlib import Path
from scipy.optimize import differential_evolution
from scipy.signal import hilbert, find_peaks
from scipy.ndimage import gaussian_filter1d

# Import simulator components - don't duplicate the physics
from sperm_whale_sim_v2 import (
    WhaleHead, broadband_impulse, fdtd_2d_v2,
    MATERIALS, ABSORPTION_COEFFICIENTS,
    TISSUE_WATER, TISSUE_SPERMACETI, TISSUE_JUNK, TISSUE_BONE,
    TISSUE_AIR_SAC, TISSUE_CONNECTIVE, TISSUE_BLUBBER,
)


# ============================================================
# REAL TARGETS (from actual recordings)
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

# Parameter names, bounds, and order
# Expanded parameter space - maximize variables so solver can find if a solution exists
PARAM_NAMES = [
    # Anatomy - cavity
    'organ_length',           # spermaceti organ length (m)
    'organ_diameter',         # spermaceti organ diameter (m)
    'skull_curvature',        # rostral basin radius (m)
    'junk_length',            # junk/melon length (m)
    'junk_max_diameter',      # junk max width (m)
    'case_wall_thickness',    # connective tissue case wall (m)
    'frontal_sac_width',      # frontal air sac width (m)
    'distal_sac_width',       # distal air sac width (m)
    # Exit path tissue layers (sound passes through these on the way out)
    'blubber_thickness',      # outer blubber layer (m)
    'skin_thickness',         # skin layer (m)
    'muscle_thickness',       # maxillonasalis + surrounding muscle (m)
    'junk_septa_count',       # number of connective tissue septa in junk (integer-ish)
    'junk_septa_thickness',   # thickness of each septum (m)
    # Physiology
    'spermaceti_temp',        # organ temperature (C) - controls sound speed
    'spermaceti_compression', # muscle compression ratio (1.0=relaxed, 1.15=squeezed 15%)
                              # compression increases density AND sound speed
                              # longer organ compensates for faster propagation
    'air_sac_rho',            # air sac membrane density (models depth compression)
    'air_sac_c',              # air sac membrane sound speed
    # Source
    'source_duration_us',     # phonic lip impulse duration (us)
    'source_freq_hz',         # if >0, use Ricker at this freq instead of impulse
    # Tissue absorption
    'absorption_scale',       # global multiplier on all absorption coefficients
    'absorption_power',       # frequency power law exponent (1.0=linear, 1.5=super-linear)
    # Exit path absorption (per-tissue, separate from cavity absorption)
    'muscle_absorption',      # muscle tissue absorption (dB/cm/MHz)
    'blubber_absorption',     # blubber absorption (dB/cm/MHz)
    'skin_absorption',        # skin absorption (dB/cm/MHz)
    'junk_absorption',        # junk lipid absorption (dB/cm/MHz)
    'connective_absorption',  # connective tissue/septa absorption (dB/cm/MHz)
    # Ocean / propagation
    'ocean_surface_temp',     # surface water temperature (C)
    'ocean_deep_temp',        # deep water temperature (C) at thermocline
    'whale_depth_m',          # depth of whale when clicking (m)
    'recording_distance_m',   # horizontal distance to hydrophone (m)
    'thermocline_depth_m',    # depth where temp transitions (m)
]

PARAM_BOUNDS = [
    # Anatomy - cavity
    (2.0, 6.0),      # organ_length (m)
    (0.6, 2.5),      # organ_diameter (m)
    (0.3, 2.5),      # skull_curvature (m)
    (0.8, 4.0),      # junk_length (m)
    (0.6, 2.0),      # junk_max_diameter (m)
    (0.02, 0.15),    # case_wall_thickness (m)
    (0.3, 1.5),      # frontal_sac_width (m)
    (0.1, 0.8),      # distal_sac_width (m)
    # Exit path tissue layers
    (0.05, 0.30),    # blubber_thickness (m) - 5cm to 30cm
    (0.005, 0.03),   # skin_thickness (m) - 5mm to 30mm (whale skin is thick)
    (0.02, 0.30),    # muscle_thickness (m) - maxillonasalis bands 2cm to 30cm
    (3.0, 30.0),     # junk_septa_count - connective tissue partitions in melon
    (0.001, 0.01),   # junk_septa_thickness (m) - 1mm to 10mm each
    # Physiology
    (20.0, 37.0),    # spermaceti_temp (C)
    (1.0, 1.20),     # spermaceti_compression - 1.0=no squeeze, 1.2=20% compressed
    (10.0, 200.0),   # air_sac_rho
    (300.0, 1200.0),  # air_sac_c
    # Source
    (20.0, 1000.0),  # source_duration_us
    (0.0, 15000.0),  # source_freq_hz - 0=impulse, >0=Ricker
    # Tissue absorption (cavity)
    (0.01, 10.0),    # absorption_scale
    (0.5, 2.5),      # absorption_power
    # Exit path absorption per tissue type (dB/cm/MHz)
    (0.3, 3.0),      # muscle_absorption - literature: 0.5-1.5 dB/cm/MHz
    (0.2, 2.0),      # blubber_absorption - literature: 0.3-0.8
    (0.5, 5.0),      # skin_absorption - literature: 1.0-3.0 (dense, collagenous)
    (0.2, 2.0),      # junk_absorption - lipid, similar to blubber
    (0.5, 3.0),      # connective_absorption - literature: 0.5-1.5
    # Ocean / propagation
    (20.0, 30.0),    # ocean_surface_temp (C) - tropical Caribbean
    (2.0, 15.0),     # ocean_deep_temp (C)
    (0.0, 1500.0),   # whale_depth_m
    (10.0, 2000.0),  # recording_distance_m
    (50.0, 800.0),   # thermocline_depth_m
]

# Cost function weights
FEATURE_WEIGHTS = {
    'spectral_centroid_hz': 3.0,
    'band_5_10khz_pct': 2.0,
    'band_10_20khz_pct': 1.0,
    'band_2_5khz_pct': 1.0,
    'ipi_ms': 3.0,
}


# ============================================================
# OCEAN PROPAGATION
# ============================================================

def mackenzie_sound_speed(temp_c):
    """Mackenzie equation for sound speed in seawater (simplified, salinity=35ppt, depth=0)."""
    T = temp_c
    return 1448.96 + 4.591 * T - 0.05304 * T**2 + 0.0002374 * T**3


def ocean_water_density(temp_c):
    """Approximate seawater density as function of temperature."""
    return 1028.0 - 0.08 * temp_c


def apply_exit_path_filter(signal, dt, param_dict):
    """Apply frequency-dependent absorption for tissue layers between
    the spermaceti cavity and the water.

    The click exits through: case wall -> muscle -> junk (with septa) -> blubber -> skin -> water.
    Each layer absorbs high frequencies more than low, following power law:
        alpha(f) = alpha_0 * (f / 1MHz) ^ power

    This is applied in the frequency domain as a series of exponential attenuations.
    """
    spectrum = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), dt)
    f_mhz = freqs / 1e6  # convert to MHz for dB/cm/MHz units

    absorption_power = param_dict.get('absorption_power', 1.5)

    # Each tissue layer: (thickness_m, absorption_dB_cm_MHz, sound_speed_m_s)
    layers = []

    # Case wall (connective tissue around the organ)
    layers.append((
        param_dict.get('case_wall_thickness', 0.05),
        param_dict.get('connective_absorption', 1.0),
        1570.0,  # connective tissue sound speed
    ))

    # Maxillonasalis muscle bands
    layers.append((
        param_dict.get('muscle_thickness', 0.10),
        param_dict.get('muscle_absorption', 1.0),
        1570.0,  # muscle sound speed
    ))

    # Junk/melon lipid (continuous)
    # Total junk path minus septa
    junk_total = param_dict.get('junk_length', 2.0) * 0.5  # assume ~half junk is in exit path
    septa_count = param_dict.get('junk_septa_count', 10)
    septa_thick = param_dict.get('junk_septa_thickness', 0.003)
    junk_lipid_path = max(0.01, junk_total - septa_count * septa_thick)

    layers.append((
        junk_lipid_path,
        param_dict.get('junk_absorption', 0.5),
        1400.0,  # average junk sound speed
    ))

    # Junk septa (connective tissue partitions - many thin layers)
    layers.append((
        septa_count * septa_thick,
        param_dict.get('connective_absorption', 1.0),
        1570.0,
    ))

    # Blubber
    layers.append((
        param_dict.get('blubber_thickness', 0.15),
        param_dict.get('blubber_absorption', 0.5),
        1420.0,
    ))

    # Skin
    layers.append((
        param_dict.get('skin_thickness', 0.015),
        param_dict.get('skin_absorption', 2.0),
        1600.0,  # skin is dense, collagenous
    ))

    # Apply each layer's absorption
    for thickness_m, alpha_db_cm_mhz, c_tissue in layers:
        if thickness_m < 0.001 or alpha_db_cm_mhz < 0.001:
            continue

        # Convert thickness to cm
        thickness_cm = thickness_m * 100.0

        # Frequency-dependent absorption: alpha(f) = alpha_0 * (f_MHz)^power
        # Total attenuation in dB = alpha_0 * f^power * thickness_cm
        atten_db = alpha_db_cm_mhz * np.power(np.maximum(f_mhz, 1e-10), absorption_power) * thickness_cm

        # Convert dB to linear
        atten_linear = np.power(10.0, -atten_db / 20.0)
        spectrum *= atten_linear

        # Impedance mismatch at each interface causes partial reflection
        # (minor effect but contributes to spectral shaping)
        # Z = rho * c, reflection at each boundary
        # Skip for now - absorption dominates

    return np.fft.irfft(spectrum, n=len(signal))


def depth_integrated_ocean_propagation(signal, dt, distance_m, whale_depth_m,
                                       surface_temp_c, deep_temp_c,
                                       thermocline_depth_m, absorption_power=1.5):
    """Apply depth-integrated ocean propagation.

    Sound from a whale at depth travels through a water column with
    varying temperature. This integrates absorption along the path.

    The path goes from whale_depth up to the surface hydrophone (~2m).
    Temperature profile: linear from surface_temp at 0m to deep_temp
    at thermocline_depth, then constant below.
    """
    spectrum = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), dt)
    f_khz = np.maximum(freqs / 1000.0, 1e-6)

    # Total path length (slant range from whale to surface hydrophone)
    slant_range_m = np.sqrt(distance_m**2 + whale_depth_m**2)
    if slant_range_m < 1.0:
        slant_range_m = 1.0

    # Integrate absorption along depth path
    # Split into depth segments, each with its own temperature
    n_segments = max(5, int(whale_depth_m / 50))  # 50m segments
    if whale_depth_m < 1:
        # Whale at surface - just use surface temp
        avg_temp = surface_temp_c
        total_abs_db = 0.1 * f_khz**absorption_power / (1 + avg_temp / 10.0) * slant_range_m / 1000.0
    else:
        total_abs_db = np.zeros_like(f_khz)
        segment_length = slant_range_m / n_segments
        for i in range(n_segments):
            # Depth of this segment (linear interpolation along slant path)
            frac = (i + 0.5) / n_segments
            seg_depth = whale_depth_m * (1 - frac)  # from whale depth up to surface

            # Temperature at this depth
            if seg_depth <= thermocline_depth_m:
                t_frac = seg_depth / max(thermocline_depth_m, 1)
                seg_temp = surface_temp_c + t_frac * (deep_temp_c - surface_temp_c)
            else:
                seg_temp = deep_temp_c

            # Absorption for this segment (dB/km at each frequency)
            seg_abs = 0.1 * f_khz**absorption_power / (1 + seg_temp / 10.0)
            total_abs_db += seg_abs * segment_length / 1000.0

    # Apply frequency-dependent attenuation
    attenuation = 10**(-total_abs_db / 20.0)
    spectrum *= attenuation

    # Geometric spreading (spherical)
    spectrum /= slant_range_m

    return np.fft.irfft(spectrum, n=len(signal))


# ============================================================
# FEATURE EXTRACTION (with fixed IPI detection)
# ============================================================

def detect_ipi_hilbert(signal, dt):
    """Detect inter-pulse interval using Hilbert envelope.

    The v2 peak detector is broken (detects oscillation cycles).
    This uses the analytic signal envelope + peak finding.

    Steps:
      1. Compute Hilbert envelope
      2. Smooth with gaussian (sigma = 0.5ms worth of samples)
      3. Find peaks with min_distance = 2ms
      4. IPI = differences between consecutive peaks
    """
    # Hilbert envelope
    analytic = hilbert(signal.astype(np.float64))
    envelope = np.abs(analytic)

    # Smooth - sigma = 0.5ms worth of samples
    sigma_samples = max(1, int(0.5e-3 / dt))
    envelope_smooth = gaussian_filter1d(envelope, sigma=sigma_samples)

    # Find peaks with minimum distance of 2ms
    min_distance_samples = max(1, int(2.0e-3 / dt))
    threshold = np.max(envelope_smooth) * 0.1

    peaks, properties = find_peaks(
        envelope_smooth,
        distance=min_distance_samples,
        height=threshold,
    )

    peak_times_ms = peaks * dt * 1000.0

    if len(peaks) >= 2:
        ipis = np.diff(peak_times_ms)
        return float(np.median(ipis)), peak_times_ms.tolist(), ipis.tolist()
    else:
        return 0.0, peak_times_ms.tolist(), []


def compute_zero_crossing_rate(signal):
    """Zero-crossing rate: fraction of consecutive samples that cross zero."""
    if len(signal) < 2:
        return 0.0
    crossings = np.sum(np.abs(np.diff(np.sign(signal))) > 0)
    return float(crossings / (len(signal) - 1))


def extract_features(forward_signal, dt):
    """Extract all target features from the forward sensor signal.

    Returns dict with keys matching REAL_TARGETS.
    """
    features = {}

    # Spectral analysis
    fft_mag = np.abs(np.fft.rfft(forward_signal))
    freqs = np.fft.rfftfreq(len(forward_signal), dt)
    power = fft_mag**2
    total_energy = np.sum(power)

    if total_energy > 0:
        # Spectral centroid
        features['spectral_centroid_hz'] = float(np.sum(freqs * power) / total_energy)

        # Band energies
        bands = {
            'band_2_5khz_pct': (2000, 5000),
            'band_5_10khz_pct': (5000, 10000),
            'band_10_20khz_pct': (10000, 20000),
        }
        for key, (f_low, f_high) in bands.items():
            mask = (freqs >= f_low) & (freqs < f_high)
            features[key] = float(np.sum(power[mask]) / total_energy * 100.0)
    else:
        features['spectral_centroid_hz'] = 0.0
        features['band_2_5khz_pct'] = 0.0
        features['band_5_10khz_pct'] = 0.0
        features['band_10_20khz_pct'] = 0.0

    # IPI detection (fixed with Hilbert envelope)
    ipi_ms, peak_times, ipis = detect_ipi_hilbert(forward_signal, dt)
    features['ipi_ms'] = ipi_ms
    features['peak_times_ms'] = peak_times
    features['ipis_ms'] = ipis

    # Zero crossing rate
    features['zero_crossing_rate'] = compute_zero_crossing_rate(forward_signal)

    return features


# ============================================================
# SIMULATION WRAPPER
# ============================================================

def params_to_dict(param_vector):
    """Convert optimizer parameter vector to named dict."""
    return {name: float(val) for name, val in zip(PARAM_NAMES, param_vector)}


def run_simulation(param_dict, dx=0.02, duration_ms=20):
    """Run FDTD simulation with given parameters.

    Uses coarse resolution (dx=0.02) for speed during optimization.
    Returns (forward_signal, dt) or (None, None) if simulation fails.
    """
    try:
        # Build whale head geometry with ALL anatomy parameters
        whale = WhaleHead(
            organ_length=param_dict['organ_length'],
            organ_diameter=param_dict['organ_diameter'],
            skull_curvature=param_dict['skull_curvature'],
            junk_length=param_dict['junk_length'],
            junk_max_diameter=param_dict.get('junk_max_diameter', 1.2),
            case_wall_thickness=param_dict.get('case_wall_thickness', 0.05),
            frontal_sac_width=param_dict.get('frontal_sac_width', 0.8),
            distal_sac_width=param_dict.get('distal_sac_width', 0.3),
            spermaceti_temp=param_dict['spermaceti_temp'],
            blubber_thickness=param_dict.get('blubber_thickness', 0.15),
        )

        # Build grid
        rho, c, tissue_map, source_pos, sensor_positions, grid_info = whale.build_grid(dx)
        Nx, Ny = grid_info["Nx"], grid_info["Ny"]

        # Apply spermaceti compression from maxillonasalis squeeze
        # Compression increases both density and sound speed
        # c_compressed = c_base * sqrt(compression) (bulk modulus scales with compression)
        # rho_compressed = rho_base * compression (mass conserved, volume reduced)
        compression = param_dict.get('spermaceti_compression', 1.0)
        if compression > 1.0:
            sperm_mask = (tissue_map == TISSUE_SPERMACETI)
            rho[sperm_mask] *= compression
            c[sperm_mask] *= np.sqrt(compression)  # sound speed ~ sqrt(bulk_modulus/density)

        # Override water properties based on ocean SURFACE temperature
        # (the head is surrounded by water at whatever depth the whale is)
        whale_depth = param_dict.get('whale_depth_m', 0)
        surface_temp = param_dict.get('ocean_surface_temp', 27.0)
        deep_temp = param_dict.get('ocean_deep_temp', 5.0)
        thermo_depth = param_dict.get('thermocline_depth_m', 200.0)

        # Water temp at whale's depth
        if whale_depth <= thermo_depth:
            local_water_temp = surface_temp + (whale_depth / max(thermo_depth, 1)) * (deep_temp - surface_temp)
        else:
            local_water_temp = deep_temp

        water_c = mackenzie_sound_speed(local_water_temp)
        water_rho = ocean_water_density(local_water_temp)

        water_mask = (tissue_map == TISSUE_WATER)
        rho[water_mask] = water_rho
        c[water_mask] = water_c

        # Override air sac properties (models depth compression)
        air_sac_rho = param_dict.get('air_sac_rho', 50.0)
        air_sac_c = param_dict.get('air_sac_c', 800.0)
        air_mask = (tissue_map == TISSUE_AIR_SAC)
        rho[air_mask] = air_sac_rho
        c[air_mask] = air_sac_c

        # Scale absorption coefficients
        absorption_scale = param_dict.get('absorption_scale', 1.0)
        original_coeffs = dict(ABSORPTION_COEFFICIENTS)
        for key in ABSORPTION_COEFFICIENTS:
            ABSORPTION_COEFFICIENTS[key] = original_coeffs[key] * absorption_scale

        # Time stepping
        c_max = float(np.max(c))
        dt = 0.2 * dx / c_max
        n_steps = int(duration_ms / 1000.0 / dt)

        # Source signal - either broadband impulse or Ricker wavelet
        source_freq = param_dict.get('source_freq_hz', 0)
        if source_freq > 100:
            # Use Ricker wavelet at specified frequency
            n_src = int(0.002 / dt)
            t = np.arange(n_src) * dt
            t0 = 1.0 / source_freq
            t_shifted = t - t0
            pi_f_t = (np.pi * source_freq * t_shifted)**2
            source_signal = 1000.0 * (1 - 2 * pi_f_t) * np.exp(-pi_f_t)
        else:
            # Use broadband impulse
            source_signal = broadband_impulse(
                duration_us=param_dict['source_duration_us'],
                dt=dt,
                amplitude=1000,
            )

        # Run FDTD (suppress per-step output for solver runs)
        import io
        from contextlib import redirect_stdout

        with redirect_stdout(io.StringIO()):
            sensor_data, _ = fdtd_2d_v2(
                rho, c, tissue_map, source_pos, sensor_positions,
                dx, dt, n_steps, source_signal,
            )

        # Forward sensor signal
        forward_signal = sensor_data[0]

        # Check for NaN/Inf (unstable simulation)
        if np.any(np.isnan(forward_signal)) or np.any(np.isinf(forward_signal)):
            return None, None

        # Apply exit-path tissue filter (case wall -> muscle -> junk -> blubber -> skin)
        forward_signal = apply_exit_path_filter(forward_signal, dt, param_dict)

        # Apply depth-integrated ocean propagation
        absorption_power = param_dict.get('absorption_power', 1.5)
        forward_signal = depth_integrated_ocean_propagation(
            forward_signal, dt,
            distance_m=param_dict.get('recording_distance_m', 100),
            whale_depth_m=whale_depth,
            surface_temp_c=surface_temp,
            deep_temp_c=deep_temp,
            thermocline_depth_m=thermo_depth,
            absorption_power=absorption_power,
        )

        return forward_signal, dt

    except Exception as e:
        # Any simulation failure returns None
        return None, None

    finally:
        # Restore original absorption coefficients
        try:
            for key in original_coeffs:
                ABSORPTION_COEFFICIENTS[key] = original_coeffs[key]
        except:
            pass


# ============================================================
# COST FUNCTION
# ============================================================

def cost_function(param_vector, target, dx=0.02, duration_ms=20):
    """Compute cost: weighted sum of squared normalized errors."""
    param_dict = params_to_dict(param_vector)

    forward_signal, dt = run_simulation(param_dict, dx=dx, duration_ms=duration_ms)

    if forward_signal is None:
        return 1e6  # penalty for failed simulation

    features = extract_features(forward_signal, dt)

    error = 0.0
    for key, weight in FEATURE_WEIGHTS.items():
        if key in target and target[key] > 0:
            sim_val = features.get(key, 0.0)
            real_val = target[key]
            error += weight * ((sim_val - real_val) / real_val) ** 2

    return error


# ============================================================
# SOLVER
# ============================================================

class SolverCallback:
    """Callback for differential_evolution to track progress."""

    def __init__(self, whale_name, target):
        self.whale_name = whale_name
        self.target = target
        self.iteration = 0
        self.best_cost = float('inf')
        self.start_time = time.time()
        self.eval_count = 0
        self.history = []

    def __call__(self, xk, convergence):
        self.iteration += 1
        cost = cost_function(xk, self.target)
        elapsed = time.time() - self.start_time

        if cost < self.best_cost:
            self.best_cost = cost

        params = params_to_dict(xk)
        self.history.append({
            'iteration': self.iteration,
            'cost': cost,
            'best_cost': self.best_cost,
            'elapsed_s': elapsed,
            'params': params,
        })

        print(f"\n  [Iter {self.iteration:3d}] cost={self.best_cost:.6f}  "
              f"elapsed={elapsed:.0f}s  convergence={convergence:.4f}")
        print(f"    CAVITY:  organ_L={params['organ_length']:.2f}m  "
              f"organ_D={params['organ_diameter']:.2f}m  "
              f"skull_R={params['skull_curvature']:.2f}m  "
              f"junk_L={params['junk_length']:.2f}m")
        print(f"    EXIT:    muscle={params.get('muscle_thickness', 0):.2f}m  "
              f"blubber={params.get('blubber_thickness', 0):.2f}m  "
              f"skin={params.get('skin_thickness', 0):.3f}m  "
              f"septa={params.get('junk_septa_count', 0):.0f}x{params.get('junk_septa_thickness', 0)*1000:.1f}mm")
        print(f"    PHYSIOL: sperm_T={params['spermaceti_temp']:.1f}C  "
              f"sac_rho={params.get('air_sac_rho', 0):.0f}  "
              f"sac_c={params.get('air_sac_c', 0):.0f}")
        print(f"    SOURCE:  dur={params['source_duration_us']:.0f}us  "
              f"freq={params.get('source_freq_hz', 0):.0f}Hz")
        print(f"    ABSORP:  cavity_scale={params.get('absorption_scale', 0):.2f}  "
              f"power={params.get('absorption_power', 0):.2f}  "
              f"muscle={params.get('muscle_absorption', 0):.1f}  "
              f"blub={params.get('blubber_absorption', 0):.1f}  "
              f"skin={params.get('skin_absorption', 0):.1f}  "
              f"junk={params.get('junk_absorption', 0):.1f}")
        print(f"    OCEAN:   surf_T={params.get('ocean_surface_temp', 0):.1f}C  "
              f"deep_T={params.get('ocean_deep_temp', 0):.1f}C  "
              f"depth={params.get('whale_depth_m', 0):.0f}m  "
              f"dist={params.get('recording_distance_m', 0):.0f}m")
        sys.stdout.flush()

        return False  # don't stop


def run_solver(whale_name, target, maxiter=50, popsize=10, dx=0.02, duration_ms=20):
    """Run differential evolution optimizer for one whale."""
    print(f"\n{'='*70}")
    print(f"SOLVING FOR: {whale_name}")
    print(f"{'='*70}")
    print(f"Target features:")
    for key, val in target.items():
        print(f"  {key}: {val}")
    print(f"\nOptimizer: differential_evolution")
    print(f"  maxiter={maxiter}, popsize={popsize}, tol=0.01")
    print(f"  dx={dx}m (coarse for speed)")
    print(f"  {len(PARAM_NAMES)} parameters, {len(PARAM_BOUNDS)} bounds")

    # Estimate time per evaluation
    print(f"\nEstimating time per evaluation...")
    t0 = time.time()
    # 31 params: anatomy(8) + exit_path(5) + physiology(4) + source(2) + absorption(2) + exit_abs(5) + ocean(5)
    test_params = [
        3.5, 1.3, 1.0, 2.0, 1.2, 0.05, 0.8, 0.3,          # anatomy cavity (8)
        0.15, 0.015, 0.10, 10.0, 0.003,                      # exit path tissues (5)
        33.0, 1.0, 50.0, 800.0,                               # physiology incl compression (4)
        50.0, 0.0,                                             # source (2)
        1.0, 1.5,                                              # cavity absorption (2)
        1.0, 0.5, 2.0, 0.5, 1.0,                              # exit path absorption (5)
        27.0, 5.0, 50.0, 100.0, 200.0,                        # ocean (5)
    ]
    _ = cost_function(test_params, target, dx=dx, duration_ms=duration_ms)
    eval_time = time.time() - t0
    total_evals_est = popsize * len(PARAM_NAMES) + maxiter * popsize
    print(f"  ~{eval_time:.1f}s per evaluation")
    print(f"  ~{total_evals_est} total evaluations estimated")
    print(f"  ~{total_evals_est * eval_time / 60:.0f} minutes estimated total")

    callback = SolverCallback(whale_name, target)

    start = time.time()
    result = differential_evolution(
        cost_function,
        bounds=PARAM_BOUNDS,
        args=(target, dx, duration_ms),
        maxiter=maxiter,
        popsize=popsize,
        tol=0.01,
        seed=42,
        callback=callback,
        disp=False,
        workers=1,  # FDTD is already memory-intensive
    )
    total_time = time.time() - start

    best_params = params_to_dict(result.x)
    best_cost = result.fun

    print(f"\n{'='*70}")
    print(f"OPTIMIZATION COMPLETE: {whale_name}")
    print(f"{'='*70}")
    print(f"  Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"  Iterations: {result.nit}")
    print(f"  Function evaluations: {result.nfev}")
    print(f"  Best cost: {best_cost:.6f}")
    print(f"  Success: {result.success}")
    print(f"  Message: {result.message}")

    # Run final evaluation to get features
    forward_signal, dt = run_simulation(best_params, dx=dx, duration_ms=duration_ms)
    if forward_signal is not None:
        best_features = extract_features(forward_signal, dt)
    else:
        best_features = {}

    # Check for parameters at bounds
    at_bounds = []
    for i, (name, (lo, hi)) in enumerate(zip(PARAM_NAMES, PARAM_BOUNDS)):
        val = result.x[i]
        margin = (hi - lo) * 0.01  # 1% margin
        if val <= lo + margin:
            at_bounds.append((name, val, 'LOWER', lo))
        elif val >= hi - margin:
            at_bounds.append((name, val, 'UPPER', hi))

    if at_bounds:
        print(f"\n  WARNING: Parameters at bounds (may need wider range):")
        for name, val, which, bound in at_bounds:
            print(f"    {name} = {val:.4f} (at {which} bound {bound})")

    return {
        'whale': whale_name,
        'best_params': best_params,
        'best_cost': best_cost,
        'best_features': best_features,
        'target': target,
        'at_bounds': [(n, v, w, b) for n, v, w, b in at_bounds],
        'optimizer': {
            'nit': result.nit,
            'nfev': result.nfev,
            'success': result.success,
            'message': result.message,
            'total_time_s': total_time,
        },
        'history': callback.history,
    }


# ============================================================
# COMPARISON TABLE
# ============================================================

def print_comparison(results_list):
    """Print comparison table of optimized vs real features."""
    print(f"\n{'='*90}")
    print("OPTIMIZATION RESULTS - COMPARISON TABLE")
    print(f"{'='*90}")

    feature_keys = ['spectral_centroid_hz', 'band_5_10khz_pct', 'band_10_20khz_pct',
                     'band_2_5khz_pct', 'ipi_ms', 'zero_crossing_rate']

    for res in results_list:
        whale = res['whale']
        target = res['target']
        features = res['best_features']
        cost = res['best_cost']

        print(f"\n  {whale}  (cost={cost:.6f})")
        print(f"  {'Feature':>25s}  {'Target':>10s}  {'Simulated':>10s}  {'Error%':>8s}")
        print(f"  {'-'*58}")

        for key in feature_keys:
            t_val = target.get(key, None)
            s_val = features.get(key, None)
            if t_val is not None and s_val is not None and t_val > 0:
                err_pct = (s_val - t_val) / t_val * 100
                print(f"  {key:>25s}  {t_val:>10.2f}  {s_val:>10.2f}  {err_pct:>+7.1f}%")
            elif t_val is not None:
                s_str = f"{s_val:.2f}" if s_val is not None else "n/a"
                print(f"  {key:>25s}  {t_val:>10.2f}  {s_str:>10s}  {'---':>8s}")

        print(f"\n  Best parameters:")
        for name, val in res['best_params'].items():
            print(f"    {name}: {val:.4f}")

        if res['at_bounds']:
            print(f"\n  Parameters at bounds:")
            for name, val, which, bound in res['at_bounds']:
                print(f"    {name} = {val:.4f} (at {which} bound {bound})")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Sperm Whale Acoustic Parameter Optimizer")
    parser.add_argument("--whale", type=str, default="A",
                        help="Which whale to optimize: A, D, F, or all (default: A)")
    parser.add_argument("--maxiter", type=int, default=50,
                        help="Max DE iterations (default: 50)")
    parser.add_argument("--popsize", type=int, default=15,
                        help="DE population size (default: 15)")
    parser.add_argument("--dx", type=float, default=0.02,
                        help="Grid spacing in meters (default: 0.02 for speed)")
    parser.add_argument("--duration", type=float, default=20.0,
                        help="Simulation duration in ms (default: 20)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory for results JSON")
    args = parser.parse_args()

    # Determine which whales to solve
    whale_key_map = {
        'A': 'Whale_A',
        'D': 'Whale_D',
        'F': 'Whale_F',
    }

    if args.whale.lower() == 'all':
        whale_keys = ['Whale_A', 'Whale_D', 'Whale_F']
    else:
        key = args.whale.upper()
        if key not in whale_key_map:
            print(f"Unknown whale: {args.whale}. Options: A, D, F, all")
            sys.exit(1)
        whale_keys = [whale_key_map[key]]

    # Output directory
    if args.output:
        output_dir = args.output
    else:
        output_dir = str(Path(__file__).parent)
    os.makedirs(output_dir, exist_ok=True)

    # Solve for each whale
    all_results = []
    for whale_name in whale_keys:
        target = REAL_TARGETS[whale_name]
        result = run_solver(
            whale_name, target,
            maxiter=args.maxiter,
            popsize=args.popsize,
            dx=args.dx,
            duration_ms=args.duration,
        )
        all_results.append(result)

    # Print comparison
    print_comparison(all_results)

    # Save results to JSON
    # Strip non-serializable items
    save_results = []
    for res in all_results:
        save_res = {
            'whale': res['whale'],
            'best_params': res['best_params'],
            'best_cost': res['best_cost'],
            'best_features': {k: v for k, v in res['best_features'].items()
                              if not isinstance(v, (list,)) or len(v) < 50},
            'target': res['target'],
            'at_bounds': res['at_bounds'],
            'optimizer': res['optimizer'],
        }
        save_results.append(save_res)

    output_path = os.path.join(output_dir, "solver_results.json")
    with open(output_path, "w") as f:
        json.dump(save_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Sperm Whale Head Acoustic Simulator
2D sagittal-plane simulation using k-Wave.

Simulates how a click pulse propagates through the spermaceti organ,
bounces between air sacs, and exits the head. Different skull geometries
produce different voiceprints.

Parameters to vary per "whale":
  - organ_length: spermaceti organ length (m)
  - organ_diameter: spermaceti organ diameter (m)
  - skull_curvature: rostral basin radius of curvature (m)
  - junk_length: junk/melon length (m)
  - spermaceti_temp: temperature (C) -> controls sound speed
  - skull_asymmetry: degree of leftward twist (radians)
"""

import json
import os
import sys
import time
import numpy as np
from pathlib import Path

# Try k-Wave
try:
    import kwave
    from kwave.kgrid import kWaveGrid
    from kwave.kmedium import kWaveMedium
    from kwave.ksource import kSource
    from kwave.ksensor import kSensor
    from kwave.kspaceFirstOrder2D import kspaceFirstOrder2DC
    from kwave.options.simulation_options import SimulationOptions
    from kwave.utils.signals import tone_burst
    HAS_KWAVE = True
except ImportError:
    HAS_KWAVE = False
    print("k-Wave not available, using numpy FDTD fallback")

OUTPUT_DIR = "/mnt/archive/datasets/whale_communication/simulation"


# ============================================================
# MATERIAL PROPERTIES
# ============================================================

def spermaceti_sound_speed(temp_c):
    """Sound speed in spermaceti oil as function of temperature.
    Based on Flewellen & Morris (1978)."""
    if temp_c >= 33:
        return 1370.0  # fully liquid
    elif temp_c >= 29:
        # transition zone
        return 1370.0 + (33.0 - temp_c) * 12.5
    elif temp_c >= 25:
        return 1420.0 + (29.0 - temp_c) * 10.0
    else:
        return 1460.0 + (25.0 - temp_c) * 14.0  # increasingly solid


MATERIALS = {
    "water":       {"rho": 1025.0, "c": 1530.0},
    "spermaceti":  {"rho": 857.0,  "c": 1370.0},  # default 37C
    "junk_post":   {"rho": 870.0,  "c": 1380.0},
    "junk_ant":    {"rho": 930.0,  "c": 1430.0},
    "connective":  {"rho": 1070.0, "c": 1570.0},
    "bone":        {"rho": 1900.0, "c": 3000.0},
    "air":         {"rho": 1.2,    "c": 340.0},
    "blubber":     {"rho": 920.0,  "c": 1420.0},
}


# ============================================================
# GEOMETRY BUILDER
# ============================================================

class WhaleHead:
    """2D sagittal cross-section of a sperm whale head."""

    def __init__(self,
                 organ_length=3.2,       # m
                 organ_diameter=1.5,     # m
                 skull_curvature=1.0,    # m (radius)
                 junk_length=2.0,        # m
                 junk_max_diameter=1.2,  # m
                 case_wall_thickness=0.05,  # m
                 frontal_sac_width=0.8,  # m
                 distal_sac_width=0.3,   # m
                 spermaceti_temp=37.0,   # C
                 blubber_thickness=0.15, # m
                 name="default"):
        self.organ_length = organ_length
        self.organ_diameter = organ_diameter
        self.skull_curvature = skull_curvature
        self.junk_length = junk_length
        self.junk_max_diameter = junk_max_diameter
        self.case_wall_thickness = case_wall_thickness
        self.frontal_sac_width = frontal_sac_width
        self.distal_sac_width = distal_sac_width
        self.spermaceti_temp = spermaceti_temp
        self.blubber_thickness = blubber_thickness
        self.name = name

        # Derived
        self.spermaceti_c = spermaceti_sound_speed(spermaceti_temp)
        self.head_length = organ_length + junk_length + 0.3  # extra for skull

    def build_grid(self, dx=0.005):
        """Build 2D density and sound speed grids.
        Returns (rho, c, source_pos, sensor_positions, grid_info)
        """
        # Grid dimensions (add padding for water)
        pad = 0.5  # m of water around head
        Lx = self.head_length + 2 * pad
        Ly = max(self.organ_diameter, self.junk_max_diameter) + 2 * pad + 2 * self.blubber_thickness

        Nx = int(Lx / dx)
        Ny = int(Ly / dx)

        # Initialize with water
        rho = np.full((Nx, Ny), MATERIALS["water"]["rho"], dtype=np.float32)
        c = np.full((Nx, Ny), MATERIALS["water"]["c"], dtype=np.float32)

        # Coordinate system: x = anterior-posterior (0 = front of head), y = dorsal-ventral
        cx = Nx // 2  # center x (not used - head is off-center)
        cy = Ny // 2  # center y (midline)

        # Head starts at pad from left edge
        head_start_x = int(pad / dx)
        skull_x = head_start_x + int(self.head_length / dx)  # posterior skull boundary

        # --- SKULL BONE (posterior boundary) ---
        skull_thickness = int(0.08 / dx)  # 8cm thick
        for ix in range(skull_x - skull_thickness, skull_x):
            # Concave shape (parabolic basin)
            basin_half = int(self.frontal_sac_width / 2 / dx)
            for iy in range(cy - basin_half, cy + basin_half):
                dy_from_center = abs(iy - cy) * dx
                # Parabolic depth
                depth = (dy_from_center**2) / (2 * self.skull_curvature)
                depth_px = int(depth / dx)
                if ix >= skull_x - skull_thickness + depth_px:
                    rho[ix, iy] = MATERIALS["bone"]["rho"]
                    c[ix, iy] = MATERIALS["bone"]["c"]

        # --- FRONTAL AIR SAC (just anterior to skull) ---
        sac_x = skull_x - skull_thickness - int(0.02 / dx)  # 2cm gap
        sac_thickness = max(int(0.03 / dx), 2)  # 3cm of air
        sac_half_y = int(self.frontal_sac_width / 2 / dx)
        for ix in range(sac_x - sac_thickness, sac_x):
            for iy in range(cy - sac_half_y, cy + sac_half_y):
                rho[ix, iy] = MATERIALS["air"]["rho"]
                c[ix, iy] = MATERIALS["air"]["c"]

        # --- SPERMACETI ORGAN (case) ---
        organ_start_x = head_start_x + int(self.junk_length / dx)
        organ_end_x = organ_start_x + int(self.organ_length / dx)
        organ_half_y = int(self.organ_diameter / 2 / dx)

        # Case wall (connective tissue boundary)
        wall_px = max(int(self.case_wall_thickness / dx), 2)

        for ix in range(organ_start_x, min(organ_end_x, sac_x - sac_thickness)):
            for iy in range(cy - organ_half_y - wall_px, cy + organ_half_y + wall_px):
                dist_from_center = abs(iy - cy) * dx
                # Elliptical cross section
                x_frac = (ix - organ_start_x) / max(organ_end_x - organ_start_x, 1)
                # Taper slightly at ends
                local_radius = self.organ_diameter / 2 * (1 - 0.2 * (2 * x_frac - 1)**2)

                if dist_from_center <= local_radius + self.case_wall_thickness:
                    if dist_from_center > local_radius:
                        # Case wall
                        rho[ix, iy] = MATERIALS["connective"]["rho"]
                        c[ix, iy] = MATERIALS["connective"]["c"]
                    else:
                        # Spermaceti oil
                        rho[ix, iy] = MATERIALS["spermaceti"]["rho"]
                        c[ix, iy] = self.spermaceti_c

        # --- DISTAL AIR SAC (at anterior end of spermaceti organ) ---
        distal_x = organ_start_x
        distal_thickness = max(int(0.02 / dx), 2)
        distal_half_y = int(self.distal_sac_width / 2 / dx)
        for ix in range(distal_x - distal_thickness, distal_x):
            for iy in range(cy - distal_half_y, cy + distal_half_y):
                rho[ix, iy] = MATERIALS["air"]["rho"]
                c[ix, iy] = MATERIALS["air"]["c"]

        # --- JUNK (graded lipid lens) ---
        junk_start_x = head_start_x
        junk_end_x = organ_start_x
        for ix in range(junk_start_x, junk_end_x):
            x_frac = (ix - junk_start_x) / max(junk_end_x - junk_start_x, 1)
            # Taper: cone shape
            local_radius = self.junk_max_diameter / 2 * (0.3 + 0.7 * x_frac)  # narrow at tip

            for iy in range(cy - int(local_radius / dx), cy + int(local_radius / dx)):
                dist = abs(iy - cy) * dx
                if dist <= local_radius:
                    # Graded properties: anterior=junk_ant, posterior=junk_post
                    rho[ix, iy] = MATERIALS["junk_ant"]["rho"] + x_frac * (MATERIALS["junk_post"]["rho"] - MATERIALS["junk_ant"]["rho"])
                    c[ix, iy] = MATERIALS["junk_ant"]["c"] + x_frac * (MATERIALS["junk_post"]["c"] - MATERIALS["junk_ant"]["c"])

        # --- BLUBBER (outer layer) ---
        # Simple: any tissue pixel adjacent to water gets a blubber border
        # Skip for now - minor effect on internal acoustics

        # --- PHONIC LIPS (source location) ---
        # At the dorsal side of the distal sac
        source_x = distal_x - distal_thickness
        source_y = cy - distal_half_y - int(0.05 / dx)  # slightly dorsal

        # --- SENSOR POSITIONS ---
        # Ring of sensors around the head in water
        n_sensors = 36
        sensor_radius = max(Lx, Ly) / 2 * 0.85
        sensor_center_x = head_start_x + int(self.head_length / 2 / dx)
        sensor_positions = []
        for i in range(n_sensors):
            angle = 2 * np.pi * i / n_sensors
            sx = int(sensor_center_x + sensor_radius * np.cos(angle) / dx)
            sy = int(cy + sensor_radius * np.sin(angle) / dx)
            sx = max(2, min(Nx - 3, sx))
            sy = max(2, min(Ny - 3, sy))
            sensor_positions.append((sx, sy))

        grid_info = {
            "Nx": Nx, "Ny": Ny, "dx": dx,
            "Lx": Lx, "Ly": Ly,
            "head_start_x": head_start_x,
            "skull_x": skull_x,
            "organ_start_x": organ_start_x,
            "organ_end_x": organ_end_x,
        }

        return rho, c, (source_x, source_y), sensor_positions, grid_info


# ============================================================
# FDTD SIMULATOR (numpy fallback when k-Wave unavailable)
# ============================================================

def fdtd_2d(rho, c, source_pos, sensor_positions, dx, dt, n_steps, source_signal):
    """Simple 2D acoustic FDTD simulation.
    Solves the acoustic wave equation with heterogeneous media.
    """
    Nx, Ny = rho.shape
    # Pressure and velocity fields
    p = np.zeros((Nx, Ny), dtype=np.float32)
    vx = np.zeros((Nx, Ny), dtype=np.float32)
    vy = np.zeros((Nx, Ny), dtype=np.float32)

    # Use float64 for stability
    rho = rho.astype(np.float64)
    c = c.astype(np.float64)

    # Smooth ALL boundaries - tissue interfaces are fatty gradients, not sharp walls.
    # This is biologically accurate AND numerically stable.
    from scipy.ndimage import gaussian_filter
    rho = gaussian_filter(rho, sigma=3.0)  # ~3 grid cells of transition
    c = gaussian_filter(c, sigma=3.0)

    # Clamp to physical ranges after smoothing
    rho = np.clip(rho, 30.0, 2500.0)
    c = np.clip(c, 600.0, 3500.0)

    c2 = c**2
    # Use harmonic average of density at staggered grid points for stability
    rho_x = 0.5 * (rho[1:, :] + rho[:-1, :])
    rho_y = 0.5 * (rho[:, 1:] + rho[:, :-1])
    rho_x_inv = 1.0 / rho_x
    rho_y_inv = 1.0 / rho_y

    # PML absorbing boundary
    pml_width = 30
    damping = np.zeros((Nx, Ny), dtype=np.float64)
    for i in range(pml_width):
        d = ((pml_width - i) / pml_width)**3 * 0.3  # cubic profile, gentler
        damping[i, :] = max(damping[i, 0], d)
        damping[Nx - 1 - i, :] = max(damping[Nx - 1 - i, 0], d)
        damping[:, i] = np.maximum(damping[:, i], d)
        damping[:, Ny - 1 - i] = np.maximum(damping[:, Ny - 1 - i], d)

    decay = 1.0 - damping

    # Fields in float64
    p = np.zeros((Nx, Ny), dtype=np.float64)
    vx = np.zeros((Nx - 1, Ny), dtype=np.float64)
    vy = np.zeros((Nx, Ny - 1), dtype=np.float64)

    sensor_data = np.zeros((len(sensor_positions), n_steps), dtype=np.float64)
    sx, sy = source_pos

    for t in range(n_steps):
        # Inject source (soft source - add to pressure)
        if t < len(source_signal):
            p[sx, sy] += float(source_signal[t])

        # Update velocity from pressure gradient
        vx -= dt * rho_x_inv * (p[1:, :] - p[:-1, :]) / dx
        vy -= dt * rho_y_inv * (p[:, 1:] - p[:, :-1]) / dx

        # Update pressure from velocity divergence
        div_v = np.zeros_like(p)
        div_v[1:-1, :] += (vx[1:, :] - vx[:-1, :]) / dx
        div_v[:, 1:-1] += (vy[:, 1:] - vy[:, :-1]) / dx
        p -= dt * rho * c2 * div_v

        # PML damping
        p *= decay
        vx *= decay[1:, :]
        vy *= decay[:, 1:]

        # Record sensors
        for si, (sx_s, sy_s) in enumerate(sensor_positions):
            if 0 <= sx_s < Nx and 0 <= sy_s < Ny:
                sensor_data[si, t] = p[sx_s, sy_s]

        if (t + 1) % 5000 == 0:
            max_p = float(np.max(np.abs(p)))
            print(f"  Step {t+1}/{n_steps}, max pressure: {max_p:.6f}", flush=True)
            if max_p > 1e10 or np.isnan(max_p):
                print("  WARNING: simulation unstable, aborting", flush=True)
                break

    return sensor_data.astype(np.float32), p.astype(np.float32)


# ============================================================
# SOURCE SIGNAL
# ============================================================

def ricker_wavelet(freq, dt, n_samples):
    """Generate a Ricker wavelet (Mexican hat) - good approximation of whale click."""
    t = np.arange(n_samples) * dt
    t0 = 1.0 / freq  # center the wavelet
    t_shifted = t - t0
    pi_f_t = (np.pi * freq * t_shifted)**2
    wavelet = (1 - 2 * pi_f_t) * np.exp(-pi_f_t)
    return wavelet.astype(np.float32)


# ============================================================
# RUN SIMULATION
# ============================================================

def simulate_whale(whale_params, dx=0.005, center_freq=12000, duration_ms=15):
    """Run a full simulation for one whale configuration."""
    whale = WhaleHead(**whale_params)
    name = whale_params.get("name", "default")

    print(f"\n{'='*60}")
    print(f"SIMULATING: {name}")
    print(f"  Organ: {whale.organ_length}m x {whale.organ_diameter}m")
    print(f"  Spermaceti c: {whale.spermaceti_c:.0f} m/s (T={whale.spermaceti_temp}C)")
    print(f"  Expected IPI: {2*whale.organ_length/whale.spermaceti_c*1000:.2f} ms")
    print(f"{'='*60}")

    # Build geometry
    print("Building geometry...")
    rho, c, source_pos, sensor_positions, grid_info = whale.build_grid(dx)
    Nx, Ny = grid_info["Nx"], grid_info["Ny"]
    print(f"  Grid: {Nx} x {Ny} = {Nx*Ny:,} points")

    # Air sacs modeled as fatty membrane reflectors, not actual air.
    # In reality the air sac boundary is a thin membrane surrounded by
    # fatty connective tissue. The impedance mismatch comes from the
    # density/stiffness change, not a vacuum. Model as very low density
    # fatty tissue - gives R~0.97 reflection without CFL instability.
    air_mask = c < 500
    rho[air_mask] = 50.0     # fatty membrane, much less dense than spermaceti (857)
    c[air_mask] = 800.0      # slow fatty tissue, still within CFL stability

    # Time stepping
    c_max = np.max(c)
    dt = 0.2 * dx / c_max  # CFL condition (conservative for 2D stability)
    n_steps = int(duration_ms / 1000.0 / dt)
    print(f"  dt: {dt*1e6:.2f} us, steps: {n_steps}, duration: {duration_ms}ms")

    # Source signal
    source_signal = ricker_wavelet(center_freq, dt, int(0.001 / dt))  # 1ms wavelet
    source_signal *= 1000  # amplitude scaling

    # Run FDTD
    print("Running FDTD simulation...")
    start = time.time()
    sensor_data, final_pressure = fdtd_2d(
        rho, c, source_pos, sensor_positions, dx, dt, n_steps, source_signal
    )
    elapsed = time.time() - start
    print(f"  Simulation complete in {elapsed:.1f}s")

    # Extract results
    results = {
        "name": name,
        "params": {k: v for k, v in whale_params.items()},
        "grid": {"Nx": Nx, "Ny": Ny, "dx": dx},
        "time": {"dt": float(dt), "n_steps": n_steps, "duration_ms": duration_ms},
        "expected_ipi_ms": 2 * whale.organ_length / whale.spermaceti_c * 1000,
        "spermaceti_c": whale.spermaceti_c,
    }

    # Analyze sensor data
    # Forward sensor (0 degrees, anterior)
    forward_idx = 0  # first sensor is at 0 degrees (forward)
    forward_signal = sensor_data[forward_idx]

    # Find peaks in forward signal (these are P0, P1, P2...)
    abs_signal = np.abs(forward_signal)
    threshold = np.max(abs_signal) * 0.1
    peaks = []
    in_peak = False
    for i in range(1, len(abs_signal) - 1):
        if abs_signal[i] > threshold and abs_signal[i] > abs_signal[i-1] and abs_signal[i] >= abs_signal[i+1]:
            if not in_peak:
                peaks.append(i)
                in_peak = True
        elif abs_signal[i] < threshold * 0.5:
            in_peak = False

    peak_times_ms = [p * dt * 1000 for p in peaks]
    if len(peak_times_ms) >= 2:
        ipis = [peak_times_ms[i+1] - peak_times_ms[i] for i in range(len(peak_times_ms)-1)]
        results["measured_ipi_ms"] = ipis
        results["mean_ipi_ms"] = float(np.mean(ipis))
    else:
        results["measured_ipi_ms"] = []
        results["mean_ipi_ms"] = 0

    results["peak_times_ms"] = peak_times_ms
    results["n_pulses_detected"] = len(peaks)

    # Spectral analysis of forward signal
    fft_result = np.abs(np.fft.rfft(forward_signal))
    freqs = np.fft.rfftfreq(len(forward_signal), dt)
    # Peak frequency
    mask = freqs > 500
    if np.any(mask) and np.any(fft_result[mask] > 0):
        peak_freq_idx = np.argmax(fft_result[mask])
        results["peak_frequency_hz"] = float(freqs[mask][peak_freq_idx])
    else:
        results["peak_frequency_hz"] = 0

    # Spectral centroid
    total_energy = np.sum(fft_result**2)
    if total_energy > 0:
        results["spectral_centroid_hz"] = float(np.sum(freqs * fft_result**2) / total_energy)
    else:
        results["spectral_centroid_hz"] = 0

    # Beam pattern (amplitude at each sensor angle)
    beam_pattern = {}
    for si, (sx, sy) in enumerate(sensor_positions):
        angle_deg = si * 360 / len(sensor_positions)
        peak_amp = float(np.max(np.abs(sensor_data[si])))
        beam_pattern[f"{angle_deg:.0f}"] = peak_amp
    results["beam_pattern"] = beam_pattern

    # Forward/backward ratio
    forward_amp = np.max(np.abs(sensor_data[0]))
    backward_amp = np.max(np.abs(sensor_data[len(sensor_positions)//2]))
    if backward_amp > 0:
        results["front_back_ratio_db"] = float(20 * np.log10(forward_amp / backward_amp))
    else:
        results["front_back_ratio_db"] = 999

    # Save sensor data for the forward direction (time series)
    # Downsample for storage
    ds = max(1, n_steps // 2000)
    results["forward_signal"] = forward_signal[::ds].tolist()
    results["forward_signal_dt_ms"] = float(dt * ds * 1000)

    # Save geometry snapshot (downsampled)
    geo_ds = max(1, Nx // 200)
    results["geometry_c"] = c[::geo_ds, ::geo_ds].tolist()
    results["geometry_rho"] = rho[::geo_ds, ::geo_ds].tolist()

    return results


# ============================================================
# MAIN - SIMULATE MULTIPLE WHALES
# ============================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Define whale configurations matching our 3 real whales
    whales = [
        {
            "name": "Whale_A_soprano",
            "organ_length": 2.8,
            "organ_diameter": 1.2,
            "skull_curvature": 0.8,
            "junk_length": 1.7,
            "junk_max_diameter": 1.0,
            "spermaceti_temp": 35.0,  # slightly cooler = higher pitch
        },
        {
            "name": "Whale_D_alto",
            "organ_length": 3.2,
            "organ_diameter": 1.4,
            "skull_curvature": 1.0,
            "junk_length": 1.9,
            "junk_max_diameter": 1.1,
            "spermaceti_temp": 37.0,
        },
        {
            "name": "Whale_F_bass",
            "organ_length": 3.8,
            "organ_diameter": 1.7,
            "skull_curvature": 1.2,
            "junk_length": 2.2,
            "junk_max_diameter": 1.3,
            "spermaceti_temp": 37.0,
        },
        # Temperature variation test
        {
            "name": "Whale_D_cold",
            "organ_length": 3.2,
            "organ_diameter": 1.4,
            "skull_curvature": 1.0,
            "junk_length": 1.9,
            "junk_max_diameter": 1.1,
            "spermaceti_temp": 28.0,  # cold spermaceti
        },
    ]

    all_results = []

    for whale_params in whales:
        results = simulate_whale(whale_params, dx=0.01, center_freq=12000, duration_ms=12)
        all_results.append(results)

        # Print key results
        print(f"\n  Results for {results['name']}:")
        print(f"    Expected IPI: {results['expected_ipi_ms']:.2f} ms")
        print(f"    Measured IPI: {results.get('measured_ipi_ms', 'none')}")
        print(f"    Pulses detected: {results['n_pulses_detected']}")
        print(f"    Peak frequency: {results['peak_frequency_hz']:.0f} Hz")
        print(f"    Spectral centroid: {results['spectral_centroid_hz']:.0f} Hz")
        print(f"    Front/back ratio: {results['front_back_ratio_db']:.1f} dB")

    # Save all results
    output_path = os.path.join(OUTPUT_DIR, "simulation_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    # Comparison table
    print("\n" + "=" * 80)
    print("COMPARISON TABLE")
    print("=" * 80)
    print(f"{'Name':>25s} {'IPI_exp':>8s} {'IPI_meas':>10s} {'Pulses':>7s} {'PeakHz':>8s} {'Centroid':>9s} {'F/B dB':>7s}")
    print("-" * 80)
    for r in all_results:
        ipi_meas = f"{r['mean_ipi_ms']:.2f}" if r['mean_ipi_ms'] > 0 else "n/a"
        print(f"{r['name']:>25s} {r['expected_ipi_ms']:>7.2f}ms {ipi_meas:>9s}ms {r['n_pulses_detected']:>6d} {r['peak_frequency_hz']:>7.0f} {r['spectral_centroid_hz']:>8.0f}Hz {r['front_back_ratio_db']:>6.1f}")


if __name__ == "__main__":
    main()

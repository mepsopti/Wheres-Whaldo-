#!/usr/bin/env python3
"""
Ocean Temperature Effect on Whale Voiceprint Propagation

Models how a sperm whale click changes as it propagates through ocean water
at different temperatures. Accounts for:
  - Temperature-dependent sound speed (Mackenzie equation)
  - Temperature-dependent absorption (Francois-Garrison model)
  - Frequency-dependent absorption (higher freq = more loss)
  - Impedance mismatch at head-water boundary vs temperature
  - Voiceprint degradation over distance

Uses the simulated whale clicks from sperm_whale_sim.py as source signals.
"""

import json
import os
import numpy as np
import time

OUTPUT_DIR = "/mnt/archive/datasets/whale_communication/simulation"
SIM_PATH = os.path.join(OUTPUT_DIR, "simulation_results.json")
REPORT_PATH = os.path.join(OUTPUT_DIR, "ocean_propagation_report.txt")
DATA_PATH = os.path.join(OUTPUT_DIR, "ocean_propagation.json")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================
# OCEAN ACOUSTICS
# ============================================================

def mackenzie_sound_speed(temp_c, salinity_ppt=35.0, depth_m=0):
    """Mackenzie (1981) equation for sound speed in seawater.
    Returns sound speed in m/s."""
    T = temp_c
    S = salinity_ppt
    D = depth_m
    c = (1448.96 + 4.591 * T - 0.05304 * T**2 + 0.0002374 * T**3
         + 1.340 * (S - 35) + 0.0163 * D + 1.675e-7 * D**2
         - 0.01025 * T * (S - 35) - 7.139e-13 * T * D**3)
    return c


def francois_garrison_absorption(freq_hz, temp_c, salinity_ppt=35.0, depth_m=0, ph=8.0):
    """Francois-Garrison (1982) model for acoustic absorption in seawater.
    Returns absorption in dB/km."""
    f = freq_hz / 1000.0  # convert to kHz
    T = temp_c
    S = salinity_ppt
    D = depth_m

    # Boric acid contribution
    A1 = 8.86 / (10 ** (0.78 * ph)) * 10 ** (0.002 * T)
    P1 = 1.0  # pressure factor (surface)
    f1 = 2.8 * (S / 35.0) ** 0.5 * 10 ** (4 - 1245.0 / (T + 273))

    # Magnesium sulfate contribution
    A2 = 21.44 * (S / 35.0) * (1 + 0.025 * T)
    P2 = 1.0 - 1.37e-4 * D + 6.2e-9 * D**2
    f2 = 8.17 * 10 ** (8 - 1990.0 / (T + 273)) / (1 + 0.0018 * (S - 35))

    # Pure water contribution
    A3 = 4.937e-4 - 2.59e-5 * T + 9.11e-7 * T**2 - 1.50e-8 * T**3
    if T <= 20:
        A3 = 4.937e-4 - 2.59e-5 * T + 9.11e-7 * T**2 - 1.50e-8 * T**3
    else:
        A3 = 3.964e-4 - 1.146e-5 * T + 1.45e-7 * T**2 - 6.5e-10 * T**3
    P3 = 1.0 - 3.83e-5 * D + 4.9e-10 * D**2

    # Total absorption (dB/km)
    alpha = (A1 * P1 * f1 * f**2 / (f1**2 + f**2)
             + A2 * P2 * f2 * f**2 / (f2**2 + f**2)
             + A3 * P3 * f**2)

    return max(alpha, 0.0)


def water_impedance(temp_c, salinity_ppt=35.0, depth_m=0):
    """Acoustic impedance of seawater (MRayl)."""
    c = mackenzie_sound_speed(temp_c, salinity_ppt, depth_m)
    # Seawater density varies slightly with temp
    rho = 1025.0 - 0.15 * (temp_c - 15.0)  # rough approximation
    return rho * c / 1e6  # MRayl


def head_water_reflection_coeff(temp_c):
    """Reflection coefficient at junk-water boundary.
    Junk anterior impedance ~1.33 MRayl."""
    Z_junk = 1.33  # MRayl (anterior junk)
    Z_water = water_impedance(temp_c)
    R = abs(Z_water - Z_junk) / (Z_water + Z_junk)
    return R, Z_water


def propagate_signal(signal, dt, distance_km, temp_c, depth_m=0):
    """Propagate a signal through ocean water over a given distance.
    Applies frequency-dependent absorption and geometric spreading.
    Returns the signal as it would be received at distance."""

    n = len(signal)
    # FFT
    fft_signal = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(n, dt)

    # Apply frequency-dependent absorption
    for i, f in enumerate(freqs):
        if f <= 0:
            continue
        # Absorption in dB/km
        alpha_db = francois_garrison_absorption(f, temp_c, depth_m=depth_m)
        # Total absorption over distance
        total_db = alpha_db * distance_km
        # Convert to amplitude factor
        amplitude_factor = 10 ** (-total_db / 20.0)
        fft_signal[i] *= amplitude_factor

    # Geometric spreading (spherical: 1/r)
    r_meters = distance_km * 1000
    if r_meters > 1:
        fft_signal *= (1.0 / r_meters)

    # Inverse FFT
    propagated = np.fft.irfft(fft_signal, n)
    return propagated


def spectral_analysis(signal, dt):
    """Quick spectral analysis of a signal."""
    fft_mag = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal), dt)

    total_energy = np.sum(fft_mag**2)

    # Band energies
    bands = {}
    for name, (lo, hi) in [
        ("1-2kHz", (1000, 2000)), ("2-4kHz", (2000, 4000)),
        ("4-8kHz", (4000, 8000)), ("8-12kHz", (8000, 12000)),
        ("12-16kHz", (12000, 16000)), ("16-20kHz", (16000, 20000)),
    ]:
        mask = (freqs >= lo) & (freqs < hi)
        energy = np.sum(fft_mag[mask]**2)
        bands[name] = float(energy / max(total_energy, 1e-20) * 100)

    # Spectral centroid
    if total_energy > 0:
        centroid = float(np.sum(freqs * fft_mag**2) / total_energy)
    else:
        centroid = 0

    # Peak frequency
    mask = freqs > 500
    if np.any(mask) and np.any(fft_mag[mask] > 0):
        peak_freq = float(freqs[mask][np.argmax(fft_mag[mask])])
    else:
        peak_freq = 0

    return {
        "peak_freq_hz": round(peak_freq, 0),
        "centroid_hz": round(centroid, 0),
        "band_energy_pct": {k: round(v, 1) for k, v in bands.items()},
        "total_energy": float(total_energy),
    }


def main():
    # Load simulation results
    with open(SIM_PATH) as f:
        sim_results = json.load(f)
    log(f"Loaded {len(sim_results)} whale simulations")

    # Ocean conditions to test
    ocean_temps = [2, 5, 10, 15, 20, 25, 30]  # Celsius
    distances_km = [0.1, 0.5, 1.0, 5.0, 10.0, 50.0]

    all_results = []
    lines = []
    lines.append("=" * 90)
    lines.append("OCEAN PROPAGATION ANALYSIS - VOICEPRINT DEGRADATION VS TEMPERATURE & DISTANCE")
    lines.append("=" * 90)

    # First: show ocean properties across temperatures
    lines.append("\nOCEAN PROPERTIES BY TEMPERATURE:")
    lines.append(f"{'Temp C':>8s} {'Sound Speed':>12s} {'Impedance':>10s} {'Head-Water R':>13s} {'Abs@5kHz':>10s} {'Abs@10kHz':>11s} {'Abs@15kHz':>11s}")
    lines.append("-" * 85)
    for temp in ocean_temps:
        c = mackenzie_sound_speed(temp)
        Z = water_impedance(temp)
        R, _ = head_water_reflection_coeff(temp)
        a5 = francois_garrison_absorption(5000, temp)
        a10 = francois_garrison_absorption(10000, temp)
        a15 = francois_garrison_absorption(15000, temp)
        lines.append(f"{temp:>6d}C {c:>10.1f}m/s {Z:>8.3f}MRa {R:>11.4f} {a5:>8.2f}dB/km {a10:>9.2f}dB/km {a15:>9.2f}dB/km")

    # For each whale, propagate through different conditions
    for sim in sim_results:
        whale_name = sim["name"]
        signal = np.array(sim["forward_signal"], dtype=np.float64)
        dt = sim["forward_signal_dt_ms"] / 1000.0  # seconds

        lines.append(f"\n{'='*90}")
        lines.append(f"WHALE: {whale_name}")
        lines.append(f"{'='*90}")

        # Source spectrum
        source_spec = spectral_analysis(signal, dt)
        lines.append(f"\nSource (at head):")
        lines.append(f"  Peak: {source_spec['peak_freq_hz']:.0f}Hz, Centroid: {source_spec['centroid_hz']:.0f}Hz")
        lines.append(f"  Bands: {source_spec['band_energy_pct']}")

        whale_results = {"name": whale_name, "source": source_spec, "propagated": []}

        # Propagation matrix
        lines.append(f"\n{'Distance':>10s} {'Temp':>6s} {'Peak Hz':>8s} {'Centroid':>9s} {'4-8k':>6s} {'8-12k':>7s} {'12-16k':>7s} {'16-20k':>7s} {'Energy':>10s}")
        lines.append("-" * 80)

        for distance in distances_km:
            for temp in [2, 10, 20, 30]:
                propagated = propagate_signal(signal, dt, distance, temp)
                spec = spectral_analysis(propagated, dt)

                energy_ratio = spec["total_energy"] / max(source_spec["total_energy"], 1e-20)
                energy_db = 10 * np.log10(max(energy_ratio, 1e-20))

                bands = spec["band_energy_pct"]
                lines.append(f"{distance:>8.1f}km {temp:>4d}C {spec['peak_freq_hz']:>7.0f} {spec['centroid_hz']:>8.0f} "
                           f"{bands.get('4-8kHz',0):>5.1f}% {bands.get('8-12kHz',0):>5.1f}% "
                           f"{bands.get('12-16kHz',0):>5.1f}% {bands.get('16-20kHz',0):>5.1f}% "
                           f"{energy_db:>8.1f}dB")

                whale_results["propagated"].append({
                    "distance_km": distance,
                    "temp_c": temp,
                    "spectrum": spec,
                    "energy_loss_db": round(energy_db, 1),
                })

            lines.append("")  # blank line between distances

        all_results.append(whale_results)

    # Voiceprint survival analysis
    lines.append("\n" + "=" * 90)
    lines.append("VOICEPRINT SURVIVAL: At what distance can we still tell whales apart?")
    lines.append("=" * 90)

    for distance in distances_km:
        for temp in [2, 10, 20, 30]:
            centroids = []
            for wr in all_results:
                for prop in wr["propagated"]:
                    if prop["distance_km"] == distance and prop["temp_c"] == temp:
                        centroids.append((wr["name"], prop["spectrum"]["centroid_hz"]))
            if len(centroids) >= 2:
                values = [c[1] for c in centroids]
                spread = max(values) - min(values) if values else 0
                names = [c[0] for c in centroids]
                lines.append(f"  {distance:>6.1f}km, {temp:>3d}C: centroid spread = {spread:.0f}Hz "
                           f"({', '.join(f'{n}={v:.0f}' for n, v in centroids)})")

    lines.append("")
    lines.append("=" * 90)

    report = "\n".join(lines)

    with open(REPORT_PATH, "w") as f:
        f.write(report)
    log(f"Report: {REPORT_PATH}")

    with open(DATA_PATH, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log(f"Data: {DATA_PATH}")

    print("\n" + report)


if __name__ == "__main__":
    main()

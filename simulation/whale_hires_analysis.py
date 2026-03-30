#!/usr/bin/env python3
"""
High-Resolution Frequency-Time Analysis of Simulated Whale Clicks

Takes the simulation results and produces:
1. Spectrogram (frequency as function of time) - high resolution
2. Pulse envelope (ramp-up and falloff of each macro-pulse)
3. Cavity interference analysis (standing wave patterns)
4. Pressure field snapshots at key moments during propagation
5. Comparison across whale geometries
"""

import json
import os
import sys
import time
import numpy as np
from pathlib import Path

OUTPUT_DIR = "/mnt/archive/datasets/whale_communication/simulation"
REPORT_PATH = os.path.join(OUTPUT_DIR, "hires_analysis.txt")
DATA_PATH = os.path.join(OUTPUT_DIR, "hires_analysis.json")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def compute_spectrogram(signal, dt, n_fft=256, hop=32):
    """High-resolution short-time FFT spectrogram.
    Returns (times_ms, freqs_hz, magnitude_db)"""
    n_samples = len(signal)
    window = np.hanning(n_fft)

    n_frames = (n_samples - n_fft) // hop + 1
    if n_frames <= 0:
        return [], [], []

    freqs = np.fft.rfftfreq(n_fft, dt)
    times = np.array([(i * hop + n_fft // 2) * dt * 1000 for i in range(n_frames)])  # ms

    spec = np.zeros((n_frames, len(freqs)))
    for i in range(n_frames):
        start = i * hop
        frame = signal[start:start + n_fft] * window
        fft_mag = np.abs(np.fft.rfft(frame))
        spec[i] = 20 * np.log10(fft_mag + 1e-20)  # dB

    return times, freqs, spec


def extract_envelope(signal, dt, smooth_ms=0.2):
    """Extract amplitude envelope with controllable smoothing.
    Returns (times_ms, envelope)"""
    abs_signal = np.abs(signal)
    # Smooth with running average
    smooth_samples = max(int(smooth_ms / 1000.0 / dt), 1)
    kernel = np.ones(smooth_samples) / smooth_samples
    envelope = np.convolve(abs_signal, kernel, mode='same')
    times = np.arange(len(envelope)) * dt * 1000  # ms
    return times, envelope


def find_macro_pulses(envelope, times_ms, min_gap_ms=0.5):
    """Find macro-pulses (P0, P1, P2...) by detecting envelope peaks
    separated by at least min_gap_ms."""
    threshold = np.max(envelope) * 0.05

    # Find regions above threshold
    above = envelope > threshold
    pulses = []
    in_pulse = False
    pulse_start = 0
    peak_val = 0
    peak_time = 0

    for i in range(len(above)):
        if above[i] and not in_pulse:
            in_pulse = True
            pulse_start = i
            peak_val = envelope[i]
            peak_time = times_ms[i]
        elif above[i] and in_pulse:
            if envelope[i] > peak_val:
                peak_val = envelope[i]
                peak_time = times_ms[i]
        elif not above[i] and in_pulse:
            pulse_end = i
            duration = times_ms[pulse_end] - times_ms[pulse_start]
            if duration > 0.01:  # ignore tiny glitches
                pulses.append({
                    "start_ms": float(times_ms[pulse_start]),
                    "end_ms": float(times_ms[pulse_end]),
                    "peak_ms": float(peak_time),
                    "peak_amplitude": float(peak_val),
                    "duration_ms": float(duration),
                })
            in_pulse = False
            peak_val = 0

    # Merge pulses that are very close (within click oscillation period)
    merged = []
    for p in pulses:
        if merged and (p["start_ms"] - merged[-1]["end_ms"]) < min_gap_ms:
            # Merge into previous
            merged[-1]["end_ms"] = p["end_ms"]
            merged[-1]["duration_ms"] = merged[-1]["end_ms"] - merged[-1]["start_ms"]
            if p["peak_amplitude"] > merged[-1]["peak_amplitude"]:
                merged[-1]["peak_amplitude"] = p["peak_amplitude"]
                merged[-1]["peak_ms"] = p["peak_ms"]
        else:
            merged.append(dict(p))

    return merged


def analyze_pulse_shape(signal, dt, pulse_info):
    """Analyze ramp-up and falloff of a single macro-pulse."""
    start_idx = int(pulse_info["start_ms"] / 1000.0 / dt)
    end_idx = int(pulse_info["end_ms"] / 1000.0 / dt)
    peak_idx = int(pulse_info["peak_ms"] / 1000.0 / dt)

    start_idx = max(0, start_idx)
    end_idx = min(len(signal), end_idx)
    peak_idx = max(start_idx, min(end_idx, peak_idx))

    if end_idx <= start_idx:
        return {}

    pulse_signal = signal[start_idx:end_idx]
    abs_pulse = np.abs(pulse_signal)

    # Ramp-up: start to peak
    ramp_samples = peak_idx - start_idx
    ramp_time_ms = ramp_samples * dt * 1000

    # Falloff: peak to end
    fall_samples = end_idx - peak_idx
    fall_time_ms = fall_samples * dt * 1000

    # Rise time (10% to 90% of peak)
    peak_val = np.max(abs_pulse)
    if peak_val > 0:
        thresh_10 = peak_val * 0.1
        thresh_90 = peak_val * 0.9
        rise_start = np.argmax(abs_pulse >= thresh_10)
        rise_end = np.argmax(abs_pulse >= thresh_90)
        rise_time_ms = (rise_end - rise_start) * dt * 1000

        # Fall time (90% to 10% after peak)
        peak_pos = np.argmax(abs_pulse)
        post_peak = abs_pulse[peak_pos:]
        fall_90_idx = 0
        fall_10_idx = len(post_peak) - 1
        for j in range(len(post_peak)):
            if post_peak[j] < thresh_90 and fall_90_idx == 0:
                fall_90_idx = j
            if post_peak[j] < thresh_10:
                fall_10_idx = j
                break
        fall_time_10_90_ms = (fall_10_idx - fall_90_idx) * dt * 1000
    else:
        rise_time_ms = 0
        fall_time_10_90_ms = 0

    # Spectral content of this pulse
    if len(pulse_signal) > 32:
        fft_pulse = np.abs(np.fft.rfft(pulse_signal))
        freqs = np.fft.rfftfreq(len(pulse_signal), dt)
        mask = freqs > 500
        if np.any(mask) and np.any(fft_pulse[mask] > 0):
            peak_freq = float(freqs[mask][np.argmax(fft_pulse[mask])])
            total_energy = np.sum(fft_pulse**2)
            centroid = float(np.sum(freqs * fft_pulse**2) / max(total_energy, 1e-20))
        else:
            peak_freq = 0
            centroid = 0
    else:
        peak_freq = 0
        centroid = 0

    return {
        "ramp_up_ms": round(ramp_time_ms, 4),
        "falloff_ms": round(fall_time_ms, 4),
        "rise_time_10_90_ms": round(rise_time_ms, 4),
        "fall_time_90_10_ms": round(fall_time_10_90_ms, 4),
        "peak_frequency_hz": round(peak_freq, 0),
        "spectral_centroid_hz": round(centroid, 0),
        "asymmetry": round(ramp_time_ms / max(fall_time_ms, 0.001), 3),  # <1 = fast attack, slow decay
    }


def cavity_interference(signal, dt):
    """Analyze cavity interference patterns.
    Look for standing wave frequencies (constructive interference)
    and nulls (destructive interference).
    """
    n = len(signal)
    fft_full = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(n, dt)

    # Normalize
    max_fft = np.max(fft_full)
    if max_fft > 0:
        fft_norm = fft_full / max_fft
    else:
        return {}

    # Find peaks (resonant frequencies - constructive interference)
    peaks = []
    for i in range(2, len(fft_norm) - 2):
        if freqs[i] < 500:
            continue
        if (fft_norm[i] > fft_norm[i-1] and fft_norm[i] > fft_norm[i+1] and
            fft_norm[i] > fft_norm[i-2] and fft_norm[i] > fft_norm[i+2] and
            fft_norm[i] > 0.1):
            peaks.append({
                "freq_hz": float(freqs[i]),
                "amplitude": float(fft_norm[i]),
            })

    # Sort by amplitude
    peaks.sort(key=lambda x: -x["amplitude"])
    peaks = peaks[:20]  # top 20

    # Find nulls (destructive interference)
    nulls = []
    for i in range(2, len(fft_norm) - 2):
        if freqs[i] < 500:
            continue
        if (fft_norm[i] < fft_norm[i-1] and fft_norm[i] < fft_norm[i+1] and
            fft_norm[i] < 0.05):
            nulls.append({
                "freq_hz": float(freqs[i]),
                "amplitude": float(fft_norm[i]),
            })
    nulls.sort(key=lambda x: x["amplitude"])
    nulls = nulls[:20]

    # Harmonic analysis: check if peaks are harmonically related
    if len(peaks) >= 2:
        fundamental = peaks[0]["freq_hz"]
        harmonics = []
        for p in peaks[1:]:
            ratio = p["freq_hz"] / fundamental
            nearest_int = round(ratio)
            if nearest_int > 0 and abs(ratio - nearest_int) < 0.1:
                harmonics.append({
                    "freq_hz": p["freq_hz"],
                    "harmonic_number": nearest_int,
                    "deviation": round(abs(ratio - nearest_int), 4),
                })
    else:
        fundamental = 0
        harmonics = []

    # Frequency band energy distribution
    bands = {}
    for name, (lo, hi) in [
        ("1-2kHz", (1000, 2000)), ("2-4kHz", (2000, 4000)),
        ("4-8kHz", (4000, 8000)), ("8-12kHz", (8000, 12000)),
        ("12-16kHz", (12000, 16000)), ("16-20kHz", (16000, 20000)),
    ]:
        mask = (freqs >= lo) & (freqs < hi)
        bands[name] = float(np.sum(fft_full[mask]**2))

    total_energy = sum(bands.values())
    if total_energy > 0:
        bands = {k: round(v / total_energy * 100, 1) for k, v in bands.items()}

    return {
        "resonant_peaks": peaks[:10],
        "interference_nulls": nulls[:10],
        "fundamental_hz": round(fundamental, 0),
        "harmonics": harmonics,
        "band_energy_pct": bands,
    }


def analyze_whale(result):
    """Full high-res analysis of one whale's simulation output."""
    name = result["name"]
    log(f"Analyzing {name}...")

    # Reconstruct signal
    signal = np.array(result["forward_signal"], dtype=np.float64)
    dt = result["forward_signal_dt_ms"] / 1000.0  # convert to seconds

    analysis = {"name": name}

    # 1. High-res spectrogram
    log(f"  Computing spectrogram...")
    n_fft = min(256, len(signal) // 4)
    hop = max(n_fft // 8, 1)
    times_ms, freqs_hz, spec_db = compute_spectrogram(signal, dt, n_fft=n_fft, hop=hop)

    # Downsample for storage (keep top 100 time frames, 128 freq bins)
    if len(times_ms) > 0:
        t_idx = np.linspace(0, len(times_ms)-1, min(100, len(times_ms)), dtype=int)
        f_idx = np.linspace(0, len(freqs_hz)-1, min(128, len(freqs_hz)), dtype=int)
        analysis["spectrogram"] = {
            "times_ms": [round(float(times_ms[i]), 4) for i in t_idx],
            "freqs_hz": [round(float(freqs_hz[i]), 0) for i in f_idx],
            "magnitude_db": [[round(float(spec_db[ti, fi]), 1) for fi in f_idx] for ti in t_idx],
        }

    # 2. Envelope and macro-pulse detection
    log(f"  Extracting envelope...")
    env_times, envelope = extract_envelope(signal, dt, smooth_ms=0.3)
    macro_pulses = find_macro_pulses(envelope, env_times, min_gap_ms=0.8)

    analysis["envelope"] = {
        "n_samples": len(envelope),
        "dt_ms": float(dt * 1000),
        "peak_amplitude": float(np.max(envelope)),
        "total_energy": float(np.sum(signal**2) * dt),
    }
    analysis["macro_pulses"] = macro_pulses
    analysis["n_macro_pulses"] = len(macro_pulses)

    if len(macro_pulses) >= 2:
        ipis = [macro_pulses[i+1]["peak_ms"] - macro_pulses[i]["peak_ms"]
                for i in range(len(macro_pulses)-1)]
        analysis["macro_ipi_ms"] = [round(ipi, 4) for ipi in ipis]
        analysis["mean_macro_ipi_ms"] = round(float(np.mean(ipis)), 4)
    else:
        analysis["macro_ipi_ms"] = []
        analysis["mean_macro_ipi_ms"] = 0

    # 3. Per-pulse shape analysis (ramp-up, falloff)
    log(f"  Analyzing pulse shapes...")
    pulse_shapes = []
    for i, pulse in enumerate(macro_pulses[:10]):  # first 10 pulses
        shape = analyze_pulse_shape(signal, dt, pulse)
        shape["pulse_index"] = i
        shape["label"] = f"P{i}"
        pulse_shapes.append(shape)
    analysis["pulse_shapes"] = pulse_shapes

    # Ramp-up/falloff summary
    if pulse_shapes:
        rise_times = [p["rise_time_10_90_ms"] for p in pulse_shapes if p.get("rise_time_10_90_ms", 0) > 0]
        fall_times = [p["fall_time_90_10_ms"] for p in pulse_shapes if p.get("fall_time_90_10_ms", 0) > 0]
        if rise_times:
            analysis["mean_rise_time_ms"] = round(float(np.mean(rise_times)), 4)
        if fall_times:
            analysis["mean_fall_time_ms"] = round(float(np.mean(fall_times)), 4)

    # 4. Cavity interference analysis
    log(f"  Analyzing cavity interference...")
    interference = cavity_interference(signal, dt)
    analysis["interference"] = interference

    # 5. Frequency evolution over time
    # Track how peak frequency changes through the click
    if len(times_ms) > 0 and len(spec_db) > 0:
        freq_evolution = []
        for ti in range(len(times_ms)):
            frame_spec = spec_db[ti]
            # Find peak frequency in this frame
            valid = freqs_hz > 500
            if np.any(valid):
                valid_spec = np.where(valid, frame_spec, -999)
                peak_idx = np.argmax(valid_spec)
                freq_evolution.append({
                    "time_ms": round(float(times_ms[ti]), 4),
                    "peak_freq_hz": round(float(freqs_hz[peak_idx]), 0),
                    "peak_amplitude_db": round(float(frame_spec[peak_idx]), 1),
                })
        # Downsample
        if len(freq_evolution) > 100:
            step = len(freq_evolution) // 100
            freq_evolution = freq_evolution[::step]
        analysis["frequency_evolution"] = freq_evolution

    return analysis


def generate_report(analyses):
    """Generate text report."""
    lines = []
    lines.append("=" * 80)
    lines.append("HIGH-RESOLUTION FREQUENCY-TIME ANALYSIS OF SIMULATED WHALE CLICKS")
    lines.append("=" * 80)
    lines.append("")

    for a in analyses:
        lines.append(f"\n{'='*60}")
        lines.append(f"WHALE: {a['name']}")
        lines.append(f"{'='*60}")

        # Macro pulses
        lines.append(f"\nMACRO-PULSES DETECTED: {a['n_macro_pulses']}")
        if a.get("mean_macro_ipi_ms"):
            lines.append(f"Mean IPI: {a['mean_macro_ipi_ms']:.3f} ms")
            lines.append(f"IPIs: {a.get('macro_ipi_ms', [])}")
        for p in a.get("macro_pulses", [])[:6]:
            lines.append(f"  Pulse at {p['peak_ms']:.3f}ms: amp={p['peak_amplitude']:.4f}, dur={p['duration_ms']:.3f}ms")

        # Pulse shapes
        lines.append(f"\nPULSE SHAPE ANALYSIS:")
        if a.get("mean_rise_time_ms"):
            lines.append(f"  Mean rise time (10-90%): {a['mean_rise_time_ms']:.4f} ms")
        if a.get("mean_fall_time_ms"):
            lines.append(f"  Mean fall time (90-10%): {a['mean_fall_time_ms']:.4f} ms")
        for ps in a.get("pulse_shapes", [])[:6]:
            lines.append(f"  {ps['label']}: rise={ps.get('rise_time_10_90_ms',0):.3f}ms, "
                        f"fall={ps.get('fall_time_90_10_ms',0):.3f}ms, "
                        f"asym={ps.get('asymmetry',0):.2f}, "
                        f"peak_f={ps.get('peak_frequency_hz',0):.0f}Hz, "
                        f"centroid={ps.get('spectral_centroid_hz',0):.0f}Hz")

        # Cavity interference
        interf = a.get("interference", {})
        lines.append(f"\nCAVITY INTERFERENCE:")
        lines.append(f"  Fundamental resonance: {interf.get('fundamental_hz', 0):.0f} Hz")
        peaks = interf.get("resonant_peaks", [])
        if peaks:
            lines.append(f"  Top resonant frequencies:")
            for p in peaks[:8]:
                lines.append(f"    {p['freq_hz']:.0f} Hz (amplitude: {p['amplitude']:.3f})")
        harmonics = interf.get("harmonics", [])
        if harmonics:
            lines.append(f"  Harmonic structure:")
            for h in harmonics[:5]:
                lines.append(f"    {h['freq_hz']:.0f} Hz = {h['harmonic_number']}x fundamental (deviation: {h['deviation']:.3f})")
        nulls = interf.get("interference_nulls", [])
        if nulls:
            lines.append(f"  Destructive interference nulls:")
            for n in nulls[:5]:
                lines.append(f"    {n['freq_hz']:.0f} Hz (amplitude: {n['amplitude']:.4f})")

        # Band energy
        bands = interf.get("band_energy_pct", {})
        if bands:
            lines.append(f"\n  Frequency band energy:")
            for band, pct in bands.items():
                bar = "#" * int(pct)
                lines.append(f"    {band:>10s}: {pct:5.1f}% {bar}")

        # Frequency evolution
        freq_evo = a.get("frequency_evolution", [])
        if freq_evo:
            lines.append(f"\nFREQUENCY EVOLUTION OVER TIME:")
            # Sample every ~2ms
            step = max(1, len(freq_evo) // 6)
            for fe in freq_evo[::step]:
                lines.append(f"  t={fe['time_ms']:6.2f}ms: peak={fe['peak_freq_hz']:6.0f}Hz, amp={fe['peak_amplitude_db']:6.1f}dB")

    # Comparison
    lines.append(f"\n{'='*80}")
    lines.append("CROSS-WHALE COMPARISON")
    lines.append(f"{'='*80}")

    header = f"{'Whale':>20s} {'Pulses':>7s} {'IPI ms':>8s} {'Rise ms':>8s} {'Fall ms':>8s} {'Fund Hz':>8s}"
    lines.append(header)
    lines.append("-" * 70)
    for a in analyses:
        lines.append(f"{a['name']:>20s} "
                    f"{a['n_macro_pulses']:>6d} "
                    f"{a.get('mean_macro_ipi_ms',0):>7.3f} "
                    f"{a.get('mean_rise_time_ms',0):>7.4f} "
                    f"{a.get('mean_fall_time_ms',0):>7.4f} "
                    f"{a.get('interference',{}).get('fundamental_hz',0):>7.0f}")

    lines.append("")
    lines.append("=" * 80)
    return "\n".join(lines)


def main():
    # Load simulation results
    sim_path = os.path.join(OUTPUT_DIR, "simulation_results.json")
    with open(sim_path) as f:
        results = json.load(f)
    log(f"Loaded {len(results)} whale simulations")

    analyses = []
    for result in results:
        a = analyze_whale(result)
        analyses.append(a)

    # Generate report
    report = generate_report(analyses)
    with open(REPORT_PATH, "w") as f:
        f.write(report)
    log(f"Report: {REPORT_PATH}")

    # Save full analysis data
    with open(DATA_PATH, "w") as f:
        json.dump(analyses, f, indent=2, default=str)
    log(f"Data: {DATA_PATH}")

    print("\n" + report)


if __name__ == "__main__":
    main()

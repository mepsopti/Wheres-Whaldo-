#!/usr/bin/env python3
"""
Deep Signal Analysis of Sperm Whale Codas
Full sweep: amplitude, frequency, intensity, patterns, grouped by coda type.
Cross-references WAV files with DominicaCodas.csv for labeled analysis.
"""

import json
import os
import sys
import time
import wave
import struct
import math
import csv
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np

# Config
WAV_DIR = "/mnt/archive/datasets/whale_communication/DSWP"
CSV_PATH = "/mnt/archive/datasets/whale_communication/sw-combinatoriality/data/DominicaCodas.csv"
DIALOGUE_PATH = "/mnt/archive/datasets/whale_communication/sw-combinatoriality/data/sperm-whale-dialogues.csv"
OUTPUT_DIR = "/mnt/archive/datasets/whale_communication/analysis"
OUTPUT_REPORT = os.path.join(OUTPUT_DIR, "deep_analysis.json")
OUTPUT_TEXT = os.path.join(OUTPUT_DIR, "deep_analysis_report.txt")
LOG_FILE = "/mnt/archive/datasets/logs/whale_deep_analysis.log"
TARGET_SR = 44100


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def read_wav(filepath):
    """Read WAV, return mono float32 at TARGET_SR, peak-normalized."""
    with wave.open(filepath, 'r') as w:
        n_channels = w.getnchannels()
        sampwidth = w.getsampwidth()
        sr = w.getframerate()
        n_frames = w.getnframes()
        raw = w.readframes(n_frames)

    if sampwidth == 2:
        fmt = f"<{n_frames * n_channels}h"
        samples = np.array(struct.unpack(fmt, raw), dtype=np.float32) / 32768.0
    elif sampwidth == 3:
        samples = []
        for i in range(0, len(raw), 3):
            val = int.from_bytes(raw[i:i+3], 'little', signed=True)
            samples.append(val / 8388608.0)
        samples = np.array(samples, dtype=np.float32)
    else:
        fmt = f"<{n_frames * n_channels}b"
        samples = np.array(struct.unpack(fmt, raw), dtype=np.float32) / 128.0

    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)

    if sr != TARGET_SR:
        duration = len(samples) / sr
        new_len = int(duration * TARGET_SR)
        x_old = np.linspace(0, 1, len(samples))
        x_new = np.linspace(0, 1, new_len)
        samples = np.interp(x_new, x_old, samples)

    # Peak normalize to [-1, 1]
    peak = np.max(np.abs(samples))
    if peak > 0.001:
        samples = samples / peak

    return samples


def spectral_analysis(samples):
    """Full spectral analysis of a signal."""
    n = len(samples)
    duration = n / TARGET_SR

    # Overall FFT
    fft_full = np.abs(np.fft.rfft(samples))
    freqs_full = np.fft.rfftfreq(n, 1.0 / TARGET_SR)

    # Energy in frequency bands
    bands = {
        "sub_100hz": (0, 100),
        "100_500hz": (100, 500),
        "500_2khz": (500, 2000),
        "2k_5khz": (2000, 5000),
        "5k_10khz": (5000, 10000),
        "10k_20khz": (10000, 20000),
        "above_20khz": (20000, TARGET_SR // 2),
    }

    total_energy = np.sum(fft_full**2)
    band_energy = {}
    for name, (lo, hi) in bands.items():
        mask = (freqs_full >= lo) & (freqs_full < hi)
        energy = np.sum(fft_full[mask]**2)
        band_energy[name] = float(energy / max(total_energy, 1e-10))

    # Peak frequencies (top 5)
    # Ignore DC
    mask = freqs_full > 50
    fft_masked = fft_full.copy()
    fft_masked[~mask] = 0
    top_indices = np.argsort(fft_masked)[-5:][::-1]
    peak_freqs = [(float(freqs_full[i]), float(fft_full[i])) for i in top_indices]

    # Spectral centroid
    if total_energy > 0:
        centroid = float(np.sum(freqs_full * fft_full**2) / total_energy)
    else:
        centroid = 0.0

    # Spectral bandwidth
    if total_energy > 0:
        bandwidth = float(np.sqrt(np.sum(((freqs_full - centroid)**2) * fft_full**2) / total_energy))
    else:
        bandwidth = 0.0

    # Spectral rolloff (frequency below which 85% energy)
    cumsum = np.cumsum(fft_full**2)
    rolloff_idx = np.searchsorted(cumsum, 0.85 * total_energy)
    rolloff = float(freqs_full[min(rolloff_idx, len(freqs_full) - 1)])

    return {
        "band_energy": band_energy,
        "peak_frequencies": peak_freqs,
        "spectral_centroid_hz": round(centroid, 1),
        "spectral_bandwidth_hz": round(bandwidth, 1),
        "spectral_rolloff_hz": round(rolloff, 1),
        "total_energy": float(total_energy),
    }


def temporal_analysis(samples):
    """Temporal/amplitude analysis."""
    n = len(samples)
    duration = n / TARGET_SR
    abs_samples = np.abs(samples)

    # RMS in windows
    window_ms = 10
    window_samples = int(TARGET_SR * window_ms / 1000)
    n_windows = n // window_samples
    rms_values = []
    for i in range(n_windows):
        chunk = samples[i * window_samples:(i + 1) * window_samples]
        rms_values.append(float(np.sqrt(np.mean(chunk**2))))

    rms_array = np.array(rms_values)

    # Dynamic range
    rms_nonzero = rms_array[rms_array > 0.001]
    if len(rms_nonzero) > 0:
        dynamic_range_db = float(20 * np.log10(np.max(rms_nonzero) / np.min(rms_nonzero)))
    else:
        dynamic_range_db = 0.0

    # Silence ratio (frames below -40dB of peak)
    silence_thresh = 0.01  # ~-40dB
    silence_ratio = float(np.sum(rms_array < silence_thresh) / max(len(rms_array), 1))

    # Attack time (time to reach 90% of peak from start of signal)
    peak_idx = np.argmax(abs_samples)
    peak_val = abs_samples[peak_idx]
    thresh_90 = peak_val * 0.9
    attack_indices = np.where(abs_samples[:peak_idx + 1] >= thresh_90)[0]
    if len(attack_indices) > 0:
        attack_time = float(attack_indices[0] / TARGET_SR)
    else:
        attack_time = float(peak_idx / TARGET_SR)

    # Intensity over time (10 equal segments)
    n_segments = 10
    seg_len = n // n_segments
    segment_rms = []
    for i in range(n_segments):
        chunk = samples[i * seg_len:(i + 1) * seg_len]
        segment_rms.append(round(float(np.sqrt(np.mean(chunk**2))), 6))

    # Zero crossing rate
    zero_crossings = np.sum(np.diff(np.sign(samples)) != 0)
    zcr = float(zero_crossings / n)

    # Autocorrelation (for periodicity detection)
    # Look for repeating patterns in the envelope
    if len(rms_values) > 20:
        rms_centered = rms_array - np.mean(rms_array)
        autocorr = np.correlate(rms_centered, rms_centered, mode='full')
        autocorr = autocorr[len(autocorr) // 2:]
        if autocorr[0] > 0:
            autocorr = autocorr / autocorr[0]
        # Find first peak after lag 0 (indicates periodicity)
        peaks = []
        for i in range(2, len(autocorr) - 1):
            if autocorr[i] > autocorr[i - 1] and autocorr[i] > autocorr[i + 1] and autocorr[i] > 0.3:
                peaks.append((i * window_ms / 1000, float(autocorr[i])))
                if len(peaks) >= 3:
                    break
        periodicity = peaks
    else:
        periodicity = []

    return {
        "duration_s": round(duration, 4),
        "rms_overall": round(float(np.sqrt(np.mean(samples**2))), 6),
        "peak_amplitude": round(float(np.max(abs_samples)), 6),
        "dynamic_range_db": round(dynamic_range_db, 1),
        "silence_ratio": round(silence_ratio, 3),
        "attack_time_s": round(attack_time, 4),
        "zero_crossing_rate": round(zcr, 6),
        "intensity_over_time": segment_rms,
        "periodicity_peaks": periodicity,
    }


def click_analysis(samples):
    """Detect clicks using adaptive threshold on envelope."""
    abs_env = np.abs(samples)

    # Smooth envelope (5ms window)
    kernel_size = int(TARGET_SR * 0.005)
    if kernel_size < 1:
        kernel_size = 1
    kernel = np.ones(kernel_size) / kernel_size
    smooth = np.convolve(abs_env, kernel, mode='same')

    # Adaptive threshold: median + 3*MAD
    median_val = np.median(smooth)
    mad = np.median(np.abs(smooth - median_val))
    threshold = median_val + 6 * mad  # aggressive to only catch real clicks

    if threshold < 0.05:
        threshold = 0.05

    # Find peaks above threshold with minimum spacing of 20ms
    min_spacing = int(TARGET_SR * 0.02)
    above = smooth > threshold
    clicks = []
    last_click = -min_spacing

    for i in range(1, len(above) - 1):
        if above[i] and smooth[i] > smooth[i-1] and smooth[i] >= smooth[i+1]:
            if i - last_click >= min_spacing:
                clicks.append(i)
                last_click = i

    click_times = [c / TARGET_SR for c in clicks]
    icis = [click_times[i+1] - click_times[i] for i in range(len(click_times) - 1)]

    # Click intensities (peak amplitude at each click)
    click_intensities = [float(smooth[c]) for c in clicks]

    # Classify ICI pattern
    if len(icis) >= 2:
        ici_cv = np.std(icis) / max(np.mean(icis), 0.001)  # coefficient of variation
        if ici_cv < 0.15:
            rhythm_type = "regular"
        elif ici_cv < 0.4:
            rhythm_type = "semi_regular"
        else:
            rhythm_type = "irregular"
    elif len(icis) == 1:
        rhythm_type = "single_interval"
    else:
        rhythm_type = "no_clicks"

    return {
        "n_clicks": len(clicks),
        "click_times_s": [round(t, 6) for t in click_times],
        "icis_s": [round(i, 6) for i in icis],
        "ici_mean_s": round(float(np.mean(icis)), 6) if icis else 0,
        "ici_std_s": round(float(np.std(icis)), 6) if icis else 0,
        "ici_cv": round(float(np.std(icis) / max(np.mean(icis), 0.001)), 4) if icis else 0,
        "click_intensities": [round(x, 4) for x in click_intensities],
        "rhythm_type": rhythm_type,
        "threshold_used": round(float(threshold), 4),
    }


def analyze_single(filepath):
    """Full analysis of one WAV file."""
    coda_id = os.path.splitext(os.path.basename(filepath))[0]
    try:
        samples = read_wav(filepath)
        spectral = spectral_analysis(samples)
        temporal = temporal_analysis(samples)
        clicks = click_analysis(samples)
        return {
            "id": coda_id,
            **temporal,
            **spectral,
            **clicks,
        }
    except Exception as e:
        return {"id": coda_id, "error": str(e)}


def load_csv_labels():
    """Load DominicaCodas.csv for coda type labels."""
    labels = {}
    try:
        with open(CSV_PATH, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                coda_num = row.get('codaNUM2018', '').strip()
                if coda_num:
                    labels[coda_num] = {
                        "coda_type": row.get('CodaType', ''),
                        "clan": row.get('Clan', ''),
                        "unit": row.get('Unit', ''),
                        "n_clicks_labeled": int(row.get('nClicks', 0)),
                        "duration_labeled": float(row.get('Duration', 0)),
                        "date": row.get('Date', ''),
                    }
    except Exception as e:
        log(f"Warning: Could not load CSV labels: {e}")
    return labels


def generate_report(records, labels):
    """Generate comprehensive text report."""
    valid = [r for r in records if "error" not in r]
    errors = [r for r in records if "error" in r]

    # Match records to labels
    labeled = []
    unlabeled = []
    for r in valid:
        if r["id"] in labels:
            r["label"] = labels[r["id"]]
            labeled.append(r)
        else:
            unlabeled.append(r)

    lines = []
    lines.append("=" * 80)
    lines.append("SPERM WHALE CODA - DEEP SIGNAL ANALYSIS REPORT")
    lines.append("=" * 80)
    lines.append("")

    # Overview
    lines.append("1. DATASET OVERVIEW")
    lines.append("-" * 40)
    lines.append(f"  Total WAV files: {len(records)}")
    lines.append(f"  Successfully analyzed: {len(valid)}")
    lines.append(f"  Errors: {len(errors)}")
    lines.append(f"  Matched to CSV labels: {len(labeled)}")
    lines.append(f"  Unlabeled: {len(unlabeled)}")
    lines.append("")

    # Duration stats
    durations = [r["duration_s"] for r in valid]
    lines.append("2. DURATION")
    lines.append("-" * 40)
    lines.append(f"  Min: {min(durations):.3f}s")
    lines.append(f"  Max: {max(durations):.3f}s")
    lines.append(f"  Mean: {np.mean(durations):.3f}s")
    lines.append(f"  Median: {np.median(durations):.3f}s")
    lines.append(f"  Std: {np.std(durations):.3f}s")
    # Duration histogram
    hist, bins = np.histogram(durations, bins=10)
    lines.append("  Distribution:")
    for i in range(len(hist)):
        bar = "#" * (hist[i] * 40 // max(hist))
        lines.append(f"    {bins[i]:.1f}-{bins[i+1]:.1f}s: {hist[i]:4d} {bar}")
    lines.append("")

    # Amplitude/Intensity
    rms_vals = [r["rms_overall"] for r in valid]
    dynamic_ranges = [r["dynamic_range_db"] for r in valid if r["dynamic_range_db"] > 0]
    lines.append("3. AMPLITUDE & INTENSITY")
    lines.append("-" * 40)
    lines.append(f"  RMS range: {min(rms_vals):.6f} - {max(rms_vals):.6f}")
    lines.append(f"  RMS mean: {np.mean(rms_vals):.6f}")
    if dynamic_ranges:
        lines.append(f"  Dynamic range: {min(dynamic_ranges):.1f}dB - {max(dynamic_ranges):.1f}dB (mean {np.mean(dynamic_ranges):.1f}dB)")
    silence_ratios = [r["silence_ratio"] for r in valid]
    lines.append(f"  Silence ratio: {min(silence_ratios):.2f} - {max(silence_ratios):.2f} (mean {np.mean(silence_ratios):.2f})")
    lines.append("")

    # Spectral
    centroids = [r["spectral_centroid_hz"] for r in valid]
    bandwidths = [r["spectral_bandwidth_hz"] for r in valid]
    rolloffs = [r["spectral_rolloff_hz"] for r in valid]
    lines.append("4. SPECTRAL CHARACTERISTICS")
    lines.append("-" * 40)
    lines.append(f"  Spectral centroid: {min(centroids):.0f}Hz - {max(centroids):.0f}Hz (mean {np.mean(centroids):.0f}Hz)")
    lines.append(f"  Spectral bandwidth: {min(bandwidths):.0f}Hz - {max(bandwidths):.0f}Hz (mean {np.mean(bandwidths):.0f}Hz)")
    lines.append(f"  Spectral rolloff (85%): {min(rolloffs):.0f}Hz - {max(rolloffs):.0f}Hz (mean {np.mean(rolloffs):.0f}Hz)")
    lines.append("")

    # Frequency band energy distribution
    lines.append("  Frequency band energy (mean across all codas):")
    band_names = ["sub_100hz", "100_500hz", "500_2khz", "2k_5khz", "5k_10khz", "10k_20khz", "above_20khz"]
    for band in band_names:
        vals = [r["band_energy"][band] for r in valid]
        mean_pct = np.mean(vals) * 100
        bar = "#" * int(mean_pct * 2)
        lines.append(f"    {band:>15s}: {mean_pct:5.1f}% {bar}")
    lines.append("")

    # Click detection
    n_clicks_list = [r["n_clicks"] for r in valid]
    lines.append("5. CLICK DETECTION")
    lines.append("-" * 40)
    lines.append(f"  Codas with clicks detected: {sum(1 for n in n_clicks_list if n > 0)}/{len(valid)}")
    lines.append(f"  Clicks per coda: {min(n_clicks_list)} - {max(n_clicks_list)} (mean {np.mean(n_clicks_list):.1f})")
    click_dist = Counter(n_clicks_list)
    lines.append("  Click count distribution:")
    for n in sorted(click_dist.keys())[:20]:
        bar = "#" * (click_dist[n] * 40 // max(click_dist.values()))
        lines.append(f"    {n:3d} clicks: {click_dist[n]:4d} {bar}")
    if max(click_dist.keys()) > 20:
        remaining = sum(v for k, v in click_dist.items() if k > 20)
        lines.append(f"    >20 clicks: {remaining}")
    lines.append("")

    # ICI analysis
    all_icis = []
    for r in valid:
        all_icis.extend(r["icis_s"])
    if all_icis:
        lines.append("6. INTER-CLICK INTERVALS (ICI)")
        lines.append("-" * 40)
        lines.append(f"  Total ICIs measured: {len(all_icis)}")
        lines.append(f"  ICI range: {min(all_icis)*1000:.1f}ms - {max(all_icis)*1000:.1f}ms")
        lines.append(f"  ICI mean: {np.mean(all_icis)*1000:.1f}ms")
        lines.append(f"  ICI median: {np.median(all_icis)*1000:.1f}ms")
        # ICI histogram
        hist, bins = np.histogram(all_icis, bins=15, range=(0, min(max(all_icis), 1.0)))
        lines.append("  ICI distribution:")
        for i in range(len(hist)):
            bar = "#" * (hist[i] * 40 // max(max(hist), 1))
            lines.append(f"    {bins[i]*1000:6.0f}-{bins[i+1]*1000:6.0f}ms: {hist[i]:5d} {bar}")

        # Rhythm types
        rhythm_dist = Counter(r["rhythm_type"] for r in valid)
        lines.append("  Rhythm classification:")
        for rt, count in rhythm_dist.most_common():
            lines.append(f"    {rt}: {count}")
        lines.append("")

    # Zero crossing rate
    zcrs = [r["zero_crossing_rate"] for r in valid]
    lines.append("7. ZERO CROSSING RATE")
    lines.append("-" * 40)
    lines.append(f"  Range: {min(zcrs):.6f} - {max(zcrs):.6f}")
    lines.append(f"  Mean: {np.mean(zcrs):.6f}")
    lines.append(f"  (Higher = more high-frequency content)")
    lines.append("")

    # Intensity over time (averaged across all codas)
    lines.append("8. AVERAGE INTENSITY PROFILE OVER TIME")
    lines.append("-" * 40)
    avg_profile = np.zeros(10)
    count = 0
    for r in valid:
        if len(r["intensity_over_time"]) == 10:
            avg_profile += np.array(r["intensity_over_time"])
            count += 1
    if count > 0:
        avg_profile /= count
        max_int = max(avg_profile)
        for i, val in enumerate(avg_profile):
            pct = int(i * 10)
            bar = "#" * int(val / max(max_int, 0.001) * 40)
            lines.append(f"    {pct:3d}-{pct+10:3d}%: {val:.4f} {bar}")
    lines.append("")

    # BY CODA TYPE (if labels available)
    if labeled:
        lines.append("=" * 80)
        lines.append("9. ANALYSIS BY CODA TYPE")
        lines.append("=" * 80)

        by_type = defaultdict(list)
        for r in labeled:
            by_type[r["label"]["coda_type"]].append(r)

        for ctype in sorted(by_type.keys(), key=lambda x: -len(by_type[x])):
            group = by_type[ctype]
            lines.append("")
            lines.append(f"  CODA TYPE: {ctype} ({len(group)} codas)")
            lines.append(f"  " + "-" * 50)

            g_dur = [r["duration_s"] for r in group]
            g_clicks = [r["n_clicks"] for r in group]
            g_rms = [r["rms_overall"] for r in group]
            g_centroid = [r["spectral_centroid_hz"] for r in group]
            g_ici_mean = [r["ici_mean_s"] for r in group if r["ici_mean_s"] > 0]
            g_rhythm = Counter(r["rhythm_type"] for r in group)
            g_silence = [r["silence_ratio"] for r in group]

            lines.append(f"    Duration: {np.mean(g_dur):.3f}s +/- {np.std(g_dur):.3f}s")
            lines.append(f"    Clicks detected: {np.mean(g_clicks):.1f} +/- {np.std(g_clicks):.1f}")
            lines.append(f"    Labeled clicks: {group[0]['label']['n_clicks_labeled']}")
            lines.append(f"    RMS: {np.mean(g_rms):.5f} +/- {np.std(g_rms):.5f}")
            lines.append(f"    Spectral centroid: {np.mean(g_centroid):.0f}Hz +/- {np.std(g_centroid):.0f}Hz")
            if g_ici_mean:
                lines.append(f"    Mean ICI: {np.mean(g_ici_mean)*1000:.1f}ms +/- {np.std(g_ici_mean)*1000:.1f}ms")
            lines.append(f"    Silence ratio: {np.mean(g_silence):.2f}")
            lines.append(f"    Rhythm: {dict(g_rhythm)}")

            # Band energy profile for this type
            lines.append(f"    Frequency profile:")
            for band in band_names:
                vals = [r["band_energy"][band] for r in group]
                lines.append(f"      {band:>15s}: {np.mean(vals)*100:.1f}%")

        # Clan comparison
        lines.append("")
        lines.append("=" * 80)
        lines.append("10. CLAN COMPARISON")
        lines.append("=" * 80)
        by_clan = defaultdict(list)
        for r in labeled:
            by_clan[r["label"]["clan"]].append(r)

        for clan in sorted(by_clan.keys()):
            group = by_clan[clan]
            lines.append(f"\n  CLAN: {clan} ({len(group)} codas)")
            lines.append(f"    Coda types used: {sorted(set(r['label']['coda_type'] for r in group))}")
            g_centroid = [r["spectral_centroid_hz"] for r in group]
            g_rms = [r["rms_overall"] for r in group]
            g_dur = [r["duration_s"] for r in group]
            lines.append(f"    Spectral centroid: {np.mean(g_centroid):.0f}Hz +/- {np.std(g_centroid):.0f}Hz")
            lines.append(f"    RMS: {np.mean(g_rms):.5f}")
            lines.append(f"    Duration: {np.mean(g_dur):.3f}s")

        # Individual whale comparison
        lines.append("")
        lines.append("=" * 80)
        lines.append("11. INDIVIDUAL WHALE COMPARISON")
        lines.append("=" * 80)
        by_whale = defaultdict(list)
        for r in labeled:
            by_whale[r["label"]["unit"]].append(r)

        for whale in sorted(by_whale.keys()):
            if whale in ('', 'ZZZ'):
                continue
            group = by_whale[whale]
            lines.append(f"\n  WHALE: {whale} ({len(group)} codas)")
            g_centroid = [r["spectral_centroid_hz"] for r in group]
            g_rms = [r["rms_overall"] for r in group]
            types_used = Counter(r["label"]["coda_type"] for r in group)
            lines.append(f"    Spectral centroid: {np.mean(g_centroid):.0f}Hz +/- {np.std(g_centroid):.0f}Hz")
            lines.append(f"    RMS: {np.mean(g_rms):.5f}")
            lines.append(f"    Coda types: {dict(types_used.most_common(5))}")

    # Commonalities
    lines.append("")
    lines.append("=" * 80)
    lines.append("12. KEY COMMONALITIES & PATTERNS")
    lines.append("=" * 80)
    lines.append("")
    lines.append("  (Auto-detected patterns across the dataset)")
    lines.append("")

    # Most consistent coda types (lowest CV in duration)
    if labeled:
        type_consistency = []
        for ctype, group in by_type.items():
            if len(group) >= 10:
                g_dur = [r["duration_s"] for r in group]
                cv = np.std(g_dur) / max(np.mean(g_dur), 0.001)
                type_consistency.append((ctype, cv, len(group), np.mean(g_dur)))

        if type_consistency:
            type_consistency.sort(key=lambda x: x[1])
            lines.append("  Most temporally consistent coda types (lowest CV in duration):")
            for ctype, cv, n, mean_dur in type_consistency[:10]:
                lines.append(f"    {ctype}: CV={cv:.3f} (n={n}, mean={mean_dur:.3f}s)")
            lines.append("")

    # Frequency clustering
    centroid_groups = {"low (<2kHz)": 0, "mid (2-5kHz)": 0, "high (>5kHz)": 0}
    for r in valid:
        c = r["spectral_centroid_hz"]
        if c < 2000:
            centroid_groups["low (<2kHz)"] += 1
        elif c < 5000:
            centroid_groups["mid (2-5kHz)"] += 1
        else:
            centroid_groups["high (>5kHz)"] += 1
    lines.append("  Frequency clustering by spectral centroid:")
    for group, count in centroid_groups.items():
        lines.append(f"    {group}: {count} codas ({count*100/len(valid):.1f}%)")
    lines.append("")

    # Periodicity
    periodic = sum(1 for r in valid if r.get("periodicity_peaks"))
    lines.append(f"  Codas with detected periodicity: {periodic}/{len(valid)} ({periodic*100/len(valid):.1f}%)")

    lines.append("")
    lines.append("=" * 80)
    lines.append("END OF REPORT")
    lines.append("=" * 80)

    return "\n".join(lines)


def main():
    log("=" * 70)
    log("DEEP SIGNAL ANALYSIS - SPERM WHALE CODAS")
    log("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load labels
    labels = load_csv_labels()
    log(f"Loaded {len(labels)} coda labels from CSV")

    # Find WAV files
    wav_files = sorted(Path(WAV_DIR).glob("*.wav"), key=lambda p: int(p.stem) if p.stem.isdigit() else 0)
    log(f"Found {len(wav_files)} WAV files")

    # Analyze all
    records = []
    start = time.time()

    for i, wav_path in enumerate(wav_files):
        record = analyze_single(str(wav_path))
        records.append(record)

        if (i + 1) % 100 == 0:
            elapsed = time.time() - start
            log(f"  Processed {i+1}/{len(wav_files)} ({elapsed:.0f}s)")

    elapsed = time.time() - start
    log(f"Analysis complete: {len(records)} codas in {elapsed:.0f}s")

    # Save raw analysis
    log("Saving raw analysis JSONL...")
    raw_path = os.path.join(OUTPUT_DIR, "deep_analysis_raw.jsonl")
    with open(raw_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    # Generate report
    log("Generating report...")
    report = generate_report(records, labels)

    with open(OUTPUT_TEXT, "w") as f:
        f.write(report)
    log(f"Report saved to {OUTPUT_TEXT}")

    # Print report
    print("\n" + report)


if __name__ == "__main__":
    main()

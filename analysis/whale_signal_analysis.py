#!/usr/bin/env python3
"""
Whale Coda Signal Analysis
Processes all 1,500 DSWP WAV files and extracts:
  - Amplitude envelope over time
  - FFT (spectrogram) over time
  - Click detection and inter-click intervals

Output: Single JSONL file with one record per WAV, plus summary HTML visualization.
"""

import json
import os
import sys
import time
import wave
import struct
import math
from pathlib import Path

import numpy as np

# Config
WAV_DIR = "/mnt/archive/datasets/whale_communication/DSWP"
OUTPUT_DIR = "/mnt/archive/datasets/whale_communication/analysis"
OUTPUT_JSONL = os.path.join(OUTPUT_DIR, "coda_signals.jsonl")
OUTPUT_HTML = os.path.join(OUTPUT_DIR, "coda_overview.html")
LOG_FILE = "/mnt/archive/datasets/logs/whale_signal_analysis.log"

# Analysis params
TARGET_SR = 44100       # Resample everything to 44.1kHz
FFT_WINDOW = 1024       # FFT window size (samples)
FFT_HOP = 256           # FFT hop size (samples)
ENVELOPE_WINDOW = 441   # ~10ms at 44.1kHz for amplitude envelope
CLICK_THRESHOLD = 0.3   # Relative amplitude threshold for click detection


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def read_wav(filepath):
    """Read WAV file, return mono float32 array at TARGET_SR."""
    with wave.open(filepath, 'r') as w:
        n_channels = w.getnchannels()
        sampwidth = w.getsampwidth()
        sr = w.getframerate()
        n_frames = w.getnframes()
        raw = w.readframes(n_frames)

    # Decode to float32
    if sampwidth == 2:
        fmt = f"<{n_frames * n_channels}h"
        samples = np.array(struct.unpack(fmt, raw), dtype=np.float32) / 32768.0
    elif sampwidth == 3:
        # 24-bit audio
        samples = []
        for i in range(0, len(raw), 3):
            val = int.from_bytes(raw[i:i+3], 'little', signed=True)
            samples.append(val / 8388608.0)
        samples = np.array(samples, dtype=np.float32)
    else:
        fmt = f"<{n_frames * n_channels}b"
        samples = np.array(struct.unpack(fmt, raw), dtype=np.float32) / 128.0

    # Mix to mono
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)

    # Resample if needed (simple linear interpolation)
    if sr != TARGET_SR:
        duration = len(samples) / sr
        new_len = int(duration * TARGET_SR)
        x_old = np.linspace(0, 1, len(samples))
        x_new = np.linspace(0, 1, new_len)
        samples = np.interp(x_new, x_old, samples)

    return samples


def compute_envelope(samples, window=ENVELOPE_WINDOW):
    """Compute amplitude envelope using RMS in sliding window."""
    # Pad
    pad = window // 2
    padded = np.pad(np.abs(samples), (pad, pad), mode='constant')
    # Cumulative sum for fast windowed mean
    cumsum = np.cumsum(padded**2)
    rms = np.sqrt((cumsum[window:] - cumsum[:-window]) / window)
    # Downsample to ~100 points for storage
    n_points = min(200, len(rms))
    indices = np.linspace(0, len(rms) - 1, n_points, dtype=int)
    return rms[indices].tolist()


def compute_spectrogram(samples, n_fft=FFT_WINDOW, hop=FFT_HOP):
    """Compute spectrogram (magnitude of STFT). Returns time x freq matrix."""
    # Apply Hann window
    window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(n_fft) / n_fft))

    n_frames_spec = (len(samples) - n_fft) // hop + 1
    if n_frames_spec <= 0:
        return [], []

    spec = []
    for i in range(n_frames_spec):
        frame = samples[i * hop:i * hop + n_fft] * window
        fft_result = np.fft.rfft(frame)
        magnitude = np.abs(fft_result)
        # Convert to dB
        magnitude_db = 20 * np.log10(magnitude + 1e-10)
        spec.append(magnitude_db)

    spec = np.array(spec)

    # Frequency bins
    freqs = np.fft.rfftfreq(n_fft, 1.0 / TARGET_SR)

    # Downsample time axis to ~50 frames for storage
    n_time = min(50, spec.shape[0])
    time_indices = np.linspace(0, spec.shape[0] - 1, n_time, dtype=int)
    spec_down = spec[time_indices]

    # Downsample freq axis to ~64 bins
    n_freq = min(64, spec_down.shape[1])
    freq_indices = np.linspace(0, spec_down.shape[1] - 1, n_freq, dtype=int)
    spec_down = spec_down[:, freq_indices]
    freqs_down = freqs[freq_indices]

    return spec_down.tolist(), freqs_down.tolist()


def detect_clicks(samples, threshold=CLICK_THRESHOLD):
    """Detect clicks in the signal. Returns list of click times in seconds."""
    # Compute envelope
    envelope = np.abs(samples)
    # Smooth
    kernel = np.ones(221) / 221  # 5ms window
    smooth = np.convolve(envelope, kernel, mode='same')

    # Peak detection
    max_amp = np.max(smooth)
    if max_amp < 0.01:
        return []

    thresh = max_amp * threshold
    above = smooth > thresh

    # Find rising edges
    clicks = []
    in_click = False
    for i in range(1, len(above)):
        if above[i] and not in_click:
            clicks.append(i / TARGET_SR)
            in_click = True
        elif not above[i]:
            in_click = False

    return clicks


def compute_icis(click_times):
    """Compute inter-click intervals from click times."""
    if len(click_times) < 2:
        return []
    return [round(click_times[i+1] - click_times[i], 6) for i in range(len(click_times) - 1)]


def analyze_wav(filepath):
    """Full analysis of a single WAV file."""
    filename = os.path.basename(filepath)
    coda_id = os.path.splitext(filename)[0]

    try:
        samples = read_wav(filepath)
    except Exception as e:
        return {"id": coda_id, "error": str(e)}

    duration = len(samples) / TARGET_SR

    # Amplitude envelope
    envelope = compute_envelope(samples)

    # Spectrogram
    spectrogram, freqs = compute_spectrogram(samples)

    # Click detection
    click_times = detect_clicks(samples)
    icis = compute_icis(click_times)

    # Peak frequency (dominant frequency in FFT of full signal)
    full_fft = np.abs(np.fft.rfft(samples))
    full_freqs = np.fft.rfftfreq(len(samples), 1.0 / TARGET_SR)
    # Ignore DC and very low freq
    mask = full_freqs > 100
    if mask.any():
        peak_idx = np.argmax(full_fft[mask])
        peak_freq = float(full_freqs[mask][peak_idx])
    else:
        peak_freq = 0.0

    # Summary stats
    record = {
        "id": coda_id,
        "duration_s": round(duration, 4),
        "n_clicks": len(click_times),
        "click_times_s": [round(t, 6) for t in click_times],
        "icis_s": icis,
        "peak_freq_hz": round(peak_freq, 1),
        "rms_amplitude": round(float(np.sqrt(np.mean(samples**2))), 6),
        "max_amplitude": round(float(np.max(np.abs(samples))), 6),
        "envelope": [round(x, 4) for x in envelope],
        "spectrogram": [[round(x, 1) for x in row] for row in spectrogram],
        "spectrogram_freqs_hz": [round(f, 1) for f in freqs],
        "envelope_time_s": [round(t, 4) for t in np.linspace(0, duration, len(envelope)).tolist()],
    }

    return record


def generate_html(records):
    """Generate overview HTML with inline visualizations."""
    html = """<!DOCTYPE html>
<html><head><title>Whale Coda Signal Analysis</title>
<style>
body { background: #1C1C1E; color: #E0E0E0; font-family: monospace; padding: 20px; }
h1 { color: #D4A843; }
h2 { color: #D4A843; font-size: 14px; margin-top: 20px; }
.coda { border: 1px solid #333; padding: 10px; margin: 5px 0; display: inline-block; width: 280px; vertical-align: top; }
.stats { font-size: 11px; color: #888; }
canvas { display: block; margin: 5px 0; }
.summary { background: #2A2A2E; padding: 15px; margin: 10px 0; border-radius: 5px; }
</style></head><body>
<h1>Sperm Whale Coda Signal Analysis</h1>
<div class="summary">
"""
    # Summary stats
    durations = [r["duration_s"] for r in records if "error" not in r]
    n_clicks_list = [r["n_clicks"] for r in records if "error" not in r]
    peak_freqs = [r["peak_freq_hz"] for r in records if "error" not in r and r["peak_freq_hz"] > 0]

    html += f"<b>Total codas:</b> {len(records)}<br>"
    html += f"<b>Duration range:</b> {min(durations):.2f}s - {max(durations):.2f}s (mean {np.mean(durations):.2f}s)<br>"
    html += f"<b>Clicks per coda:</b> {min(n_clicks_list)} - {max(n_clicks_list)} (mean {np.mean(n_clicks_list):.1f})<br>"
    if peak_freqs:
        html += f"<b>Peak frequency range:</b> {min(peak_freqs):.0f}Hz - {max(peak_freqs):.0f}Hz<br>"
    html += "</div>\n"

    # Individual codas (first 100 for file size)
    html += "<h2>Individual Codas (first 100)</h2>\n"
    for r in records[:100]:
        if "error" in r:
            continue
        html += f'<div class="coda">'
        html += f'<b>Coda {r["id"]}</b> '
        html += f'<span class="stats">{r["duration_s"]:.2f}s, {r["n_clicks"]} clicks, {r["peak_freq_hz"]:.0f}Hz peak</span><br>'

        # Draw envelope as SVG
        env = r["envelope"]
        if env:
            max_e = max(env) if max(env) > 0 else 1
            points = " ".join(f"{i*280/len(env)},{40-e/max_e*35}" for i, e in enumerate(env))
            html += f'<svg width="280" height="45" style="background:#111"><polyline points="{points}" fill="none" stroke="#D4A843" stroke-width="1"/></svg>'

        # ICI pattern
        if r["icis_s"]:
            ici_str = " ".join(f"{ici*1000:.0f}ms" for ici in r["icis_s"])
            html += f'<div class="stats">ICIs: {ici_str}</div>'

        html += '</div>\n'

    html += "</body></html>"
    return html


def main():
    log("=" * 70)
    log("WHALE CODA SIGNAL ANALYSIS")
    log(f"Input: {WAV_DIR}")
    log(f"Output: {OUTPUT_DIR}")
    log("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Find all WAV files
    wav_files = sorted(Path(WAV_DIR).glob("*.wav"), key=lambda p: int(p.stem) if p.stem.isdigit() else 0)
    log(f"Found {len(wav_files)} WAV files")

    # Process all
    records = []
    start = time.time()

    with open(OUTPUT_JSONL, "w") as fout:
        for i, wav_path in enumerate(wav_files):
            record = analyze_wav(str(wav_path))
            records.append(record)
            fout.write(json.dumps(record) + "\n")

            if (i + 1) % 100 == 0:
                elapsed = time.time() - start
                log(f"  Processed {i+1}/{len(wav_files)} codas ({elapsed:.0f}s)")

    elapsed = time.time() - start
    errors = sum(1 for r in records if "error" in r)
    log(f"\nProcessed {len(records)} codas in {elapsed:.0f}s ({errors} errors)")

    # Generate HTML overview
    log("Generating HTML overview...")
    html = generate_html(records)
    with open(OUTPUT_HTML, "w") as f:
        f.write(html)
    log(f"HTML saved to {OUTPUT_HTML}")

    # Summary stats
    valid = [r for r in records if "error" not in r]
    durations = [r["duration_s"] for r in valid]
    clicks = [r["n_clicks"] for r in valid]

    log("\n" + "=" * 70)
    log("SUMMARY")
    log(f"  Total codas: {len(valid)}")
    log(f"  Duration: {min(durations):.2f}s - {max(durations):.2f}s (mean {np.mean(durations):.2f}s)")
    log(f"  Clicks: {min(clicks)} - {max(clicks)} (mean {np.mean(clicks):.1f})")

    # Click distribution
    from collections import Counter
    click_dist = Counter(clicks)
    log("  Click count distribution:")
    for n, count in sorted(click_dist.items()):
        log(f"    {n} clicks: {count} codas")

    log(f"\nOutput: {OUTPUT_JSONL}")
    log(f"HTML:   {OUTPUT_HTML}")
    log("=" * 70)


if __name__ == "__main__":
    main()

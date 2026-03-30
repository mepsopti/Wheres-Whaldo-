#!/usr/bin/env python3
"""
Modal Decomposition of Sperm Whale Clicks
==========================================
Decomposes real recorded sperm whale clicks into individual resonant modes
(damped sinusoids) using the Matrix Pencil Method, then maps them to
predicted spermaceti organ cavity modes.

Author: Jaak (Whale Acoustic ID agent)
"""

import numpy as np
import pandas as pd
import json
import os
import warnings
from pathlib import Path
from scipy.io import wavfile
from scipy.signal import hilbert, butter, filtfilt, resample
from scipy.linalg import svd, lstsq
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================
BASE_DIR = Path("/mnt/archive/datasets/whale_communication")
DSWP_DIR = BASE_DIR / "DSWP"
ANALYSIS_DIR = BASE_DIR / "analysis"
CODA_CSV = BASE_DIR / "sw-combinatoriality" / "data" / "DominicaCodas.csv"

TARGET_SR = 44100  # Resample everything to 44.1kHz
CLICK_WINDOW_PRE = 0.005   # 5ms before peak
CLICK_WINDOW_POST = 0.015  # 15ms after peak
MIN_CLICK_SNR = 0.1        # Minimum peak amplitude as fraction of file max
MAX_CLICKS_PER_WHALE = 60  # Limit clicks per whale

# Whale physical parameters (spermaceti organ length, speed of sound)
WHALE_PARAMS = {
    'A': {'L': 3.5, 'c': 1370, 'D': 0.6},
    'D': {'L': 3.8, 'c': 1370, 'D': 0.65},
    'F': {'L': 4.8, 'c': 1370, 'D': 0.8},
}

WHALE_COLORS = {'A': '#1f77b4', 'D': '#ff7f0e', 'F': '#2ca02c'}
WHALE_NAMES = {'A': 'Whale A (L=3.5m)', 'D': 'Whale D (L=3.8m)', 'F': 'Whale F (L=4.8m)'}


# ============================================================
# Step 1: Load and isolate individual clicks
# ============================================================

def get_whale_wav_files():
    """Map whale IDs to WAV file paths using DominicaCodas.csv."""
    dom = pd.read_csv(CODA_CSV)
    wavs_available = set()
    for f in os.listdir(DSWP_DIR):
        if f.endswith('.wav'):
            wavs_available.add(int(f.replace('.wav', '')))

    whale_files = {}
    for whale_id in ['A', 'D', 'F']:
        codas = dom[dom['Unit'] == whale_id]['codaNUM2018'].values
        files = [DSWP_DIR / f"{c}.wav" for c in codas if c in wavs_available]
        whale_files[whale_id] = files
        print(f"  Whale {whale_id}: {len(files)} WAV files available")
    return whale_files


def load_and_normalize_wav(filepath):
    """Load WAV, convert to mono float, resample to TARGET_SR."""
    sr, data = wavfile.read(filepath)
    if data.dtype == np.int16:
        data = data.astype(np.float64) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float64) / 2147483648.0
    elif data.dtype == np.float32:
        data = data.astype(np.float64)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != TARGET_SR:
        n_samples = int(len(data) * TARGET_SR / sr)
        data = resample(data, n_samples)
    return data


def detect_clicks(signal, sr=TARGET_SR, min_gap_ms=10):
    """Detect click peaks using Hilbert envelope."""
    nyq = sr / 2
    b, a = butter(4, [200/nyq, min(20000/nyq, 0.99)], btype='band')
    filtered = filtfilt(b, a, signal)

    analytic = hilbert(filtered)
    envelope = np.abs(analytic)

    win = int(0.001 * sr)
    kernel = np.ones(win) / win
    envelope_smooth = np.convolve(envelope, kernel, mode='same')

    threshold = np.max(envelope_smooth) * 0.3
    above = envelope_smooth > threshold

    min_gap = int(min_gap_ms * sr / 1000)
    peaks = []
    i = 0
    while i < len(above):
        if above[i]:
            start = i
            while i < len(above) and above[i]:
                i += 1
            end = i
            peak_idx = start + np.argmax(envelope_smooth[start:end])
            peaks.append(peak_idx)
            i = end + min_gap
        else:
            i += 1

    return peaks, envelope_smooth


def extract_click_windows(signal, peaks, sr=TARGET_SR):
    """Extract click windows around each peak."""
    pre_samples = int(CLICK_WINDOW_PRE * sr)
    post_samples = int(CLICK_WINDOW_POST * sr)

    clicks = []
    for p in peaks:
        start = p - pre_samples
        end = p + post_samples
        if start >= 0 and end < len(signal):
            click = signal[start:end].copy()
            if np.max(np.abs(click)) > MIN_CLICK_SNR * np.max(np.abs(signal)):
                clicks.append(click)
    return clicks


def collect_clicks_for_whale(wav_files, max_clicks=MAX_CLICKS_PER_WHALE):
    """Collect clean clicks from multiple WAV files for one whale."""
    all_clicks = []
    for wf in wav_files:
        if len(all_clicks) >= max_clicks:
            break
        try:
            sig = load_and_normalize_wav(str(wf))
            peaks, env = detect_clicks(sig)
            clicks = extract_click_windows(sig, peaks)
            all_clicks.extend(clicks)
        except Exception:
            continue
    all_clicks.sort(key=lambda c: np.max(np.abs(c)), reverse=True)
    return all_clicks[:max_clicks]


# ============================================================
# Step 2: Matrix Pencil Method for Damped Sinusoid Decomposition
# ============================================================

def matrix_pencil_method(signal, sr, n_modes_max=30, sv_threshold=0.02):
    """
    Extract damped sinusoidal modes using the Matrix Pencil Method.
    
    For a real-valued signal, eigenvalues come in conjugate pairs.
    We keep only the positive-frequency member of each pair, with
    amplitude doubled to account for the conjugate contribution.
    """
    N = len(signal)
    L = min(N // 3, 300)

    # Build Hankel-like data matrices
    Y = np.zeros((N - L, L + 1))
    for i in range(N - L):
        Y[i, :] = signal[i:i+L+1]

    Y1 = Y[:, :-1]
    Y2 = Y[:, 1:]

    U, s, Vh = svd(Y1, full_matrices=False)

    # Determine number of modes from singular value gap
    s_normalized = s / s[0]
    n_modes = np.sum(s_normalized > sv_threshold)
    n_modes = min(n_modes, n_modes_max)
    n_modes = max(n_modes, 4)

    U_r = U[:, :n_modes]
    s_r = s[:n_modes]
    Vh_r = Vh[:n_modes, :]

    S_inv = np.diag(1.0 / s_r)
    M = U_r.conj().T @ Y2 @ Vh_r.conj().T @ S_inv

    eigenvalues = np.linalg.eigvals(M)

    dt = 1.0 / sr
    
    # Only keep eigenvalues with positive imaginary part (or real positive)
    # to avoid conjugate duplicates
    raw_modes = []
    for z in eigenvalues:
        if np.abs(z) < 1e-10:
            continue
        s_pole = np.log(z) / dt
        freq = np.imag(s_pole) / (2 * np.pi)
        damping = -np.real(s_pole)

        # Keep only positive frequencies
        if freq < 50 or freq > 22000:
            continue
        if damping < 0:  # growing modes are unphysical
            continue

        Q = np.pi * freq / damping if damping > 0 else 0
        raw_modes.append({
            'freq': freq,
            'damping': damping,
            'Q': Q,
            'z': z,
        })

    if not raw_modes:
        return []

    # Deduplicate conjugate pairs: merge modes within 1% frequency of each other
    raw_modes.sort(key=lambda m: m['freq'])
    deduped = []
    used = set()
    for i, m in enumerate(raw_modes):
        if i in used:
            continue
        # Check if there's a near-duplicate
        merged = False
        for j in range(i+1, len(raw_modes)):
            if j in used:
                continue
            if abs(raw_modes[j]['freq'] - m['freq']) / m['freq'] < 0.01:
                # Merge: average frequency/damping, keep the one with lower damping
                merged_mode = {
                    'freq': (m['freq'] + raw_modes[j]['freq']) / 2,
                    'damping': min(m['damping'], raw_modes[j]['damping']),
                    'Q': max(m['Q'], raw_modes[j]['Q']),
                    'z': m['z'],  # keep one for reconstruction
                }
                deduped.append(merged_mode)
                used.add(i)
                used.add(j)
                merged = True
                break
        if not merged and i not in used:
            deduped.append(m)
            used.add(i)

    # Extract amplitudes via least-squares fit
    t = np.arange(N) * dt
    Z_matrix = np.zeros((N, len(deduped)), dtype=complex)
    for j, m in enumerate(deduped):
        Z_matrix[:, j] = m['z'] ** np.arange(N)

    try:
        coeffs, _, _, _ = lstsq(Z_matrix, signal)
    except Exception:
        coeffs = np.linalg.pinv(Z_matrix) @ signal

    for j, m in enumerate(deduped):
        # Double amplitude to account for conjugate pair contribution
        m['amplitude'] = float(np.abs(coeffs[j]) * 2)
        m['phase'] = float(np.angle(coeffs[j]))
        del m['z']

    deduped.sort(key=lambda m: m['amplitude'], reverse=True)
    return deduped


def reconstruct_signal(modes, N, sr):
    """Reconstruct signal from modal decomposition."""
    dt = 1.0 / sr
    t = np.arange(N) * dt
    reconstructed = np.zeros(N)
    for m in modes:
        alpha = m['damping']
        f = m['freq']
        A = m['amplitude']
        phi = m['phase']
        reconstructed += A * np.exp(-alpha * t) * np.cos(2 * np.pi * f * t + phi)
    return reconstructed


# ============================================================
# Step 3: Predicted Cavity Modes
# ============================================================

def compute_predicted_modes(whale_id, n_longitudinal=15, n_transverse=5):
    """Compute predicted longitudinal and transverse cavity modes."""
    params = WHALE_PARAMS[whale_id]
    L = params['L']
    c = params['c']
    D = params['D']

    modes = {'longitudinal': [], 'transverse': [], 'combined': []}

    for n in range(1, n_longitudinal + 1):
        f = n * c / (2 * L)
        modes['longitudinal'].append({'n': n, 'freq': f})
        modes['combined'].append({'type': 'longitudinal', 'n': n, 'freq': f})

    for m in range(1, n_transverse + 1):
        for n in range(0, 3):
            f = c * np.sqrt((m / D)**2 + (n / (2 * L))**2)
            if f < 22000:
                modes['transverse'].append({'m': m, 'n': n, 'freq': f})
                modes['combined'].append({'type': f'transverse_m{m}_n{n}', 'freq': f})

    return modes


# ============================================================
# Step 4: Ramp-Up Analysis
# ============================================================

def envelope_model(t, A_max, tau_up, tau_down, t0):
    """Ramp-up/decay envelope model."""
    t_shifted = t - t0
    result = np.zeros_like(t)
    mask = t_shifted > 0
    result[mask] = A_max * (1 - np.exp(-t_shifted[mask] / tau_up)) * np.exp(-t_shifted[mask] / tau_down)
    return result


def fit_ramp_up(click, sr=TARGET_SR):
    """Fit ramp-up/decay model to click envelope."""
    analytic = hilbert(click)
    envelope = np.abs(analytic)

    win = max(int(0.0005 * sr), 3)
    kernel = np.ones(win) / win
    envelope = np.convolve(envelope, kernel, mode='same')

    t = np.arange(len(click)) / sr
    peak_idx = np.argmax(envelope)
    peak_time = t[peak_idx]

    try:
        p0 = [np.max(envelope), 0.001, 0.003, t[max(0, peak_idx - int(0.003*sr))]]
        bounds = ([0, 0.0001, 0.0001, 0], [np.max(envelope)*2, 0.01, 0.05, peak_time + 0.001])
        popt, pcov = curve_fit(envelope_model, t, envelope, p0=p0, bounds=bounds, maxfev=5000)
        return {
            'A_max': popt[0],
            'tau_up': popt[1],
            'tau_down': popt[2],
            't0': popt[3],
            'envelope': envelope,
            'fitted': envelope_model(t, *popt),
            'time': t,
        }
    except Exception:
        return {
            'A_max': np.max(envelope),
            'tau_up': None,
            'tau_down': None,
            't0': None,
            'envelope': envelope,
            'fitted': None,
            'time': t,
        }


# ============================================================
# Main Analysis
# ============================================================

def run_analysis():
    print("=" * 70)
    print("MODAL DECOMPOSITION OF SPERM WHALE CLICKS")
    print("=" * 70)

    # --- Step 1: Collect clicks ---
    print("\n[Step 1] Loading WAV files and extracting clicks...")
    whale_wavs = get_whale_wav_files()

    whale_clicks = {}
    for wid in ['A', 'D', 'F']:
        print(f"\n  Processing Whale {wid}...")
        clicks = collect_clicks_for_whale(whale_wavs[wid])
        whale_clicks[wid] = clicks
        print(f"  -> Collected {len(clicks)} clean clicks")

    # --- Step 2: Modal decomposition ---
    print("\n" + "=" * 70)
    print("[Step 2] Matrix Pencil Method - Damped Sinusoid Decomposition")
    print("=" * 70)

    whale_modes = {}
    whale_all_modes = {}

    for wid in ['A', 'D', 'F']:
        print(f"\n--- Whale {wid} ---")
        click_modes_list = []
        all_modes_flat = []

        for i, click in enumerate(whale_clicks[wid]):
            peak_idx = np.argmax(np.abs(click))
            analysis_window = click[peak_idx:]
            if len(analysis_window) < 100:
                continue

            modes = matrix_pencil_method(analysis_window, TARGET_SR, n_modes_max=25)
            if modes:
                click_modes_list.append(modes)
                all_modes_flat.extend(modes)

        whale_modes[wid] = click_modes_list
        whale_all_modes[wid] = all_modes_flat

        if click_modes_list:
            n_modes_per_click = [len(m) for m in click_modes_list]
            print(f"  Analyzed {len(click_modes_list)} clicks")
            print(f"  Modes per click: mean={np.mean(n_modes_per_click):.1f}, "
                  f"median={np.median(n_modes_per_click):.0f}, "
                  f"range=[{np.min(n_modes_per_click)}-{np.max(n_modes_per_click)}]")

            # Print top modes
            freq_amp_pairs = sorted(all_modes_flat, key=lambda m: m['amplitude'], reverse=True)
            print(f"\n  Top 15 modes (highest amplitude across all clicks):")
            print(f"  {'#':>3} {'Freq (Hz)':>10} {'Amplitude':>12} {'Damping (1/s)':>14} "
                  f"{'Q factor':>10} {'tau (ms)':>10}")
            seen_freqs = set()
            rank = 0
            for m in freq_amp_pairs:
                # Skip if we already printed a mode at a very similar frequency
                freq_bin = round(m['freq'] / 50) * 50
                if freq_bin in seen_freqs:
                    continue
                seen_freqs.add(freq_bin)
                rank += 1
                tau = 1000.0 / m['damping'] if m['damping'] > 0 else float('inf')
                print(f"  {rank:3d} {m['freq']:10.1f} {m['amplitude']:12.6f} "
                      f"{m['damping']:14.1f} {m['Q']:10.1f} {tau:10.3f}")
                if rank >= 15:
                    break

    # --- Step 3: Predicted cavity modes ---
    print("\n" + "=" * 70)
    print("[Step 3] Predicted Cavity Modes")
    print("=" * 70)

    predicted_modes = {}
    for wid in ['A', 'D', 'F']:
        predicted_modes[wid] = compute_predicted_modes(wid)
        p = WHALE_PARAMS[wid]
        print(f"\n--- Whale {wid} (L={p['L']}m, D={p['D']}m, c={p['c']}m/s) ---")
        print(f"  Longitudinal modes f_n = n*{p['c']}/(2*{p['L']}):")
        for m in predicted_modes[wid]['longitudinal'][:10]:
            print(f"    n={m['n']:2d}: {m['freq']:8.1f} Hz")
        print(f"  Transverse modes:")
        for m in predicted_modes[wid]['transverse'][:6]:
            print(f"    m={m['m']}, n={m['n']}: {m['freq']:.1f} Hz")

    # --- Mode matching ---
    print("\n" + "=" * 70)
    print("[Step 3b] Matching Detected Modes to Predicted Cavity Modes")
    print("=" * 70)

    for wid in ['A', 'D', 'F']:
        print(f"\n--- Whale {wid} ---")
        if not whale_all_modes.get(wid):
            continue
        # Get the strongest detected modes (deduplicated by frequency)
        sorted_modes = sorted(whale_all_modes[wid], key=lambda m: m['amplitude'], reverse=True)
        top_detected = []
        seen = set()
        for m in sorted_modes:
            fb = round(m['freq'] / 100) * 100
            if fb not in seen:
                seen.add(fb)
                top_detected.append(m)
            if len(top_detected) >= 10:
                break
        
        pred_long = predicted_modes[wid]['longitudinal']
        pred_trans = predicted_modes[wid]['transverse']
        all_pred = [(pm['freq'], f"n={pm['n']} longitudinal") for pm in pred_long]
        all_pred += [(pm['freq'], f"m={pm['m']},n={pm['n']} transverse") for pm in pred_trans]
        
        print(f"  {'Detected (Hz)':>14} {'Nearest Predicted':>18} {'Predicted Type':>25} {'Error %':>10}")
        for dm in top_detected:
            best_err = float('inf')
            best_pred = None
            for pf, ptype in all_pred:
                err = abs(dm['freq'] - pf) / pf * 100
                if err < best_err:
                    best_err = err
                    best_pred = (pf, ptype)
            if best_pred:
                print(f"  {dm['freq']:14.1f} {best_pred[0]:18.1f} {best_pred[1]:>25} {best_err:9.1f}%")

    # --- Step 4: Ramp-up analysis ---
    print("\n" + "=" * 70)
    print("[Step 4] Ramp-Up Envelope Analysis")
    print("=" * 70)

    whale_rampup = {}
    for wid in ['A', 'D', 'F']:
        print(f"\n--- Whale {wid} ---")
        rampups = []
        for click in whale_clicks[wid][:30]:
            result = fit_ramp_up(click)
            if result['tau_up'] is not None:
                rampups.append(result)

        whale_rampup[wid] = rampups

        if rampups:
            tau_ups = [r['tau_up'] * 1000 for r in rampups]
            tau_downs = [r['tau_down'] * 1000 for r in rampups]
            print(f"  Fitted {len(rampups)}/{min(30, len(whale_clicks[wid]))} clicks")
            print(f"  tau_up:   mean={np.mean(tau_ups):.3f} +/- {np.std(tau_ups):.3f} ms, "
                  f"range=[{np.min(tau_ups):.3f} - {np.max(tau_ups):.3f}] ms")
            print(f"  tau_down: mean={np.mean(tau_downs):.3f} +/- {np.std(tau_downs):.3f} ms, "
                  f"range=[{np.min(tau_downs):.3f} - {np.max(tau_downs):.3f}] ms")
            print(f"  Ratio tau_down/tau_up: {np.mean(tau_downs)/np.mean(tau_ups):.1f}x")

            # Compare to predicted
            if whale_all_modes.get(wid):
                top_mode = sorted(whale_all_modes[wid], key=lambda m: m['amplitude'], reverse=True)[0]
                predicted_tau = top_mode['Q'] / (np.pi * top_mode['freq']) * 1000 if top_mode['freq'] > 0 else 0
                print(f"  Dominant mode: f={top_mode['freq']:.0f}Hz, Q={top_mode['Q']:.1f}")
                print(f"  Predicted decay tau = Q/(pi*f) = {predicted_tau:.3f} ms")
                print(f"  Measured tau_down = {np.mean(tau_downs):.3f} ms")

    # --- Step 5: Mode fingerprints ---
    print("\n" + "=" * 70)
    print("[Step 5] Mode Fingerprints - Cross-Whale Comparison")
    print("=" * 70)

    fingerprints = {}
    for wid in ['A', 'D', 'F']:
        if not whale_all_modes.get(wid):
            continue

        # Cluster modes by frequency (within 5% bands)
        modes_sorted = sorted(whale_all_modes[wid], key=lambda m: m['freq'])
        clusters = []
        current_cluster = [modes_sorted[0]]

        for m in modes_sorted[1:]:
            if m['freq'] < current_cluster[-1]['freq'] * 1.05:
                current_cluster.append(m)
            else:
                clusters.append(current_cluster)
                current_cluster = [m]
        clusters.append(current_cluster)

        fp = []
        for cluster in clusters:
            if len(cluster) >= 3:  # Require at least 3 occurrences
                fp.append({
                    'freq': np.mean([m['freq'] for m in cluster]),
                    'freq_std': np.std([m['freq'] for m in cluster]),
                    'amplitude': np.mean([m['amplitude'] for m in cluster]),
                    'Q': np.mean([m['Q'] for m in cluster]),
                    'Q_std': np.std([m['Q'] for m in cluster]),
                    'count': len(cluster),
                    'consistency': len(cluster) / len(whale_modes[wid]),  # fraction of clicks containing this mode
                })

        fp.sort(key=lambda x: x['amplitude'], reverse=True)
        fingerprints[wid] = fp

        print(f"\n--- Whale {wid} Mode Fingerprint (top 12) ---")
        print(f"  {'Freq (Hz)':>10} {'Freq SD':>8} {'Amplitude':>10} {'Q':>8} "
              f"{'Q SD':>8} {'Count':>6} {'Consistency':>12}")
        for m in fp[:12]:
            print(f"  {m['freq']:10.1f} {m['freq_std']:8.1f} {m['amplitude']:10.6f} "
                  f"{m['Q']:8.1f} {m['Q_std']:8.1f} {m['count']:6d} "
                  f"{m['consistency']:11.0%}")

    # --- Cross-whale comparison ---
    print("\n--- Cross-Whale Mode Comparison ---")
    print("Modes unique to each whale (not within 10% of any mode in other whales):")
    for wid in ['A', 'D', 'F']:
        other_whales = [w for w in ['A', 'D', 'F'] if w != wid]
        other_freqs = []
        for ow in other_whales:
            other_freqs.extend([m['freq'] for m in fingerprints.get(ow, [])])
        
        unique = []
        for m in fingerprints.get(wid, []):
            is_unique = True
            for of in other_freqs:
                if abs(m['freq'] - of) / m['freq'] < 0.10:
                    is_unique = False
                    break
            if is_unique:
                unique.append(m)
        
        print(f"\n  Whale {wid}: {len(unique)} unique modes")
        for m in unique[:5]:
            print(f"    {m['freq']:.1f} Hz (amp={m['amplitude']:.6f}, Q={m['Q']:.1f}, "
                  f"consistency={m['consistency']:.0%})")

    # ============================================================
    # Step 6: Visualization
    # ============================================================
    print("\n" + "=" * 70)
    print("[Step 6] Generating visualization...")
    print("=" * 70)

    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    # --- Panel A: Mode Spectrum per Whale ---
    ax_a = fig.add_subplot(gs[0, 0])
    for wid in ['A', 'D', 'F']:
        if wid not in fingerprints:
            continue
        fp = fingerprints[wid]
        freqs = [m['freq'] for m in fp[:25]]
        amps = [m['amplitude'] for m in fp[:25]]
        if not amps:
            continue
        max_amp = max(amps)
        amps_norm = [a / max_amp for a in amps]

        # Offset slightly for visibility
        offset = {'A': -30, 'D': 0, 'F': 30}
        for f_val, a_val in zip(freqs, amps_norm):
            ax_a.vlines(f_val + offset[wid], 0, a_val, colors=WHALE_COLORS[wid],
                       alpha=0.7, linewidth=2.5)

        # Label for legend (only first)
        ax_a.vlines([], 0, 0, colors=WHALE_COLORS[wid], linewidth=2.5,
                    label=WHALE_NAMES[wid])

    # Predicted cavity modes as dashed lines
    for wid in ['A', 'D', 'F']:
        for pm in predicted_modes[wid]['longitudinal'][:8]:
            ax_a.axvline(pm['freq'], color=WHALE_COLORS[wid], linestyle='--',
                        alpha=0.2, linewidth=1)

    ax_a.set_xlabel('Frequency (Hz)', fontsize=12)
    ax_a.set_ylabel('Normalized Mode Amplitude', fontsize=12)
    ax_a.set_title('A. Detected Resonant Modes per Whale\n(dashed = predicted cavity modes)',
                   fontsize=13, fontweight='bold')
    ax_a.legend(fontsize=10, loc='upper right')
    ax_a.set_xlim(0, 18000)
    ax_a.set_ylim(0, 1.15)
    ax_a.grid(True, alpha=0.3)

    # --- Panel B: Damping Map ---
    ax_b = fig.add_subplot(gs[0, 1])
    for wid in ['A', 'D', 'F']:
        if wid not in fingerprints:
            continue
        fp = fingerprints[wid]
        valid = [m for m in fp if 0 < m['Q'] < 500]
        if not valid:
            continue
        freqs = [m['freq'] for m in valid]
        Qs = [m['Q'] for m in valid]
        max_amp = max(m['amplitude'] for m in valid)
        sizes = [max(20, m['amplitude'] / max_amp * 300) for m in valid]
        ax_b.scatter(freqs, Qs, s=sizes, c=WHALE_COLORS[wid], alpha=0.6,
                    edgecolors='black', linewidth=0.5, label=WHALE_NAMES[wid],
                    zorder=3)

    ax_b.set_xlabel('Frequency (Hz)', fontsize=12)
    ax_b.set_ylabel('Q Factor (Quality)', fontsize=12)
    ax_b.set_title('B. Damping Map\n(size = relative amplitude)', fontsize=13, fontweight='bold')
    ax_b.legend(fontsize=10, loc='upper left')
    ax_b.set_xlim(0, 18000)
    ax_b.grid(True, alpha=0.3)
    ax_b.set_yscale('log')

    # --- Panel C: Ramp-Up Envelope ---
    ax_c = fig.add_subplot(gs[1, 0])
    for wid in ['A', 'D', 'F']:
        if not whale_rampup.get(wid):
            continue
        rampups = whale_rampup[wid]
        target_len = len(rampups[0]['time'])
        avg_env = np.zeros(target_len)
        count = 0
        for r in rampups:
            if len(r['envelope']) == target_len:
                mx = np.max(r['envelope'])
                if mx > 0:
                    avg_env += r['envelope'] / mx
                    count += 1
        if count > 0:
            avg_env /= count
            t_ms = rampups[0]['time'] * 1000

            ax_c.plot(t_ms, avg_env, color=WHALE_COLORS[wid], linewidth=2,
                     label=f'{WHALE_NAMES[wid]} (n={count})')

            # Best fitted model
            best = next((r for r in rampups if r['fitted'] is not None), None)
            if best and best['fitted'] is not None:
                mx = np.max(best['fitted'])
                if mx > 0:
                    fitted_norm = best['fitted'] / mx
                    ax_c.plot(t_ms, fitted_norm, color=WHALE_COLORS[wid],
                             linestyle='--', alpha=0.5, linewidth=1.5)

            # Mark tau_up and tau_down
            tau_up_avg = np.mean([r['tau_up'] * 1000 for r in rampups if r['tau_up']])
            tau_down_avg = np.mean([r['tau_down'] * 1000 for r in rampups if r['tau_down']])

    ax_c.axvline(CLICK_WINDOW_PRE * 1000, color='gray', linestyle=':', alpha=0.5, label='Click onset')
    ax_c.set_xlabel('Time (ms)', fontsize=12)
    ax_c.set_ylabel('Normalized Amplitude', fontsize=12)
    ax_c.set_title('C. Average Click Envelope\n(solid = measured, dashed = ramp-up/decay fit)',
                   fontsize=13, fontweight='bold')
    ax_c.legend(fontsize=9, loc='upper right')
    ax_c.grid(True, alpha=0.3)

    # --- Panel D: Mode Reconstruction ---
    ax_d = fig.add_subplot(gs[1, 1])

    best_wid = 'A'
    if whale_clicks.get(best_wid) and whale_modes.get(best_wid):
        best_click = whale_clicks[best_wid][0]
        peak_idx = np.argmax(np.abs(best_click))
        analysis_part = best_click[peak_idx:]

        best_modes = whale_modes[best_wid][0] if whale_modes[best_wid] else []

        if best_modes:
            n_recon = min(10, len(best_modes))
            top_modes = best_modes[:n_recon]
            reconstructed = reconstruct_signal(top_modes, len(analysis_part), TARGET_SR)

            t_ms = np.arange(len(analysis_part)) / TARGET_SR * 1000
            orig_norm = analysis_part / np.max(np.abs(analysis_part))
            recon_max = np.max(np.abs(reconstructed))
            recon_norm = reconstructed / recon_max if recon_max > 0 else reconstructed

            ax_d.plot(t_ms, orig_norm, color='#1f77b4', linewidth=1,
                     alpha=0.8, label='Original click')
            ax_d.plot(t_ms, recon_norm, color='red', linestyle='--',
                     linewidth=1.2, alpha=0.8,
                     label=f'Top {n_recon} modes')
            residual = orig_norm - recon_norm
            ax_d.fill_between(t_ms, residual, 0, color='gray', alpha=0.2,
                            label='Residual')

            corr = np.corrcoef(analysis_part, reconstructed)[0, 1]
            rmse = np.sqrt(np.mean((orig_norm - recon_norm)**2))
            ax_d.text(0.02, 0.02,
                     f'Correlation: {corr:.3f}\nRMSE: {rmse:.3f}',
                     transform=ax_d.transAxes, fontsize=10,
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
                     verticalalignment='bottom')

            # Print reconstruction info
            print(f"\n  Reconstruction quality (Whale A, best click):")
            print(f"    Modes used: {n_recon}")
            print(f"    Correlation: {corr:.4f}")
            print(f"    RMSE (normalized): {rmse:.4f}")
            freqlist = [str(round(m["freq"])) + "Hz" for m in top_modes]
            print(f"    Mode frequencies: {freqlist}")

    ax_d.set_xlabel('Time (ms)', fontsize=12)
    ax_d.set_ylabel('Normalized Amplitude', fontsize=12)
    ax_d.set_title('D. Modal Reconstruction (Whale A, best click)',
                   fontsize=13, fontweight='bold')
    ax_d.legend(fontsize=10, loc='upper right')
    ax_d.grid(True, alpha=0.3)

    plt.suptitle('Sperm Whale Click Modal Decomposition\n'
                'Matrix Pencil Method - Damped Sinusoid Analysis',
                fontsize=16, fontweight='bold', y=0.99)

    fig.savefig(ANALYSIS_DIR / 'modal_decomposition.png', dpi=300, bbox_inches='tight',
                facecolor='white')
    print(f"\n  Figure saved: {ANALYSIS_DIR / 'modal_decomposition.png'}")
    plt.close()

    # ============================================================
    # Save results JSON
    # ============================================================
    results = {
        'metadata': {
            'analysis': 'Modal Decomposition of Sperm Whale Clicks',
            'method': 'Matrix Pencil Method (deduped conjugate pairs)',
            'sample_rate': TARGET_SR,
            'click_window': f'{CLICK_WINDOW_PRE*1000}ms pre, {CLICK_WINDOW_POST*1000}ms post',
            'sv_threshold': 0.02,
        },
        'whales': {}
    }

    for wid in ['A', 'D', 'F']:
        whale_result = {
            'physical_params': WHALE_PARAMS[wid],
            'n_clicks_analyzed': len(whale_modes.get(wid, [])),
            'n_total_clicks_extracted': len(whale_clicks.get(wid, [])),
            'predicted_modes': {
                'longitudinal': [{'n': m['n'], 'freq_hz': round(m['freq'], 1)}
                                for m in predicted_modes[wid]['longitudinal'][:10]],
                'transverse': [{'freq_hz': round(m['freq'], 1)}
                              for m in predicted_modes[wid]['transverse'][:6]],
            },
            'detected_modes_fingerprint': [
                {
                    'freq_hz': round(m['freq'], 1),
                    'freq_std_hz': round(m['freq_std'], 1),
                    'amplitude': round(float(m['amplitude']), 6),
                    'Q_factor': round(float(m['Q']), 1),
                    'Q_std': round(float(m['Q_std']), 1),
                    'n_occurrences': m['count'],
                    'consistency': round(float(m['consistency']), 3),
                }
                for m in fingerprints.get(wid, [])[:15]
            ],
        }

        if whale_rampup.get(wid):
            rampups = [r for r in whale_rampup[wid] if r['tau_up'] is not None]
            if rampups:
                whale_result['ramp_up'] = {
                    'n_fitted': len(rampups),
                    'tau_up_ms': {
                        'mean': round(np.mean([r['tau_up']*1000 for r in rampups]), 4),
                        'std': round(np.std([r['tau_up']*1000 for r in rampups]), 4),
                        'min': round(np.min([r['tau_up']*1000 for r in rampups]), 4),
                        'max': round(np.max([r['tau_up']*1000 for r in rampups]), 4),
                    },
                    'tau_down_ms': {
                        'mean': round(np.mean([r['tau_down']*1000 for r in rampups]), 4),
                        'std': round(np.std([r['tau_down']*1000 for r in rampups]), 4),
                        'min': round(np.min([r['tau_down']*1000 for r in rampups]), 4),
                        'max': round(np.max([r['tau_down']*1000 for r in rampups]), 4),
                    },
                }

        results['whales'][wid] = whale_result

    with open(ANALYSIS_DIR / 'modal_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved: {ANALYSIS_DIR / 'modal_results.json'}")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    run_analysis()

#!/usr/bin/env python3
"""
Muscle Tension Optimizer for Sperm Whale Click Synthesis
========================================================
Uses scipy.optimize.differential_evolution to find optimal muscle tension
parameters (and envelope/anatomy tweaks) for each whale, minimizing the
difference between synthesized and real spectral features.

Imports all physics from mode_synthesizer.py - this is just the optimization wrapper.

Output:
    tension_optimized.png          - 3-row comparison figure
    tension_optimized_results.json - full parameter dump
    whale_{A,D,F}_optimized.wav    - optimized click audio
"""

import json
import sys
import time
import warnings
import numpy as np
from pathlib import Path
from scipy.optimize import differential_evolution
from scipy.io import wavfile

warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Import everything from the mode synthesizer
sys.path.insert(0, str(Path(__file__).parent))
from mode_synthesizer import (
    compute_cavity_modes,
    compute_base_amplitudes,
    apply_muscle_damping,
    synthesize_click,
    apply_exit_path_filter,
    apply_ocean_propagation,
    compute_band_energies,
    compute_spectral_centroid,
    save_wav,
    get_whale_wav_files,
    get_mean_real_click,
    WHALE_CONFIGS,
    REAL_TARGETS,
    TARGET_SR,
    OUTPUT_DIR,
)

# ============================================================
# OPTIMIZATION PARAMETERS
# ============================================================
PARAM_NAMES = [
    'tension_0',      # muscle band 1 (highest freq, ~16kHz)
    'tension_1',      # band 2 (~12kHz)
    'tension_2',      # band 3 (~9kHz)
    'tension_3',      # band 4 (~7kHz)
    'tension_4',      # band 5 (~5kHz)
    'tension_5',      # band 6 (~3.5kHz)
    'tau_up',         # envelope ramp-up time (s)
    'tau_down',       # envelope decay time (s)
    'spermaceti_c',   # sound speed (temperature effect)
    'organ_length',   # slight adjustment
    'organ_diameter', # slight adjustment
]

PARAM_BOUNDS = [
    (0.0, 1.0),       # tension_0
    (0.0, 1.0),       # tension_1
    (0.0, 1.0),       # tension_2
    (0.0, 1.0),       # tension_3
    (0.0, 1.0),       # tension_4
    (0.0, 1.0),       # tension_5
    (0.0001, 0.005),  # tau_up (0.1ms to 5ms)
    (0.0005, 0.030),  # tau_down (0.5ms to 30ms)
    (1350, 1530),     # spermaceti_c (temperature range)
    (2.5, 6.0),       # organ_length (m)
    (0.8, 2.5),       # organ_diameter (m)
]

# Muscle band center frequencies (from WHALE_CONFIGS)
MUSCLE_BAND_FREQS = [16000, 12000, 9000, 7000, 5000, 3500]

# Real targets - using the exact keys from mode_synthesizer
TARGETS = {
    'Whale_A': {
        'spectral_centroid_hz': 7849,
        'band_sub_100hz_pct': 0.5,
        'band_100_500hz_pct': 0.75,
        'band_500_2khz_pct': 5.5,
        'band_2_5khz_pct': 19.2,
        'band_5_10khz_pct': 49.5,
        'band_10_20khz_pct': 22.3,
        'band_above_20khz_pct': 2.28,
    },
    'Whale_D': {
        'spectral_centroid_hz': 5693,
        'band_sub_100hz_pct': 16.1,
        'band_100_500hz_pct': 1.52,
        'band_500_2khz_pct': 12.2,
        'band_2_5khz_pct': 18.7,
        'band_5_10khz_pct': 31.4,
        'band_10_20khz_pct': 17.7,
        'band_above_20khz_pct': 1.14,
    },
    'Whale_F': {
        'spectral_centroid_hz': 5333,
        'band_sub_100hz_pct': 29.5,
        'band_100_500hz_pct': 2.80,
        'band_500_2khz_pct': 7.1,
        'band_2_5khz_pct': 15.4,
        'band_5_10khz_pct': 24.7,
        'band_10_20khz_pct': 19.4,
        'band_above_20khz_pct': 1.1,
    },
}


# ============================================================
# COST FUNCTION
# ============================================================
def synthesize_from_params(params, junk_length=2.0):
    """Run the full synthesis pipeline from a flat parameter vector.
    Returns the final signal and dt."""
    tensions = list(params[:6])
    tau_up = params[6]
    tau_down = params[7]
    spermaceti_c = params[8]
    organ_length = params[9]
    organ_diameter = params[10]

    # 1. Cavity modes from anatomy
    modes = compute_cavity_modes(organ_length, organ_diameter, spermaceti_c)
    # 2. Base amplitudes
    modes = compute_base_amplitudes(modes, organ_length)
    # 3. Muscle damping
    modes = apply_muscle_damping(modes, tensions, MUSCLE_BAND_FREQS)

    # 4. Synthesize
    duration_s = tau_up + 5 * tau_down
    duration_s = max(duration_s, 0.015)
    duration_s = min(duration_s, 0.150)
    signal, t = synthesize_click(modes, sample_rate=TARGET_SR, duration_s=duration_s,
                                  tau_up_s=tau_up, tau_down_s=tau_down)

    dt = 1.0 / TARGET_SR

    # 5. Exit-path filter
    signal = apply_exit_path_filter(signal, dt, junk_length=junk_length)
    # 6. Ocean propagation
    signal = apply_ocean_propagation(signal, dt)

    return signal, dt


def cost_function(params, target, junk_length=2.0):
    """Compute cost: weighted sum of squared normalized errors vs real data."""
    np.random.seed(42)  # deterministic for optimizer

    try:
        signal, dt = synthesize_from_params(params, junk_length=junk_length)
    except Exception:
        return 100.0  # penalty for invalid params

    # Extract features
    bands = compute_band_energies(signal, dt)
    centroid = compute_spectral_centroid(signal, dt)

    if centroid < 100 or not bands:
        return 100.0  # degenerate signal

    # Cost: weighted sum of squared normalized errors
    error = 0.0

    # Centroid (weight 3.0) - most important
    error += 3.0 * ((centroid - target['spectral_centroid_hz']) / target['spectral_centroid_hz'])**2

    # Primary bands (weight 2.0) - the key identity bands
    primary_keys = ['band_2_5khz_pct', 'band_5_10khz_pct', 'band_10_20khz_pct']
    for key in primary_keys:
        real_val = target.get(key, 0)
        synth_val = bands.get(key, 0)
        if real_val > 1.0:
            error += 2.0 * ((synth_val - real_val) / real_val)**2

    # Secondary bands (weight 1.0) - still important for overall shape
    secondary_keys = ['band_sub_100hz_pct', 'band_500_2khz_pct']
    for key in secondary_keys:
        real_val = target.get(key, 0)
        synth_val = bands.get(key, 0)
        if real_val > 1.0:
            error += 1.0 * ((synth_val - real_val) / real_val)**2

    # Minor bands (weight 0.5)
    minor_keys = ['band_100_500hz_pct', 'band_above_20khz_pct']
    for key in minor_keys:
        real_val = target.get(key, 0)
        synth_val = bands.get(key, 0)
        if real_val > 1.0:
            error += 0.5 * ((synth_val - real_val) / real_val)**2

    return error


# ============================================================
# PROGRESS CALLBACK
# ============================================================
class ProgressTracker:
    def __init__(self, whale_name, target):
        self.whale_name = whale_name
        self.target = target
        self.iteration = 0
        self.best_cost = float('inf')
        self.start_time = time.time()
        self.history = []

    def callback(self, xk, convergence=0):
        self.iteration += 1
        np.random.seed(42)
        try:
            signal, dt = synthesize_from_params(xk, junk_length=self._junk_length)
            bands = compute_band_energies(signal, dt)
            centroid = compute_spectral_centroid(signal, dt)
            current_cost = cost_function(xk, self.target, junk_length=self._junk_length)
        except Exception:
            return

        if current_cost < self.best_cost:
            self.best_cost = current_cost

        self.history.append(current_cost)

        if self.iteration % 10 == 0 or self.iteration == 1:
            elapsed = time.time() - self.start_time
            centroid_err = (centroid - self.target['spectral_centroid_hz']) / self.target['spectral_centroid_hz'] * 100
            band_5_10_err = 0
            real_5_10 = self.target.get('band_5_10khz_pct', 0)
            synth_5_10 = bands.get('band_5_10khz_pct', 0)
            if real_5_10 > 1:
                band_5_10_err = (synth_5_10 - real_5_10) / real_5_10 * 100

            tensions = xk[:6]
            tau_up_ms = xk[6] * 1000
            tau_down_ms = xk[7] * 1000
            c = xk[8]
            L = xk[9]
            D = xk[10]

            print(f"[{self.whale_name} Iter {self.iteration:3d}] cost={current_cost:.4f}  "
                  f"centroid_err={centroid_err:+.1f}%  5-10kHz_err={band_5_10_err:+.1f}%  "
                  f"({elapsed:.0f}s)")
            print(f"  tensions=[{', '.join(f'{t:.2f}' for t in tensions)}]  "
                  f"tau_up={tau_up_ms:.2f}ms  tau_down={tau_down_ms:.1f}ms  "
                  f"c={c:.0f}  L={L:.2f}  D={D:.2f}")


# ============================================================
# RUN OPTIMIZATION FOR ONE WHALE
# ============================================================
def optimize_whale(whale_name, target, initial_config):
    """Run differential_evolution for one whale."""
    print(f"\n{'='*70}")
    print(f"OPTIMIZING {whale_name}")
    print(f"{'='*70}")
    print(f"Target centroid: {target['spectral_centroid_hz']} Hz")
    print(f"Target 5-10kHz: {target['band_5_10khz_pct']}%")
    print(f"Target 2-5kHz:  {target['band_2_5khz_pct']}%")
    print(f"Target 10-20kHz: {target['band_10_20khz_pct']}%")

    # Initial guess from current config
    x0 = [
        *initial_config['muscle_tensions'],  # 6 tensions
        initial_config['tau_up'],
        initial_config['tau_down'],
        initial_config['spermaceti_c'],
        initial_config['organ_length'],
        initial_config['organ_diameter'],
    ]
    junk_length = initial_config.get('junk_length', 2.0)

    # Evaluate initial cost
    np.random.seed(42)
    init_cost = cost_function(x0, target, junk_length=junk_length)
    print(f"\nInitial cost: {init_cost:.4f}")
    print(f"Initial params: tensions={[f'{t:.2f}' for t in x0[:6]]}")
    print(f"  tau_up={x0[6]*1000:.2f}ms  tau_down={x0[7]*1000:.1f}ms  "
          f"c={x0[8]:.0f}  L={x0[9]:.2f}  D={x0[10]:.2f}")

    # Setup progress tracker
    tracker = ProgressTracker(whale_name, target)
    tracker._junk_length = junk_length

    # Time a single evaluation for ETA
    t0 = time.time()
    np.random.seed(42)
    _ = cost_function(x0, target, junk_length=junk_length)
    eval_time = time.time() - t0
    print(f"\nSingle eval time: {eval_time*1000:.1f}ms")

    popsize = 20
    maxiter = 200
    total_evals = popsize * len(PARAM_BOUNDS) * (maxiter + 1)
    eta_min = total_evals * eval_time / 60
    print(f"Estimated: {total_evals} evals, ~{eta_min:.1f} minutes")
    print(f"\nStarting differential_evolution (popsize={popsize}, maxiter={maxiter})...")
    print("-" * 70)

    start = time.time()
    result = differential_evolution(
        cost_function,
        bounds=PARAM_BOUNDS,
        args=(target, junk_length),
        seed=42,
        popsize=popsize,
        maxiter=maxiter,
        tol=1e-6,
        mutation=(0.5, 1.5),
        recombination=0.8,
        callback=tracker.callback,
        disp=False,
        init='latinhypercube',
        workers=1,  # deterministic
    )
    elapsed = time.time() - start

    print(f"\n{'='*70}")
    print(f"{whale_name} OPTIMIZATION COMPLETE")
    print(f"{'='*70}")
    print(f"  Final cost: {result.fun:.6f}")
    print(f"  Iterations: {result.nit}")
    print(f"  Function evals: {result.nfev}")
    print(f"  Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Success: {result.success}")
    print(f"  Message: {result.message}")

    # Extract optimized params
    opt = result.x
    tensions = list(opt[:6])
    tau_up = opt[6]
    tau_down = opt[7]
    spermaceti_c = opt[8]
    organ_length = opt[9]
    organ_diameter = opt[10]

    print(f"\n  Optimized tensions: [{', '.join(f'{t:.3f}' for t in tensions)}]")
    print(f"  tau_up:       {tau_up*1000:.3f} ms")
    print(f"  tau_down:     {tau_down*1000:.3f} ms")
    print(f"  spermaceti_c: {spermaceti_c:.1f} m/s")
    print(f"  organ_length: {organ_length:.3f} m")
    print(f"  organ_diameter: {organ_diameter:.3f} m")

    # Generate final click and extract features
    np.random.seed(42)
    signal, dt = synthesize_from_params(opt, junk_length=junk_length)
    bands = compute_band_energies(signal, dt)
    centroid = compute_spectral_centroid(signal, dt)

    print(f"\n  Optimized centroid: {centroid:.0f} Hz (target: {target['spectral_centroid_hz']})")
    band_keys_display = [
        ('band_sub_100hz_pct', '<100Hz'),
        ('band_100_500hz_pct', '100-500Hz'),
        ('band_500_2khz_pct', '0.5-2kHz'),
        ('band_2_5khz_pct', '2-5kHz'),
        ('band_5_10khz_pct', '5-10kHz'),
        ('band_10_20khz_pct', '10-20kHz'),
        ('band_above_20khz_pct', '>20kHz'),
    ]
    print(f"\n  {'Band':<12} {'Real':>8} {'Optimized':>10} {'Error':>8}")
    print(f"  {'-'*40}")
    for key, label in band_keys_display:
        real_val = target.get(key, 0)
        opt_val = bands.get(key, 0)
        err = 0
        if real_val > 0.1:
            err = (opt_val - real_val) / real_val * 100
        print(f"  {label:<12} {real_val:>7.1f}% {opt_val:>9.1f}% {err:>+7.1f}%")

    return {
        'whale': whale_name,
        'cost': float(result.fun),
        'nit': int(result.nit),
        'nfev': int(result.nfev),
        'elapsed_s': round(elapsed, 1),
        'params': {
            'tensions': [round(t, 4) for t in tensions],
            'tau_up_s': round(tau_up, 6),
            'tau_down_s': round(tau_down, 6),
            'spermaceti_c': round(spermaceti_c, 1),
            'organ_length': round(organ_length, 4),
            'organ_diameter': round(organ_diameter, 4),
        },
        'features': {
            'spectral_centroid_hz': round(centroid, 1),
            **{k: round(bands.get(k, 0), 2) for k in [kk for kk, _ in band_keys_display]},
        },
        'signal': signal,
        'dt': dt,
        'history': tracker.history,
        'initial_cost': init_cost,
    }


# ============================================================
# COMPARISON TABLE
# ============================================================
def print_comparison_table(results):
    """Print the final comparison table."""
    print(f"\n\n{'='*80}")
    print("OPTIMIZED RESULTS - FINAL COMPARISON")
    print(f"{'='*80}\n")

    whales = ['Whale_A', 'Whale_D', 'Whale_F']
    r = {res['whale']: res for res in results}

    # Header
    header = f"{'':>14}"
    for w in whales:
        header += f"  {w:>8}                 "
    print(header)

    sub_header = f"{'':>14}"
    for w in whales:
        sub_header += f"  {'Real':>8} {'Opt':>8} {'Err':>7}"
    print(sub_header)
    print("-" * 80)

    # Centroid row
    row = f"{'Centroid (Hz)':>14}"
    for w in whales:
        real = TARGETS[w]['spectral_centroid_hz']
        opt = r[w]['features']['spectral_centroid_hz']
        err = (opt - real) / real * 100
        row += f"  {real:>8.0f} {opt:>8.0f} {err:>+6.1f}%"
    print(row)

    # Band rows
    band_display = [
        ('band_sub_100hz_pct', '<100Hz'),
        ('band_500_2khz_pct', '0.5-2kHz'),
        ('band_2_5khz_pct', '2-5kHz'),
        ('band_5_10khz_pct', '5-10kHz'),
        ('band_10_20khz_pct', '10-20kHz'),
        ('band_above_20khz_pct', '>20kHz'),
    ]
    for key, label in band_display:
        row = f"{label:>14}"
        for w in whales:
            real = TARGETS[w].get(key, 0)
            opt = r[w]['features'].get(key, 0)
            err = (opt - real) / real * 100 if real > 0.1 else 0
            row += f"  {real:>7.1f}% {opt:>7.1f}% {err:>+6.1f}%"
        print(row)

    # Cost row
    row = f"{'Cost':>14}"
    for w in whales:
        init = r[w]['initial_cost']
        final = r[w]['cost']
        improvement = (1 - final / init) * 100 if init > 0 else 0
        row += f"  {init:>8.4f} {final:>8.4f} {improvement:>+5.0f}%  "
    print(row)

    print()
    for w in whales:
        p = r[w]['params']
        print(f"{w}: tensions=[{', '.join(f'{t:.3f}' for t in p['tensions'])}]  "
              f"tau_up={p['tau_up_s']*1000:.3f}ms  tau_down={p['tau_down_s']*1000:.2f}ms  "
              f"c={p['spermaceti_c']:.0f}  L={p['organ_length']:.3f}  D={p['organ_diameter']:.3f}")


# ============================================================
# FIGURE: tension_optimized.png (20x16, 300dpi)
# ============================================================
def create_figure(results, whale_files, output_dir):
    """Create the 3-row + summary comparison figure."""
    print("\n[Figure] Creating tension_optimized.png...")

    whales = ['Whale_A', 'Whale_D', 'Whale_F']
    r = {res['whale']: res for res in results}
    colors = {'Whale_A': '#1f77b4', 'Whale_D': '#ff7f0e', 'Whale_F': '#2ca02c'}
    band_colors_bar = ['#440154', '#31688e', '#35b779', '#fde725', '#e76f51', '#d62728', '#9467bd']

    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(4, 3, figure=fig, hspace=0.40, wspace=0.30,
                  height_ratios=[1, 1, 1, 0.5])

    dt = 1.0 / TARGET_SR

    for col, whale_name in enumerate(whales):
        res = r[whale_name]
        signal = res['signal']
        whale_id = whale_name.split('_')[1]
        target = TARGETS[whale_name]
        opt_bands = res['features']
        opt_params = res['params']

        # --- Row 1: Spectrum comparison (optimized vs real) ---
        ax = fig.add_subplot(gs[0, col])

        # Synthesized spectrum
        synth_fft = np.abs(np.fft.rfft(signal))
        synth_freqs = np.fft.rfftfreq(len(signal), dt) / 1000
        synth_db = 20 * np.log10(synth_fft / max(np.max(synth_fft), 1e-30) + 1e-30)
        ax.plot(synth_freqs, synth_db, color=colors[whale_name], linewidth=1.2,
                label='Optimized', zorder=3)

        # Real click spectrum
        try:
            real_click, _ = get_mean_real_click(whale_id, whale_files)
            if real_click is not None:
                real_fft = np.abs(np.fft.rfft(real_click))
                real_freqs = np.fft.rfftfreq(len(real_click), dt) / 1000
                real_db = 20 * np.log10(real_fft / max(np.max(real_fft), 1e-30) + 1e-30)
                ax.plot(real_freqs, real_db, color='gray', linewidth=0.8, alpha=0.7,
                        label='Real', zorder=2)
        except Exception:
            pass

        centroid_err = (opt_bands['spectral_centroid_hz'] - target['spectral_centroid_hz']) / target['spectral_centroid_hz'] * 100
        ax.set_title(f'{whale_name} - Spectrum\nCentroid: {opt_bands["spectral_centroid_hz"]:.0f} Hz '
                     f'(target {target["spectral_centroid_hz"]}, err {centroid_err:+.1f}%)',
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('Frequency (kHz)')
        ax.set_ylabel('Magnitude (dB)')
        ax.set_xlim(0, 22)
        ax.set_ylim(-60, 5)
        ax.legend(fontsize=8)
        ax.axvline(target['spectral_centroid_hz']/1000, color='red', linestyle=':', alpha=0.4, linewidth=0.8)

        # --- Row 2: Band energy bars (optimized vs real) ---
        ax = fig.add_subplot(gs[1, col])

        band_keys = [
            ('band_sub_100hz_pct', '<100'),
            ('band_100_500hz_pct', '100-500'),
            ('band_500_2khz_pct', '0.5-2k'),
            ('band_2_5khz_pct', '2-5k'),
            ('band_5_10khz_pct', '5-10k'),
            ('band_10_20khz_pct', '10-20k'),
            ('band_above_20khz_pct', '>20k'),
        ]
        x = np.arange(len(band_keys))
        width = 0.35

        real_vals = [target.get(k, 0) for k, _ in band_keys]
        opt_vals = [opt_bands.get(k, 0) for k, _ in band_keys]

        bars1 = ax.bar(x - width/2, opt_vals, width, label='Optimized',
                       color=[band_colors_bar[i] for i in range(len(band_keys))], alpha=0.85)
        bars2 = ax.bar(x + width/2, real_vals, width, label='Real',
                       color='gray', alpha=0.5)

        ax.set_title(f'{whale_name} - Band Energy', fontsize=10)
        ax.set_xlabel('Frequency Band')
        ax.set_ylabel('Energy (%)')
        ax.set_xticks(x)
        ax.set_xticklabels([l for _, l in band_keys], rotation=45, fontsize=8)
        ax.legend(fontsize=8)

        # --- Row 3: Muscle tension patterns (bar chart) ---
        ax = fig.add_subplot(gs[2, col])

        tension_labels = ['16kHz', '12kHz', '9kHz', '7kHz', '5kHz', '3.5kHz']
        tensions = opt_params['tensions']
        orig_tensions = WHALE_CONFIGS[whale_name]['muscle_tensions']

        x_t = np.arange(6)
        width_t = 0.35
        ax.bar(x_t - width_t/2, tensions, width_t, label='Optimized',
               color=colors[whale_name], alpha=0.85)
        ax.bar(x_t + width_t/2, orig_tensions, width_t, label='Original',
               color='gray', alpha=0.5)

        ax.set_title(f'{whale_name} - Muscle Tensions', fontsize=10)
        ax.set_xlabel('Muscle Band')
        ax.set_ylabel('Tension (0-1)')
        ax.set_xticks(x_t)
        ax.set_xticklabels(tension_labels, fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)

        # Annotate values
        for i, (t_opt, t_orig) in enumerate(zip(tensions, orig_tensions)):
            ax.text(i - width_t/2, t_opt + 0.02, f'{t_opt:.2f}', ha='center', va='bottom', fontsize=7)

    # --- Bottom: Summary metrics table ---
    ax = fig.add_subplot(gs[3, :])
    ax.axis('off')

    table_data = []
    for whale_name in whales:
        res = r[whale_name]
        p = res['params']
        f = res['features']
        t = TARGETS[whale_name]
        centroid_err = (f['spectral_centroid_hz'] - t['spectral_centroid_hz']) / t['spectral_centroid_hz'] * 100
        band_5_10_err = (f['band_5_10khz_pct'] - t['band_5_10khz_pct']) / t['band_5_10khz_pct'] * 100
        band_2_5_err = (f['band_2_5khz_pct'] - t['band_2_5khz_pct']) / t['band_2_5khz_pct'] * 100

        table_data.append([
            whale_name,
            f"{res['cost']:.4f}",
            f"{f['spectral_centroid_hz']:.0f} ({centroid_err:+.1f}%)",
            f"{f['band_5_10khz_pct']:.1f}% ({band_5_10_err:+.1f}%)",
            f"{f['band_2_5khz_pct']:.1f}% ({band_2_5_err:+.1f}%)",
            f"[{', '.join(f'{t:.2f}' for t in p['tensions'])}]",
            f"up={p['tau_up_s']*1000:.2f} dn={p['tau_down_s']*1000:.1f}ms",
            f"c={p['spermaceti_c']:.0f} L={p['organ_length']:.2f} D={p['organ_diameter']:.2f}",
        ])

    table = ax.table(
        cellText=table_data,
        colLabels=['Whale', 'Cost', 'Centroid', '5-10kHz', '2-5kHz',
                   'Tensions', 'Envelope', 'Anatomy'],
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.5)
    for (row, col_idx), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#cccccc')
            cell.set_fontsize(9)

    fig.suptitle('Muscle Tension Optimization - Differential Evolution Results',
                 fontsize=14, fontweight='bold', y=0.98)

    out_path = output_dir / 'tension_optimized.png'
    fig.savefig(str(out_path), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("MUSCLE TENSION OPTIMIZER")
    print("Differential Evolution on Mode Synthesizer Parameters")
    print("=" * 70)
    print(f"Parameters: {len(PARAM_NAMES)}")
    for name, (lo, hi) in zip(PARAM_NAMES, PARAM_BOUNDS):
        print(f"  {name:>16}: [{lo}, {hi}]")

    # Load whale WAV files for real comparison
    print("\nLoading whale WAV file map...")
    try:
        whale_files = get_whale_wav_files()
        for wid, files in whale_files.items():
            print(f"  Whale {wid}: {len(files)} files")
    except Exception as e:
        print(f"  Warning: could not load WAV files: {e}")
        whale_files = {}

    # Optimize all 3 whales sequentially
    all_results = []
    total_start = time.time()

    for whale_name in ['Whale_A', 'Whale_D', 'Whale_F']:
        target = TARGETS[whale_name]
        config = WHALE_CONFIGS[whale_name]
        result = optimize_whale(whale_name, target, config)
        all_results.append(result)

    total_elapsed = time.time() - total_start
    print(f"\n\nTotal optimization time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")

    # Print final comparison table
    print_comparison_table(all_results)

    # Save results JSON (without signal arrays)
    json_results = []
    for res in all_results:
        json_res = {k: v for k, v in res.items() if k not in ('signal', 'dt')}
        json_results.append(json_res)

    json_path = OUTPUT_DIR / 'tension_optimized_results.json'
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\nSaved: {json_path}")

    # Save WAV files
    print("\nSaving optimized WAV files...")
    for res in all_results:
        whale_id = res['whale'].split('_')[1]
        wav_path = OUTPUT_DIR / f'whale_{whale_id}_optimized.wav'
        save_wav(res['signal'], wav_path, sample_rate=TARGET_SR)

    # Create figure
    create_figure(all_results, whale_files, OUTPUT_DIR)

    print("\n" + "=" * 70)
    print("DONE - All outputs saved to:")
    print(f"  {OUTPUT_DIR}/tension_optimizer.py")
    print(f"  {OUTPUT_DIR}/tension_optimized.png")
    print(f"  {OUTPUT_DIR}/tension_optimized_results.json")
    print(f"  {OUTPUT_DIR}/whale_A_optimized.wav")
    print(f"  {OUTPUT_DIR}/whale_D_optimized.wav")
    print(f"  {OUTPUT_DIR}/whale_F_optimized.wav")
    print("=" * 70)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Generate Panel A: Spectral Voiceprint per Whale (publication quality)."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# --- Load data ---
spec = pd.read_feather('/mnt/archive/datasets/whale_communication/ceti_vowels/clickspec.ft')
meta = pd.read_csv('/mnt/archive/datasets/whale_communication/ceti_vowels/codamd.csv')

# Frequency columns (skip codanum, clicknum)
freq_cols = [c for c in spec.columns if c not in ('codanum', 'clicknum')]
freqs = np.array([float(c) for c in freq_cols])

# Merge on codanum so each click row gets a whale label
spec['codanum'] = spec['codanum'].astype(int)
merged = spec.merge(meta[['codanum', 'whale']], on='codanum', how='inner')

# Target whales
TARGET_WHALES = ['ATWOOD', 'FORK', 'PINCHY', 'SAM', 'TBB', 'JOCASTA', 'LAIUS']

# --- Compute mean spectral envelope per whale ---
envelopes = {}
for whale in TARGET_WHALES:
    subset = merged[merged['whale'] == whale]
    if len(subset) == 0:
        print(f'WARNING: no data for {whale}')
        continue
    mean_spec = subset[freq_cols].mean(axis=0).values.astype(float)
    # Normalize to 0-1
    mn, mx = mean_spec.min(), mean_spec.max()
    if mx > mn:
        mean_spec = (mean_spec - mn) / (mx - mn)
    # Smooth
    mean_spec = gaussian_filter1d(mean_spec, sigma=3)
    envelopes[whale] = mean_spec
    print(f'{whale}: {len(subset)} clicks')

# --- Plot ---
fig, ax = plt.subplots(figsize=(14, 8), dpi=300)

colors = plt.cm.tab10.colors
for i, (whale, env) in enumerate(envelopes.items()):
    ax.plot(freqs, env, color=colors[i % len(colors)], linewidth=2, label=whale, alpha=0.9)

# X limit: 0-25 kHz
ax.set_xlim(0, 25000)
ax.set_ylim(0, 1.05)

ax.set_xlabel('Frequency (Hz)', fontsize=14)
ax.set_ylabel('Normalized Power', fontsize=14)
ax.set_title('Spectral Voiceprint per Whale', fontsize=18, fontweight='bold', pad=20)
ax.text(0.5, 1.02, 'Mean spectral envelope across all clicks - each whale has a distinct resonance signature',
        transform=ax.transAxes, ha='center', va='bottom', fontsize=12, color='#555555')

ax.tick_params(axis='both', labelsize=12)
ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
ax.set_facecolor('white')
fig.patch.set_facecolor('white')

# Format x-axis as kHz
from matplotlib.ticker import FuncFormatter
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x/1000:.0f}' if x >= 1000 else f'{x:.0f}'))
ax.set_xlabel('Frequency (kHz)', fontsize=14)

ax.legend(fontsize=12, loc='upper right', framealpha=0.9, edgecolor='#cccccc')

plt.tight_layout()
out = '/mnt/archive/datasets/whale_communication/analysis/voiceprint_spectral_A.png'
fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f'Saved: {out}')

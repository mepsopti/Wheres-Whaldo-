#!/usr/bin/env python3
"""
Generate colored click train spectral overlay - Panel D variant.
Same layout/proportions as the inferno version (voiceprint_overlay_D.png)
but using colored spectral silhouettes instead of heatmaps.

Each click appears as a vertical colored shape where the width at each
frequency represents the spectral power at that frequency.

Author: Jaak (Whale Acoustic ID Agent)
Date: 2026-03-29
"""

import pandas as pd
import numpy as np
import pyarrow.feather as feather
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print('Loading data...')

# Load CETI data
codamd = pd.read_csv('/mnt/archive/datasets/whale_communication/ceti_vowels/codamd.csv')
clickspec = feather.read_feather('/mnt/archive/datasets/whale_communication/ceti_vowels/clickspec.ft')

# Identify frequency columns and merge whale labels
freq_cols = [c for c in clickspec.columns if c not in ['codanum', 'clicknum']]
freqs = np.array([float(f) for f in freq_cols])

codamd['codanum'] = codamd['codanum'].astype(str)
clickspec['codanum'] = clickspec['codanum'].astype(str)
click_whale = clickspec.merge(codamd[['codanum', 'whale']], on='codanum', how='left')
click_whale = click_whale.dropna(subset=['whale'])

# Same 7 whales, same order as inferno version
overlay_whales = ['ATWOOD', 'FORK', 'PINCHY', 'SAM', 'TBB', 'JOCASTA', 'LAIUS']

# Frequency range: 0-25 kHz
freq_display_max = 25000
freq_mask = freqs <= freq_display_max
display_freqs = freqs[freq_mask]
display_freqs_khz = display_freqs / 1000.0

MAX_CLICKS = 8

# Distinct bright colors for each whale
whale_colors = {
    'ATWOOD':  '#E63946',   # red
    'FORK':    '#FF8C00',   # orange
    'PINCHY':  '#2EC4B6',   # teal
    'SAM':     '#3A86FF',   # blue
    'TBB':     '#8338EC',   # purple
    'JOCASTA': '#FF006E',   # hot pink
    'LAIUS':   '#06D6A0',   # green
}

# Build spectrograms per whale
print('Building spectrograms...')
whale_spectrograms = {}
for wname in overlay_whales:
    whale_clicks_df = click_whale[click_whale['whale'] == wname]
    codas = whale_clicks_df.groupby('codanum')

    click_spectra = {}
    for coda_id, coda_group in codas:
        coda_sorted = coda_group.sort_values('clicknum')
        for j, (_, click_row) in enumerate(coda_sorted.iterrows()):
            if j >= MAX_CLICKS:
                break
            if j not in click_spectra:
                click_spectra[j] = []
            spec = click_row[freq_cols].values.astype(float)
            click_spectra[j].append(spec[freq_mask])

    if not click_spectra:
        continue

    max_pos = min(max(click_spectra.keys()) + 1, MAX_CLICKS)
    spectrogram = np.zeros((len(display_freqs), max_pos))
    for j in range(max_pos):
        if j in click_spectra and click_spectra[j]:
            spectrogram[:, j] = np.mean(click_spectra[j], axis=0)

    # Normalize per whale (0-1)
    if spectrogram.max() > 0:
        spectrogram = spectrogram / spectrogram.max()

    whale_spectrograms[wname] = spectrogram
    n_codas = len(list(codas))
    print(f'  {wname}: {n_codas} codas, {max_pos} click positions')

n_whales = len(whale_spectrograms)
print(f'Building figure with {n_whales} whale columns...')

# ---- Publication figure: same size as inferno version ----
fig, axes = plt.subplots(1, n_whales, figsize=(14, 8), sharey=True)
if n_whales == 1:
    axes = [axes]

fig.patch.set_facecolor('white')

for idx, wname in enumerate(list(whale_spectrograms.keys())):
    spectrogram = whale_spectrograms[wname]
    ax = axes[idx]
    n_clicks = spectrogram.shape[1]
    color = whale_colors.get(wname, '#888888')

    ax.set_facecolor('white')

    # For each click position, draw the spectral envelope as a filled silhouette
    for j in range(n_clicks):
        col = spectrogram[:, j]
        if col.max() == 0:
            continue

        # Width of silhouette scales with power; max width = 0.45 (half a column)
        half_width = col * 0.45

        # Center of this click column
        center_x = j

        # Draw filled shape: left edge and right edge symmetric around center
        left = center_x - half_width
        right = center_x + half_width

        # Fill between left and right at each frequency
        ax.fill_betweenx(display_freqs_khz, left, right,
                         color=color, alpha=0.7, linewidth=0)

        # Add a thin outline for definition
        ax.plot(right, display_freqs_khz, color=color, alpha=0.9, linewidth=0.4)
        ax.plot(left, display_freqs_khz, color=color, alpha=0.9, linewidth=0.4)

    ax.set_xlim(-0.5, MAX_CLICKS - 0.5)
    ax.set_ylim(0, 25)
    ax.set_xlabel('Click #', fontsize=12)
    ax.set_xticks(range(min(n_clicks, MAX_CLICKS)))
    ax.set_xticklabels([str(i + 1) for i in range(min(n_clicks, MAX_CLICKS))], fontsize=10)
    ax.tick_params(axis='y', labelsize=10)

    # Whale name as column title, in the whale's color
    ax.set_title(wname, fontsize=13, fontweight='bold', pad=8, color=color)

    if idx == 0:
        ax.set_ylabel('Frequency (kHz)', fontsize=13)

    # Light grid for readability
    ax.grid(True, alpha=0.15, color='#cccccc', linewidth=0.5)

# Title and subtitle
fig.suptitle('Click Train Spectral Overlay per Whale',
             fontsize=16, fontweight='bold', y=0.98)
fig.text(0.5, 0.935,
         "Each click's spectral envelope shown as a colored silhouette - width = power at each frequency",
         ha='center', fontsize=12, fontstyle='italic', color='#444444')

plt.subplots_adjust(left=0.06, right=0.96, bottom=0.08, top=0.90, wspace=0.08)

outpath = '/mnt/archive/datasets/whale_communication/analysis/voiceprint_colored_D.png'
plt.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
print(f'\nSaved: {outpath}')
print('Done!')

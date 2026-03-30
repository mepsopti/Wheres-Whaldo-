#!/usr/bin/env python3
"""
Sperm Whale Acoustic Simulator - Real 3D Skull Geometry

Uses NHM London specimen NHMUK ZD 2007.100 (stranded 1930s Norway)
3D CT scan: 88,491 vertices, 175,076 faces, OBJ format in mm.

Extracts a sagittal plane cross-section from the real cranium,
maps it onto a 2D FDTD grid, adds soft tissue structures anchored
to the skull anatomy, and runs the acoustic simulation.

Compares results to the parameterized parabolic skull model from
sperm_whale_sim.py to quantify the effect of real skull geometry.
"""

import json
import os
import sys
import time
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter, binary_fill_holes, binary_dilation, binary_erosion
from scipy.interpolate import interp1d
import trimesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.gridspec as gridspec

# ============================================================
# PATHS
# ============================================================

SKULL_OBJ = "/mnt/archive/datasets/whale_communication/3d_scans/sperm_whale_cranium_high/source/12_2007_online.obj"
OUTPUT_DIR = "/mnt/archive/datasets/whale_communication/analysis"
OUTPUT_FIG = os.path.join(OUTPUT_DIR, "skull_sim_results.png")
OUTPUT_SCRIPT = os.path.join(OUTPUT_DIR, "skull_geometry_sim.py")

# ============================================================
# MATERIAL PROPERTIES
# ============================================================

MATERIALS = {
    "water":       {"rho": 1025.0, "c": 1530.0, "label": 0},
    "spermaceti":  {"rho": 857.0,  "c": 1370.0, "label": 1},
    "junk_post":   {"rho": 870.0,  "c": 1380.0, "label": 2},
    "junk_ant":    {"rho": 930.0,  "c": 1450.0, "label": 3},
    "connective":  {"rho": 1070.0, "c": 1570.0, "label": 4},
    "bone":        {"rho": 1900.0, "c": 3000.0, "label": 5},
    "air_sac":     {"rho": 50.0,   "c": 800.0,  "label": 6},  # fatty membrane model
    "blubber":     {"rho": 930.0,  "c": 1430.0, "label": 7},
}

TISSUE_NAMES = {v["label"]: k for k, v in MATERIALS.items()}
TISSUE_COLORS = {
    0: "#2B65EC",  # water - blue
    1: "#FFD700",  # spermaceti - gold
    2: "#FFA500",  # junk_post - orange
    3: "#FF8C00",  # junk_ant - dark orange
    4: "#8B4513",  # connective - brown
    5: "#F5F5DC",  # bone - beige/ivory
    6: "#FF1493",  # air sac - magenta
    7: "#90EE90",  # blubber - light green
}


def spermaceti_sound_speed(temp_c):
    """Sound speed in spermaceti oil as function of temperature."""
    if temp_c >= 33:
        return 1370.0
    elif temp_c >= 29:
        return 1370.0 + (33.0 - temp_c) * 12.5
    elif temp_c >= 25:
        return 1420.0 + (29.0 - temp_c) * 10.0
    else:
        return 1460.0 + (25.0 - temp_c) * 14.0


# ============================================================
# STEP 1: LOAD AND ORIENT THE SKULL
# ============================================================

def load_skull_cross_section(obj_path, sagittal_y=-600.0, plot_debug=False):
    """Load the 3D skull OBJ, slice at the sagittal plane, return 2D cross-section.

    The NHM specimen coordinate system:
      X: anterior-posterior (longest axis, ~3400mm)
      Y: left-right (asymmetric, range ~-1519 to +385)
      Z: dorsal-ventral (~1670mm)

    We slice at Y ~ -600 (near the median, good sagittal section).
    Returns points in mm as (anterior-posterior, dorsal-ventral) arrays.
    """
    print(f"Loading skull mesh from {obj_path}...")
    mesh = trimesh.load(obj_path)
    print(f"  Vertices: {mesh.vertices.shape[0]:,}, Faces: {mesh.faces.shape[0]:,}")
    print(f"  Bounds (mm): X=[{mesh.vertices[:,0].min():.0f}, {mesh.vertices[:,0].max():.0f}]"
          f" Y=[{mesh.vertices[:,1].min():.0f}, {mesh.vertices[:,1].max():.0f}]"
          f" Z=[{mesh.vertices[:,2].min():.0f}, {mesh.vertices[:,2].max():.0f}]")

    # Slice at sagittal plane (Y = sagittal_y)
    print(f"  Slicing at Y={sagittal_y}mm (sagittal plane)...")
    section = mesh.section(plane_origin=[0, sagittal_y, 0], plane_normal=[0, 1, 0])
    if section is None:
        raise ValueError(f"No cross-section found at Y={sagittal_y}")

    # Get 3D vertices of the section (these lie on the Y=sagittal_y plane)
    # Use the 3D vertices directly - X and Z coordinates
    section_verts_3d = section.vertices
    xs = section_verts_3d[:, 0]  # anterior-posterior
    zs = section_verts_3d[:, 2]  # dorsal-ventral

    print(f"  Cross-section: {len(xs)} vertices")
    print(f"  X range: {xs.min():.0f} to {xs.max():.0f} mm ({xs.max()-xs.min():.0f} mm)")
    print(f"  Z range: {zs.min():.0f} to {zs.max():.0f} mm ({zs.max()-zs.min():.0f} mm)")

    return xs, zs, mesh


# ============================================================
# STEP 2: MAP SKULL ONTO ACOUSTIC GRID
# ============================================================

def build_skull_geometry(skull_xs_mm, skull_zs_mm, dx=0.01):
    """Build a 2D FDTD grid with real skull cross-section and soft tissue.

    Grid: 6m x 3m domain, dx=1cm
    Skull coordinates converted from mm to grid indices.

    Returns: rho, c, tissue_map, source_pos, sensor_positions, grid_info
    """
    # Domain size - must be large enough for skull + organ + junk + water padding
    # Skull ~1.7m, organ ~3.2m anterior to skull, junk ~2.0m anterior to organ
    # Total head length ~5.5m anterior, plus padding = 8m needed
    Lx = 8.0  # meters, anterior-posterior
    Ly = 3.0  # meters, dorsal-ventral
    Nx = int(Lx / dx)
    Ny = int(Ly / dx)

    print(f"\nBuilding grid: {Nx} x {Ny} = {Nx*Ny:,} cells (dx={dx*100:.0f}cm)")

    # Initialize with water
    rho = np.full((Nx, Ny), MATERIALS["water"]["rho"], dtype=np.float64)
    c = np.full((Nx, Ny), MATERIALS["water"]["c"], dtype=np.float64)
    tissue_map = np.zeros((Nx, Ny), dtype=np.int32)  # 0 = water

    # Convert skull coords from mm to meters
    skull_x_m = skull_xs_mm / 1000.0
    skull_z_m = skull_zs_mm / 1000.0

    # Center the skull in the grid
    # Skull X range is roughly -1.3m to 2.1m (AP), total ~3.4m
    # Skull Z range is roughly -0.15m to 1.5m (DV), total ~1.7m
    skull_x_center = (skull_x_m.max() + skull_x_m.min()) / 2
    skull_z_center = (skull_z_m.max() + skull_z_m.min()) / 2

    # Place skull posterior end at x=7.0m, leaving room for:
    # - organ (3.2m) + junk (2.0m) = 5.2m anterior
    # - so junk tip at ~1.8m, with ~1.5m water padding in front
    # - ~0.5m water behind skull
    skull_x_offset = 7.0 - skull_x_m.max()
    skull_z_offset = Ly / 2 - skull_z_center  # center vertically

    skull_gx = ((skull_x_m + skull_x_offset) / dx).astype(int)
    skull_gz = ((skull_z_m + skull_z_offset) / dx).astype(int)

    # Clip to grid bounds
    skull_gx = np.clip(skull_gx, 0, Nx - 1)
    skull_gz = np.clip(skull_gz, 0, Ny - 1)

    # Create skull bone region by filling the cross-section outline
    # First, rasterize the cross-section polygon
    skull_mask = np.zeros((Nx, Ny), dtype=bool)

    # Sort cross-section points by angle from centroid to form outline
    cx_skull = int(np.mean(skull_gx))
    cz_skull = int(np.mean(skull_gz))

    # Instead of sorting by angle (which may not work for complex shapes),
    # use the section edges to rasterize properly.
    # Simpler approach: scatter the points and dilate to form solid bone
    for gx, gz in zip(skull_gx, skull_gz):
        if 0 <= gx < Nx and 0 <= gz < Ny:
            skull_mask[gx, gz] = True

    # Dilate to fill gaps between scattered points, then fill interior
    skull_mask = binary_dilation(skull_mask, iterations=3)
    skull_mask = binary_fill_holes(skull_mask)
    # Erode back to approximate original thickness
    skull_mask = binary_erosion(skull_mask, iterations=1)

    # Apply bone properties
    rho[skull_mask] = MATERIALS["bone"]["rho"]
    c[skull_mask] = MATERIALS["bone"]["c"]
    tissue_map[skull_mask] = MATERIALS["bone"]["label"]

    # Find skull boundaries for placing soft tissue
    skull_cols = np.where(skull_mask.any(axis=1))[0]
    skull_x_min_g = skull_cols.min()  # anterior edge of skull
    skull_x_max_g = skull_cols.max()  # posterior edge of skull

    # For each column of the skull, find dorsal and ventral edges
    skull_dorsal = np.zeros(Nx, dtype=int)
    skull_ventral = np.zeros(Nx, dtype=int)
    skull_present = np.zeros(Nx, dtype=bool)

    for ix in range(Nx):
        col = skull_mask[ix, :]
        if col.any():
            skull_present[ix] = True
            rows = np.where(col)[0]
            skull_dorsal[ix] = rows.max()  # dorsal = higher Z = higher index
            skull_ventral[ix] = rows.min()  # ventral = lower Z = lower index

    # Find the rostral basin - the concave depression in the anterior portion
    # It's the region where skull dorsal surface dips down anteriorly
    # Typically in the anterior 40% of the skull
    basin_start = skull_x_min_g
    basin_end = skull_x_min_g + int(0.5 * (skull_x_max_g - skull_x_min_g))
    basin_region = range(basin_start, basin_end)

    # Skull midline (approximate center DV of skull)
    skull_mid_z = int(np.mean([skull_dorsal[ix] for ix in range(skull_x_min_g, skull_x_max_g) if skull_present[ix]]))

    print(f"  Skull grid extent: x=[{skull_x_min_g}, {skull_x_max_g}], "
          f"mid_z={skull_mid_z}")
    print(f"  Skull length on grid: {(skull_x_max_g - skull_x_min_g) * dx:.2f}m")

    # ================================================================
    # STEP 3: ADD SOFT TISSUE STRUCTURES
    # ================================================================

    # Key dimensions (scaled to match skull)
    skull_length_g = skull_x_max_g - skull_x_min_g
    skull_length_m = skull_length_g * dx

    # Spermaceti organ length ~ 2x skull length in a real whale
    # NHM specimen skull is ~1.6m, organ would be ~3.2m
    organ_length_m = 3.2
    organ_length_g = int(organ_length_m / dx)
    organ_diameter_m = 1.4
    organ_half_g = int(organ_diameter_m / 2 / dx)

    # Junk length ~ 2.0m
    junk_length_m = 2.0
    junk_length_g = int(junk_length_m / dx)
    junk_max_diameter_m = 1.2
    junk_half_g = int(junk_max_diameter_m / 2 / dx)

    # Position everything relative to the skull
    # The spermaceti organ sits dorsal to the skull, anterior to it
    # Frontal sac sits between posterior skull and posterior spermaceti
    # The organ extends anteriorly from the skull

    # Organ posterior end = just anterior to skull posterior
    organ_end_x = skull_x_max_g - int(0.05 / dx)  # slight overlap with skull region
    organ_start_x = organ_end_x - organ_length_g

    # Junk starts at anterior end of organ and extends forward
    junk_end_x = organ_start_x
    junk_start_x = junk_end_x - junk_length_g

    # The organ sits DORSAL - above the skull dorsal surface
    # Center of organ is above the skull
    organ_center_z = skull_mid_z + int(0.4 / dx)  # 40cm above skull midline

    print(f"  Organ: x=[{organ_start_x}, {organ_end_x}] ({organ_length_m}m), "
          f"center_z={organ_center_z}")
    print(f"  Junk: x=[{junk_start_x}, {junk_end_x}] ({junk_length_m}m)")

    # --- FRONTAL AIR SAC ---
    # Thin layer between skull rostral basin and spermaceti organ
    # In reality, it spans ~0.8m x 0.8m - the largest reflector
    # Place it along the skull dorsal surface AND extending into the
    # posterior part of the organ cavity
    frontal_sac_thickness = max(int(0.03 / dx), 3)  # 3cm
    frontal_sac_width = int(0.8 / 2 / dx)  # half-width in DV

    # Primary region: along skull dorsal surface
    for ix in range(max(0, basin_start), min(Nx, organ_end_x)):
        if skull_present[ix]:
            dorsal = skull_dorsal[ix]
            for iz in range(dorsal + 1, dorsal + 1 + frontal_sac_thickness):
                if 0 <= iz < Ny:
                    rho[ix, iz] = MATERIALS["air_sac"]["rho"]
                    c[ix, iz] = MATERIALS["air_sac"]["c"]
                    tissue_map[ix, iz] = MATERIALS["air_sac"]["label"]

    # Extended region: vertical sac at the posterior end of the organ
    # This is the main reflecting surface
    frontal_sac_x = organ_end_x - int(0.05 / dx)
    for ix in range(max(0, frontal_sac_x - frontal_sac_thickness), min(Nx, frontal_sac_x)):
        for iz in range(max(0, organ_center_z - frontal_sac_width),
                        min(Ny, organ_center_z + frontal_sac_width)):
            if tissue_map[ix, iz] != MATERIALS["bone"]["label"]:
                rho[ix, iz] = MATERIALS["air_sac"]["rho"]
                c[ix, iz] = MATERIALS["air_sac"]["c"]
                tissue_map[ix, iz] = MATERIALS["air_sac"]["label"]

    # --- SPERMACETI ORGAN (case) ---
    case_wall_thickness = max(int(0.05 / dx), 2)  # 5cm connective tissue wall

    for ix in range(max(0, organ_start_x), min(Nx, organ_end_x)):
        x_frac = (ix - organ_start_x) / max(organ_end_x - organ_start_x, 1)
        # Elliptical taper
        local_radius = organ_half_g * (1 - 0.15 * (2 * x_frac - 1) ** 2)

        for iy in range(max(0, int(organ_center_z - local_radius - case_wall_thickness)),
                        min(Ny, int(organ_center_z + local_radius + case_wall_thickness))):
            dist = abs(iy - organ_center_z)

            # Skip if this is already bone
            if tissue_map[ix, iy] == MATERIALS["bone"]["label"]:
                continue
            # Skip if this is already air sac
            if tissue_map[ix, iy] == MATERIALS["air_sac"]["label"]:
                continue

            if dist <= local_radius + case_wall_thickness and dist > local_radius:
                # Case wall (connective tissue)
                rho[ix, iy] = MATERIALS["connective"]["rho"]
                c[ix, iy] = MATERIALS["connective"]["c"]
                tissue_map[ix, iy] = MATERIALS["connective"]["label"]
            elif dist <= local_radius:
                # Spermaceti oil
                rho[ix, iy] = MATERIALS["spermaceti"]["rho"]
                c[ix, iy] = MATERIALS["spermaceti"]["c"]
                tissue_map[ix, iy] = MATERIALS["spermaceti"]["label"]

    # --- DISTAL AIR SAC ---
    # At the anterior end of spermaceti organ, between case and junk
    # ~0.3m diameter disc
    distal_sac_x = organ_start_x
    distal_sac_thickness = max(int(0.03 / dx), 3)  # 3cm thick
    distal_sac_half = int(0.3 / 2 / dx)  # 0.3m diameter
    distal_center_z = organ_center_z

    for ix in range(max(0, distal_sac_x - distal_sac_thickness), min(Nx, distal_sac_x + 1)):
        for iy in range(max(0, distal_center_z - distal_sac_half),
                        min(Ny, distal_center_z + distal_sac_half)):
            if tissue_map[ix, iy] != MATERIALS["bone"]["label"]:
                rho[ix, iy] = MATERIALS["air_sac"]["rho"]
                c[ix, iy] = MATERIALS["air_sac"]["c"]
                tissue_map[ix, iy] = MATERIALS["air_sac"]["label"]

    # --- JUNK (GRIN gradient lipid lens) ---
    # Ventral and anterior, below the case
    # Center of junk is below the organ center
    junk_center_z = organ_center_z - int(0.5 / dx)  # 50cm below organ center

    for ix in range(max(0, junk_start_x), min(Nx, junk_end_x)):
        x_frac = (ix - junk_start_x) / max(junk_end_x - junk_start_x, 1)
        # Cone shape - narrow at tip (anterior), wide at base (posterior)
        local_radius = junk_half_g * (0.3 + 0.7 * x_frac)

        for iy in range(max(0, int(junk_center_z - local_radius)),
                        min(Ny, int(junk_center_z + local_radius))):
            dist = abs(iy - junk_center_z) * dx

            if tissue_map[ix, iy] != 0:  # don't overwrite existing structures
                continue

            if dist <= local_radius * dx:
                # GRIN gradient: posterior=junk_post, anterior=junk_ant
                frac = 1.0 - x_frac  # anterior fraction
                rho_val = MATERIALS["junk_post"]["rho"] + frac * (MATERIALS["junk_ant"]["rho"] - MATERIALS["junk_post"]["rho"])
                c_val = MATERIALS["junk_post"]["c"] + frac * (MATERIALS["junk_ant"]["c"] - MATERIALS["junk_post"]["c"])
                rho[ix, iy] = rho_val
                c[ix, iy] = c_val
                # Label as junk_post or junk_ant based on position
                tissue_map[ix, iy] = MATERIALS["junk_post"]["label"] if x_frac > 0.5 else MATERIALS["junk_ant"]["label"]

    # --- BLUBBER (outer layer) ---
    # Any tissue cell adjacent to water gets blubber if within a certain distance
    blubber_thickness = int(0.15 / dx)  # 15cm
    tissue_mask = tissue_map > 0  # any tissue
    blubber_zone = binary_dilation(tissue_mask, iterations=blubber_thickness) & (~tissue_mask)
    rho[blubber_zone] = MATERIALS["blubber"]["rho"]
    c[blubber_zone] = MATERIALS["blubber"]["c"]
    tissue_map[blubber_zone] = MATERIALS["blubber"]["label"]

    # --- PHONIC LIPS (source location) ---
    # At the dorsal side of the distal sac, between the two air sacs
    source_x = distal_sac_x - distal_sac_thickness
    source_y = distal_center_z + distal_sac_half + int(0.03 / dx)  # just dorsal of distal sac

    # Clamp to valid range
    source_x = max(2, min(Nx - 3, source_x))
    source_y = max(2, min(Ny - 3, source_y))

    print(f"  Phonic lips (source): ({source_x}, {source_y})")

    # --- SENSOR POSITIONS ---
    # Index 0: Forward (anterior) on-axis - 1m in front of junk tip in water
    # Index 1: Backward (posterior) on-axis - 0.5m behind skull in water
    # Index 2+: Ring of sensors for beam pattern
    sensor_positions = []

    # Forward receiver: on the organ axis, well in front of the head
    fwd_x = max(35, junk_start_x - int(1.0 / dx))
    fwd_y = organ_center_z  # on-axis
    sensor_positions.append((fwd_x, fwd_y))

    # Backward receiver: behind the skull
    bwd_x = min(Nx - 35, skull_x_max_g + int(0.5 / dx))
    bwd_y = organ_center_z
    sensor_positions.append((bwd_x, bwd_y))

    # Ring for beam pattern (36 sensors)
    n_ring = 36
    sensor_radius = min(Lx, Ly) / 2 * 0.85
    sensor_center_x = (organ_start_x + organ_end_x) // 2
    sensor_center_y = organ_center_z
    for i in range(n_ring):
        angle = 2 * np.pi * i / n_ring
        sx = int(sensor_center_x + sensor_radius * np.cos(angle) / dx)
        sy = int(sensor_center_y + sensor_radius * np.sin(angle) / dx)
        sx = max(2, min(Nx - 3, sx))
        sy = max(2, min(Ny - 3, sy))
        sensor_positions.append((sx, sy))

    print(f"  Forward receiver: ({fwd_x}, {fwd_y})")
    print(f"  Backward receiver: ({bwd_x}, {bwd_y})")

    grid_info = {
        "Nx": Nx, "Ny": Ny, "dx": dx,
        "Lx": Lx, "Ly": Ly,
        "skull_x_min_g": int(skull_x_min_g),
        "skull_x_max_g": int(skull_x_max_g),
        "organ_start_x": organ_start_x,
        "organ_end_x": organ_end_x,
        "organ_center_z": organ_center_z,
        "junk_start_x": junk_start_x,
        "junk_end_x": junk_end_x,
        "source_pos": (source_x, source_y),
    }

    return rho, c, tissue_map, (source_x, source_y), sensor_positions, grid_info


# ============================================================
# PARAMETERIZED GEOMETRY (from sperm_whale_sim.py for comparison)
# ============================================================

def build_parameterized_geometry(dx=0.01):
    """Build the original parameterized parabolic skull geometry.
    Adapted from sperm_whale_sim.py WhaleHead class, default params.
    Returns same format as build_skull_geometry.
    """
    organ_length = 3.2
    organ_diameter = 1.4
    skull_curvature = 1.0
    junk_length = 2.0
    junk_max_diameter = 1.2
    case_wall_thickness = 0.05
    frontal_sac_width = 0.8
    distal_sac_width = 0.3
    blubber_thickness = 0.15
    spermaceti_c = 1370.0

    # Match domain to skull geometry sim
    Lx = 8.0
    Ly = 3.0
    Nx = int(Lx / dx)
    Ny = int(Ly / dx)

    rho = np.full((Nx, Ny), MATERIALS["water"]["rho"], dtype=np.float64)
    c_grid = np.full((Nx, Ny), MATERIALS["water"]["c"], dtype=np.float64)
    tissue_map = np.zeros((Nx, Ny), dtype=np.int32)

    cy = Ny // 2
    head_length = organ_length + junk_length + 0.3
    pad = (Lx - head_length) / 2
    head_start_x = int(pad / dx)
    skull_x = head_start_x + int(head_length / dx)

    # Skull bone
    skull_thickness = int(0.08 / dx)
    for ix in range(skull_x - skull_thickness, skull_x):
        basin_half = int(frontal_sac_width / 2 / dx)
        for iy in range(cy - basin_half, cy + basin_half):
            dy_from_center = abs(iy - cy) * dx
            depth = (dy_from_center ** 2) / (2 * skull_curvature)
            depth_px = int(depth / dx)
            if ix >= skull_x - skull_thickness + depth_px:
                rho[ix, iy] = MATERIALS["bone"]["rho"]
                c_grid[ix, iy] = MATERIALS["bone"]["c"]
                tissue_map[ix, iy] = MATERIALS["bone"]["label"]

    # Frontal air sac
    sac_x = skull_x - skull_thickness - int(0.02 / dx)
    sac_thickness = max(int(0.03 / dx), 2)
    sac_half_y = int(frontal_sac_width / 2 / dx)
    for ix in range(sac_x - sac_thickness, sac_x):
        for iy in range(cy - sac_half_y, cy + sac_half_y):
            rho[ix, iy] = MATERIALS["air_sac"]["rho"]
            c_grid[ix, iy] = MATERIALS["air_sac"]["c"]
            tissue_map[ix, iy] = MATERIALS["air_sac"]["label"]

    # Spermaceti organ
    organ_start_x = head_start_x + int(junk_length / dx)
    organ_end_x = organ_start_x + int(organ_length / dx)
    organ_half_y = int(organ_diameter / 2 / dx)
    wall_px = max(int(case_wall_thickness / dx), 2)

    for ix in range(organ_start_x, min(organ_end_x, sac_x - sac_thickness)):
        for iy in range(cy - organ_half_y - wall_px, cy + organ_half_y + wall_px):
            dist_from_center = abs(iy - cy) * dx
            x_frac = (ix - organ_start_x) / max(organ_end_x - organ_start_x, 1)
            local_radius = organ_diameter / 2 * (1 - 0.2 * (2 * x_frac - 1) ** 2)

            if dist_from_center <= local_radius + case_wall_thickness:
                if dist_from_center > local_radius:
                    rho[ix, iy] = MATERIALS["connective"]["rho"]
                    c_grid[ix, iy] = MATERIALS["connective"]["c"]
                    tissue_map[ix, iy] = MATERIALS["connective"]["label"]
                else:
                    rho[ix, iy] = MATERIALS["spermaceti"]["rho"]
                    c_grid[ix, iy] = spermaceti_c
                    tissue_map[ix, iy] = MATERIALS["spermaceti"]["label"]

    # Distal air sac
    distal_x = organ_start_x
    distal_thickness = max(int(0.02 / dx), 2)
    distal_half_y = int(distal_sac_width / 2 / dx)
    for ix in range(distal_x - distal_thickness, distal_x):
        for iy in range(cy - distal_half_y, cy + distal_half_y):
            rho[ix, iy] = MATERIALS["air_sac"]["rho"]
            c_grid[ix, iy] = MATERIALS["air_sac"]["c"]
            tissue_map[ix, iy] = MATERIALS["air_sac"]["label"]

    # Junk
    junk_start_x = head_start_x
    junk_end_x = organ_start_x
    for ix in range(junk_start_x, junk_end_x):
        x_frac = (ix - junk_start_x) / max(junk_end_x - junk_start_x, 1)
        local_radius = junk_max_diameter / 2 * (0.3 + 0.7 * x_frac)
        for iy in range(cy - int(local_radius / dx), cy + int(local_radius / dx)):
            dist = abs(iy - cy) * dx
            if dist <= local_radius:
                rho[ix, iy] = MATERIALS["junk_ant"]["rho"] + x_frac * (MATERIALS["junk_post"]["rho"] - MATERIALS["junk_ant"]["rho"])
                c_grid[ix, iy] = MATERIALS["junk_ant"]["c"] + x_frac * (MATERIALS["junk_post"]["c"] - MATERIALS["junk_ant"]["c"])
                tissue_map[ix, iy] = MATERIALS["junk_post"]["label"] if x_frac > 0.5 else MATERIALS["junk_ant"]["label"]

    # Source
    source_x = distal_x - distal_thickness
    source_y = cy - distal_half_y - int(0.05 / dx)

    # Sensors - same layout as skull model: [0]=forward, [1]=backward, [2:]=ring
    sensor_positions = []

    # Forward on-axis
    fwd_x = max(35, junk_start_x - int(1.0 / dx))
    sensor_positions.append((fwd_x, cy))

    # Backward on-axis
    bwd_x = min(Nx - 35, skull_x + int(0.5 / dx))
    sensor_positions.append((bwd_x, cy))

    # Ring
    n_ring = 36
    sensor_radius = min(Lx, Ly) / 2 * 0.85
    sensor_center_x = (organ_start_x + organ_end_x) // 2
    for i in range(n_ring):
        angle = 2 * np.pi * i / n_ring
        sx = int(sensor_center_x + sensor_radius * np.cos(angle) / dx)
        sy = int(cy + sensor_radius * np.sin(angle) / dx)
        sx = max(2, min(Nx - 3, sx))
        sy = max(2, min(Ny - 3, sy))
        sensor_positions.append((sx, sy))

    grid_info = {
        "Nx": Nx, "Ny": Ny, "dx": dx,
        "Lx": Lx, "Ly": Ly,
        "organ_start_x": organ_start_x,
        "organ_end_x": organ_end_x,
        "source_pos": (source_x, source_y),
    }

    return rho, c_grid, tissue_map, (source_x, source_y), sensor_positions, grid_info


# ============================================================
# FDTD SIMULATOR
# ============================================================

def ricker_wavelet(freq, dt, n_samples):
    """Ricker wavelet (Mexican hat) - approximation of whale click."""
    t = np.arange(n_samples) * dt
    t0 = 1.0 / freq
    t_shifted = t - t0
    pi_f_t = (np.pi * freq * t_shifted) ** 2
    wavelet = (1 - 2 * pi_f_t) * np.exp(-pi_f_t)
    return wavelet.astype(np.float64)


def fdtd_2d(rho, c, source_pos, sensor_positions, dx, dt, n_steps, source_signal,
            capture_snapshot_step=None):
    """2D acoustic FDTD with heterogeneous media, PML boundaries.

    Same approach as sperm_whale_sim.py but returns snapshot if requested.
    """
    Nx, Ny = rho.shape

    rho = rho.astype(np.float64)
    c = c.astype(np.float64)

    # Gaussian-smooth ALL tissue boundaries (sigma=3 grid cells) for stability
    rho = gaussian_filter(rho, sigma=3.0)
    c = gaussian_filter(c, sigma=3.0)

    # Clamp to physical ranges
    rho = np.clip(rho, 30.0, 2500.0)
    c = np.clip(c, 600.0, 3500.0)

    c2 = c ** 2
    rho_x = 0.5 * (rho[1:, :] + rho[:-1, :])
    rho_y = 0.5 * (rho[:, 1:] + rho[:, :-1])
    rho_x_inv = 1.0 / rho_x
    rho_y_inv = 1.0 / rho_y

    # PML absorbing boundary
    pml_width = 30
    damping = np.zeros((Nx, Ny), dtype=np.float64)
    for i in range(pml_width):
        d = ((pml_width - i) / pml_width) ** 3 * 0.3
        damping[i, :] = np.maximum(damping[i, :], d)
        damping[Nx - 1 - i, :] = np.maximum(damping[Nx - 1 - i, :], d)
        damping[:, i] = np.maximum(damping[:, i], d)
        damping[:, Ny - 1 - i] = np.maximum(damping[:, Ny - 1 - i], d)

    decay = 1.0 - damping

    p = np.zeros((Nx, Ny), dtype=np.float64)
    vx = np.zeros((Nx - 1, Ny), dtype=np.float64)
    vy = np.zeros((Nx, Ny - 1), dtype=np.float64)

    sensor_data = np.zeros((len(sensor_positions), n_steps), dtype=np.float64)
    sx, sy = source_pos

    snapshot = None

    for t in range(n_steps):
        # Inject source (soft source)
        if t < len(source_signal):
            p[sx, sy] += float(source_signal[t])

        # Update velocity
        vx -= dt * rho_x_inv * (p[1:, :] - p[:-1, :]) / dx
        vy -= dt * rho_y_inv * (p[:, 1:] - p[:, :-1]) / dx

        # Update pressure
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

        # Capture snapshot
        if capture_snapshot_step is not None and t == capture_snapshot_step:
            snapshot = p.copy()

        if (t + 1) % 5000 == 0:
            max_p = float(np.max(np.abs(p)))
            print(f"  Step {t + 1}/{n_steps}, max |p|: {max_p:.6f}", flush=True)
            if max_p > 1e10 or np.isnan(max_p):
                print("  WARNING: simulation unstable, aborting", flush=True)
                break

    return sensor_data.astype(np.float64), p.astype(np.float64), snapshot


# ============================================================
# ANALYSIS
# ============================================================

def analyze_signal(sensor_data, sensor_positions, dt, label=""):
    """Analyze sensor data: find peaks, IPI, spectrum.

    Uses Hilbert envelope to find P0-P3 multi-path pulse arrivals,
    not individual oscillation cycles.
    """
    from scipy.signal import hilbert

    # Forward sensor (index 0 = dedicated on-axis anterior)
    forward_signal = sensor_data[0]
    # Backward sensor (index 1 = dedicated on-axis posterior)
    backward_signal = sensor_data[1]

    # Compute envelope via Hilbert transform for pulse detection
    analytic = hilbert(forward_signal)
    envelope = np.abs(analytic)
    # Smooth envelope to find pulse envelopes, not carrier oscillations
    # Use ~0.2ms window - short enough to separate P0-P3 (IPI ~4.7ms)
    # but long enough to merge individual carrier cycles (~0.08ms at 12kHz)
    smooth_samples = max(int(0.0002 / dt), 3)
    envelope_smooth = gaussian_filter(envelope, sigma=smooth_samples)

    threshold = np.max(envelope_smooth) * 0.1

    # Find peaks in the smoothed envelope - these are the P0, P1, P2, P3 arrivals
    # Require minimum separation of 1ms between peaks (to avoid double-counting)
    min_sep = int(0.001 / dt)
    peaks = []
    for i in range(1, len(envelope_smooth) - 1):
        if (envelope_smooth[i] > threshold and
            envelope_smooth[i] > envelope_smooth[i - 1] and
            envelope_smooth[i] >= envelope_smooth[i + 1]):
            if not peaks or (i - peaks[-1]) >= min_sep:
                peaks.append(i)

    peak_times_ms = [p * dt * 1000 for p in peaks]
    ipis = []
    if len(peak_times_ms) >= 2:
        ipis = [peak_times_ms[i + 1] - peak_times_ms[i] for i in range(len(peak_times_ms) - 1)]

    # Spectrum
    fft_result = np.abs(np.fft.rfft(forward_signal))
    freqs = np.fft.rfftfreq(len(forward_signal), dt)

    # Peak frequency (above 500 Hz)
    mask = freqs > 500
    peak_freq = 0
    if np.any(mask) and np.any(fft_result[mask] > 0):
        peak_freq = float(freqs[mask][np.argmax(fft_result[mask])])

    # Spectral centroid
    total_energy = np.sum(fft_result ** 2)
    spectral_centroid = float(np.sum(freqs * fft_result ** 2) / total_energy) if total_energy > 0 else 0

    # Front/back ratio
    forward_amp = np.max(np.abs(forward_signal))
    backward_amp = np.max(np.abs(backward_signal))
    fb_ratio_db = 20 * np.log10(forward_amp / max(backward_amp, 1e-30))

    # Beam pattern (ring sensors start at index 2)
    beam_amps = []
    for si in range(2, len(sensor_positions)):
        beam_amps.append(float(np.max(np.abs(sensor_data[si]))))

    results = {
        "label": label,
        "n_pulses": len(peaks),
        "peak_times_ms": peak_times_ms,
        "ipis_ms": ipis,
        "mean_ipi_ms": float(np.mean(ipis)) if ipis else 0,
        "peak_frequency_hz": peak_freq,
        "spectral_centroid_hz": spectral_centroid,
        "front_back_ratio_db": fb_ratio_db,
        "beam_pattern": beam_amps,
        "forward_signal": forward_signal,
        "backward_signal": backward_signal,
        "envelope": envelope_smooth,
        "fft_result": fft_result,
        "fft_freqs": freqs,
    }

    return results


# ============================================================
# VISUALIZATION
# ============================================================

def create_figure(skull_tissue_map, skull_results, param_results,
                  skull_snapshot, skull_grid_info, dt):
    """Create 4-panel comparison figure."""
    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(2, 2, hspace=0.3, wspace=0.25)

    # --- Panel A: Skull cross-section with tissue types ---
    ax_a = fig.add_subplot(gs[0, 0])

    # Create custom colormap for tissue types
    n_tissues = max(TISSUE_COLORS.keys()) + 1
    colors = ["#2B65EC"] * n_tissues  # default water
    for label, color in TISSUE_COLORS.items():
        colors[label] = color
    cmap = ListedColormap(colors)

    # Transpose for display (x=horizontal, y=vertical)
    im_a = ax_a.imshow(skull_tissue_map.T, origin="lower", cmap=cmap,
                        vmin=-0.5, vmax=n_tissues - 0.5, aspect="auto",
                        extent=[0, skull_grid_info["Lx"], 0, skull_grid_info["Ly"]])

    ax_a.set_xlabel("Anterior-Posterior (m)")
    ax_a.set_ylabel("Dorsal-Ventral (m)")
    ax_a.set_title("A: Skull-Derived Tissue Map (Sagittal Section)")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=TISSUE_COLORS[v["label"]], label=k)
                       for k, v in MATERIALS.items()]
    ax_a.legend(handles=legend_elements, loc="upper left", fontsize=7,
                ncol=2, framealpha=0.8)

    # --- Panel B: Acoustic field snapshot ---
    ax_b = fig.add_subplot(gs[0, 1])
    if skull_snapshot is not None:
        vmax = np.percentile(np.abs(skull_snapshot), 99.5)
        im_b = ax_b.imshow(skull_snapshot.T, origin="lower", cmap="RdBu_r",
                            vmin=-vmax, vmax=vmax, aspect="auto",
                            extent=[0, skull_grid_info["Lx"], 0, skull_grid_info["Ly"]])
        plt.colorbar(im_b, ax=ax_b, label="Pressure (Pa)", shrink=0.8)
    ax_b.set_xlabel("Anterior-Posterior (m)")
    ax_b.set_ylabel("Dorsal-Ventral (m)")
    ax_b.set_title("B: Acoustic Field Snapshot (Skull Model)")

    # --- Panel C: Forward receiver comparison ---
    ax_c = fig.add_subplot(gs[1, 0])
    t_skull = np.arange(len(skull_results["forward_signal"])) * dt * 1000
    t_param = np.arange(len(param_results["forward_signal"])) * dt * 1000

    # Normalize for comparison
    skull_fwd = skull_results["forward_signal"]
    param_fwd = param_results["forward_signal"]
    skull_norm = skull_fwd / max(np.max(np.abs(skull_fwd)), 1e-30)
    param_norm = param_fwd / max(np.max(np.abs(param_fwd)), 1e-30)

    ax_c.plot(t_skull, skull_norm, "b-", alpha=0.6, linewidth=0.5, label="Skull-derived")
    ax_c.plot(t_param, param_norm, "r-", alpha=0.4, linewidth=0.5, label="Parameterized")

    # Overlay envelopes
    skull_env = skull_results["envelope"]
    param_env = param_results["envelope"]
    skull_env_norm = skull_env / max(np.max(skull_env), 1e-30)
    param_env_norm = param_env / max(np.max(param_env), 1e-30)
    ax_c.plot(t_skull, skull_env_norm, "b-", alpha=0.9, linewidth=1.5, label="Skull envelope")
    ax_c.plot(t_param, param_env_norm, "r-", alpha=0.7, linewidth=1.5, label="Param envelope")
    ax_c.set_xlabel("Time (ms)")
    ax_c.set_ylabel("Normalized Pressure")
    ax_c.set_title("C: Forward Receiver - Time Series Comparison")
    ax_c.legend(fontsize=10)
    ax_c.set_xlim(0, 12)
    ax_c.grid(True, alpha=0.3)

    # Mark detected peaks
    for pt in skull_results["peak_times_ms"][:5]:
        ax_c.axvline(pt, color="blue", alpha=0.3, linestyle="--", linewidth=0.5)
    for pt in param_results["peak_times_ms"][:5]:
        ax_c.axvline(pt, color="red", alpha=0.3, linestyle=":", linewidth=0.5)

    # --- Panel D: Spectral comparison ---
    ax_d = fig.add_subplot(gs[1, 1])
    skull_fft = skull_results["fft_result"]
    param_fft = param_results["fft_result"]
    skull_freqs = skull_results["fft_freqs"]
    param_freqs = param_results["fft_freqs"]

    # Normalize
    skull_fft_norm = skull_fft / max(np.max(skull_fft), 1e-30)
    param_fft_norm = param_fft / max(np.max(param_fft), 1e-30)

    ax_d.semilogy(skull_freqs / 1000, skull_fft_norm, "b-", alpha=0.8,
                   linewidth=0.8, label="Skull-derived")
    ax_d.semilogy(param_freqs / 1000, param_fft_norm, "r-", alpha=0.6,
                   linewidth=0.8, label="Parameterized")
    ax_d.set_xlabel("Frequency (kHz)")
    ax_d.set_ylabel("Normalized Amplitude (log)")
    ax_d.set_title("D: Spectral Comparison (FFT)")
    ax_d.legend(fontsize=10)
    ax_d.set_xlim(0, 40)
    ax_d.grid(True, alpha=0.3)

    # Mark peak frequencies
    ax_d.axvline(skull_results["peak_frequency_hz"] / 1000, color="blue",
                  alpha=0.4, linestyle="--", linewidth=1)
    ax_d.axvline(param_results["peak_frequency_hz"] / 1000, color="red",
                  alpha=0.4, linestyle=":", linewidth=1)

    fig.suptitle("Sperm Whale Acoustic Simulation: Real Skull vs Parameterized Geometry\n"
                 "NHM London Specimen NHMUK ZD 2007.100 - Sagittal Plane FDTD",
                 fontsize=14, fontweight="bold")

    return fig


# ============================================================
# MAIN
# ============================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    dx = 0.01  # 1cm grid spacing
    center_freq = 12000  # 12kHz Ricker wavelet
    duration_ms = 12

    print("=" * 70)
    print("SPERM WHALE ACOUSTIC SIMULATOR - REAL 3D SKULL GEOMETRY")
    print("NHM London Specimen NHMUK ZD 2007.100")
    print("=" * 70)

    # ---- Step 1: Load skull ----
    print("\n--- STEP 1: Load skull cross-section ---")
    skull_xs, skull_zs, mesh = load_skull_cross_section(SKULL_OBJ, sagittal_y=-600.0)

    # ---- Step 2-3: Build skull-derived geometry ----
    print("\n--- STEP 2-3: Build skull-derived geometry ---")
    skull_rho, skull_c, skull_tissue, skull_src, skull_sensors, skull_info = \
        build_skull_geometry(skull_xs, skull_zs, dx=dx)

    # ---- Build parameterized geometry ----
    print("\n--- Building parameterized geometry (comparison) ---")
    param_rho, param_c, param_tissue, param_src, param_sensors, param_info = \
        build_parameterized_geometry(dx=dx)

    # ---- Step 4: Run simulations ----
    # Time stepping
    c_max_skull = np.max(gaussian_filter(skull_c, sigma=3.0).clip(600, 3500))
    c_max_param = np.max(gaussian_filter(param_c, sigma=3.0).clip(600, 3500))
    c_max = max(c_max_skull, c_max_param)
    dt = 0.2 * dx / c_max  # CFL condition
    n_steps = int(duration_ms / 1000.0 / dt)

    print(f"\ndt: {dt * 1e6:.2f} us, steps: {n_steps}, duration: {duration_ms}ms")
    print(f"CFL c_max: {c_max:.0f} m/s")

    # Source signal
    source_signal = ricker_wavelet(center_freq, dt, int(0.001 / dt))
    source_signal *= 1000

    # Estimate snapshot time: P1 arrives at forward sensor ~2-4ms after source
    # Expected IPI ~ 2 * 3.2 / 1370 = 4.67ms
    # P1 should be around 3-5ms
    snapshot_step = int(0.004 / dt)  # 4ms

    print("\n--- STEP 4a: Running SKULL-DERIVED simulation ---")
    t0 = time.time()
    skull_sensor_data, skull_final_p, skull_snapshot = fdtd_2d(
        skull_rho, skull_c, skull_src, skull_sensors, dx, dt, n_steps,
        source_signal, capture_snapshot_step=snapshot_step
    )
    t_skull = time.time() - t0
    print(f"  Skull sim completed in {t_skull:.1f}s")

    print("\n--- STEP 4b: Running PARAMETERIZED simulation ---")
    t0 = time.time()
    param_sensor_data, param_final_p, param_snapshot = fdtd_2d(
        param_rho, param_c, param_src, param_sensors, dx, dt, n_steps,
        source_signal, capture_snapshot_step=snapshot_step
    )
    t_param = time.time() - t0
    print(f"  Parameterized sim completed in {t_param:.1f}s")

    # ---- Step 5: Analyze and compare ----
    print("\n--- STEP 5: Analysis ---")
    skull_results = analyze_signal(skull_sensor_data, skull_sensors, dt, "Skull-derived")
    param_results = analyze_signal(param_sensor_data, param_sensors, dt, "Parameterized")

    expected_ipi_ms = 2 * 3.2 / 1370.0 * 1000  # 4.67ms

    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)
    print(f"{'Metric':<30s} {'Skull-Derived':>15s} {'Parameterized':>15s}")
    print("-" * 60)
    print(f"{'Expected IPI (ms)':<30s} {expected_ipi_ms:>15.2f} {expected_ipi_ms:>15.2f}")
    print(f"{'Measured mean IPI (ms)':<30s} {skull_results['mean_ipi_ms']:>15.2f} {param_results['mean_ipi_ms']:>15.2f}")
    print(f"{'Pulses detected':<30s} {skull_results['n_pulses']:>15d} {param_results['n_pulses']:>15d}")
    print(f"{'Peak frequency (Hz)':<30s} {skull_results['peak_frequency_hz']:>15.0f} {param_results['peak_frequency_hz']:>15.0f}")
    print(f"{'Spectral centroid (Hz)':<30s} {skull_results['spectral_centroid_hz']:>15.0f} {param_results['spectral_centroid_hz']:>15.0f}")
    print(f"{'Front/back ratio (dB)':<30s} {skull_results['front_back_ratio_db']:>15.1f} {param_results['front_back_ratio_db']:>15.1f}")

    print(f"\n{'Peak times (ms):'}")
    print(f"  Skull:  {[f'{t:.2f}' for t in skull_results['peak_times_ms'][:6]]}")
    print(f"  Param:  {[f'{t:.2f}' for t in param_results['peak_times_ms'][:6]]}")

    if skull_results['ipis_ms']:
        print(f"\n{'IPIs (ms):'}")
        print(f"  Skull:  {[f'{t:.2f}' for t in skull_results['ipis_ms'][:5]]}")
    if param_results['ipis_ms']:
        print(f"  Param:  {[f'{t:.2f}' for t in param_results['ipis_ms'][:5]]}")

    # IPI difference
    if skull_results['mean_ipi_ms'] > 0 and param_results['mean_ipi_ms'] > 0:
        ipi_diff = skull_results['mean_ipi_ms'] - param_results['mean_ipi_ms']
        print(f"\n  IPI difference (skull - param): {ipi_diff:.3f} ms")
        print(f"  IPI difference as % of expected: {abs(ipi_diff)/expected_ipi_ms*100:.1f}%")

    # Beam pattern comparison
    skull_beam = skull_results['beam_pattern']
    param_beam = param_results['beam_pattern']
    if skull_beam and param_beam:
        skull_directivity = max(skull_beam) / max(np.mean(skull_beam), 1e-30)
        param_directivity = max(param_beam) / max(np.mean(param_beam), 1e-30)
        print(f"\n  Directivity index (max/mean):")
        print(f"    Skull:  {skull_directivity:.2f}")
        print(f"    Param:  {param_directivity:.2f}")

    # ---- Step 6: Visualization ----
    print("\n--- STEP 6: Creating figure ---")
    fig = create_figure(skull_tissue, skull_results, param_results,
                        skull_snapshot, skull_info, dt)

    fig.savefig(OUTPUT_FIG, dpi=300, bbox_inches="tight")
    print(f"  Figure saved: {OUTPUT_FIG}")
    plt.close(fig)

    # Print tissue composition stats
    print("\n--- Tissue Composition (skull-derived model) ---")
    total_cells = skull_tissue.size
    for name, props in MATERIALS.items():
        count = np.sum(skull_tissue == props["label"])
        pct = count / total_cells * 100
        if count > 0:
            print(f"  {name:>15s}: {count:>8,} cells ({pct:.2f}%)")

    print(f"\nSimulation wall time: skull={t_skull:.1f}s, param={t_param:.1f}s")
    print(f"\nDone. This is a significant step - real anatomy driving the acoustic model.")


if __name__ == "__main__":
    main()

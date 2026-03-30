#!/usr/bin/env python3
"""
3D FDTD Acoustic Simulation of Sperm Whale Head
Uses real skull CT/OBJ scan voxelized onto a 3D grid.

Based on the 2D FDTD approach from sperm_whale_sim.py, extended to 3D
with real skull geometry from /mnt/archive/datasets/whale_communication/3d_scans/
"""

import time
import numpy as np
import trimesh
from scipy.ndimage import gaussian_filter, binary_dilation, binary_erosion
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ============================================================
# MATERIAL PROPERTIES
# ============================================================
MATERIALS = {
    "seawater":    {"rho": 1025.0, "c": 1530.0},
    "spermaceti":  {"rho": 857.0,  "c": 1370.0},
    "junk_post":   {"rho": 870.0,  "c": 1380.0},
    "junk_ant":    {"rho": 930.0,  "c": 1430.0},
    "connective":  {"rho": 1070.0, "c": 1570.0},
    "bone":        {"rho": 1900.0, "c": 3000.0},
    "air_sac":     {"rho": 50.0,   "c": 800.0},   # fatty membrane model
}

OUTPUT_DIR = "/mnt/archive/datasets/whale_communication/analysis"

# ============================================================
# STEP 1: Load and process skull OBJ
# ============================================================
def load_skull():
    """Load OBJ, convert mm->m, orient with longest axis=X."""
    print("=" * 60)
    print("STEP 1: Loading skull mesh")
    print("=" * 60)
    
    obj_path = "/mnt/archive/datasets/whale_communication/3d_scans/sperm_whale_cranium_high/source/12_2007_online.obj"
    mesh = trimesh.load(obj_path)
    print(f"  Loaded: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    # Convert mm to meters
    mesh.vertices *= 0.001
    bounds = mesh.bounds
    dims = bounds[1] - bounds[0]
    print(f"  Dimensions (m): X={dims[0]:.3f}, Y={dims[1]:.3f}, Z={dims[2]:.3f}")
    
    # Identify axes: longest = anterior-posterior (X)
    # The OBJ X-range is 3.4m (longest), Y is 1.9m, Z is 1.67m
    # X is already the longest axis (anterior-posterior)
    # Y appears to be dorsal-ventral (range 1.9m)
    # Z is left-right (range 1.67m)
    # This is correct orientation for a skull
    
    # Center the mesh at origin
    centroid = mesh.centroid.copy()
    mesh.vertices -= centroid
    bounds = mesh.bounds
    dims = bounds[1] - bounds[0]
    
    print(f"  Centered. New bounds:")
    print(f"    X: {bounds[0][0]:.3f} to {bounds[1][0]:.3f} (A-P, {dims[0]:.3f}m)")
    print(f"    Y: {bounds[0][1]:.3f} to {bounds[1][1]:.3f} (D-V, {dims[1]:.3f}m)")
    print(f"    Z: {bounds[0][2]:.3f} to {bounds[1][2]:.3f} (L-R, {dims[2]:.3f}m)")
    
    return mesh, dims

# ============================================================
# STEP 2 & 3: Create 3D grid and voxelize
# ============================================================
def build_3d_domain(mesh, dx=0.01):
    """Create 3D grid, voxelize skull, assign tissue properties."""
    print("\n" + "=" * 60)
    print(f"STEP 2: Creating 3D grid (dx={dx*100:.1f}cm)")
    print("=" * 60)
    
    bounds = mesh.bounds
    dims = bounds[1] - bounds[0]
    
    # Domain: skull + 0.5m buffer on each side
    buffer = 0.5
    domain_min = bounds[0] - buffer
    domain_max = bounds[1] + buffer
    domain_size = domain_max - domain_min
    
    Nx = int(np.ceil(domain_size[0] / dx))
    Ny = int(np.ceil(domain_size[1] / dx))
    Nz = int(np.ceil(domain_size[2] / dx))
    
    total_points = Nx * Ny * Nz
    # Fields: p, vx, vy, vz, rho_grid, c_grid = 6 arrays
    mem_gb = total_points * 6 * 8 / 1e9
    
    print(f"  Domain: {domain_size[0]:.2f} x {domain_size[1]:.2f} x {domain_size[2]:.2f} m")
    print(f"  Grid: {Nx} x {Ny} x {Nz} = {total_points:,} points")
    print(f"  Estimated memory: {mem_gb:.2f} GB")
    
    if mem_gb > 40:
        print(f"  WARNING: Memory {mem_gb:.1f}GB exceeds 40GB limit!")
        # Increase dx
        new_dx = dx * (mem_gb / 40) ** (1/3)
        print(f"  Increasing dx to {new_dx*100:.1f}cm")
        return build_3d_domain(mesh, dx=new_dx)
    
    # CFL condition: dt < dx / (sqrt(3) * c_max)
    c_max = MATERIALS["bone"]["c"]  # 3000 m/s
    dt = 0.5 * dx / (np.sqrt(3) * c_max)  # safety factor 0.5
    print(f"  CFL dt: {dt*1e6:.2f} us (safety factor 0.5)")
    
    # Wavelength resolution check
    freq = 12000  # Hz
    lambda_min = MATERIALS["air_sac"]["c"] / freq  # shortest wavelength
    ppw = lambda_min / dx
    print(f"  Points per wavelength (at 12kHz in air_sac): {ppw:.1f}")
    if ppw < 6:
        print(f"  WARNING: Under-resolved. Need at least 6 ppw.")
    
    print("\n" + "=" * 60)
    print("STEP 3: Voxelizing skull and building tissue map")
    print("=" * 60)
    
    # Initialize with seawater
    rho = np.full((Nx, Ny, Nz), MATERIALS["seawater"]["rho"], dtype=np.float64)
    c = np.full((Nx, Ny, Nz), MATERIALS["seawater"]["c"], dtype=np.float64)
    
    # Voxelize skull mesh
    print("  Voxelizing skull mesh...")
    t0 = time.time()
    
    # Use trimesh voxelization
    # Create a voxel grid aligned with our domain
    pitch = dx
    voxelized = mesh.voxelized(pitch=pitch)
    vox_filled = voxelized.fill()  # fill interior
    
    # Get the voxel indices and transform to our grid
    vox_points = vox_filled.points  # centers of filled voxels in mesh coords
    print(f"  Voxelized in {time.time()-t0:.1f}s: {len(vox_points)} filled voxels")
    
    # Convert mesh coordinates to grid indices
    grid_indices_x = ((vox_points[:, 0] - domain_min[0]) / dx).astype(int)
    grid_indices_y = ((vox_points[:, 1] - domain_min[1]) / dx).astype(int)
    grid_indices_z = ((vox_points[:, 2] - domain_min[2]) / dx).astype(int)
    
    # Clamp to grid bounds
    valid = ((grid_indices_x >= 0) & (grid_indices_x < Nx) &
             (grid_indices_y >= 0) & (grid_indices_y < Ny) &
             (grid_indices_z >= 0) & (grid_indices_z < Nz))
    gx = grid_indices_x[valid]
    gy = grid_indices_y[valid]
    gz = grid_indices_z[valid]
    
    # Create bone mask
    bone_mask = np.zeros((Nx, Ny, Nz), dtype=bool)
    bone_mask[gx, gy, gz] = True
    n_bone = np.sum(bone_mask)
    print(f"  Bone voxels: {n_bone:,}")
    
    # Assign bone properties
    rho[bone_mask] = MATERIALS["bone"]["rho"]
    c[bone_mask] = MATERIALS["bone"]["c"]
    
    # --- Determine skull anatomical landmarks ---
    # Skull center in grid coords
    skull_center_grid = np.array([
        (0 - domain_min[0]) / dx,  # mesh was centered at 0
        (0 - domain_min[1]) / dx,
        (0 - domain_min[2]) / dx
    ]).astype(int)
    
    # Skull extents in grid
    skull_xmin = int((bounds[0][0] - domain_min[0]) / dx)
    skull_xmax = int((bounds[1][0] - domain_min[0]) / dx)
    skull_ymin = int((bounds[0][1] - domain_min[1]) / dx)
    skull_ymax = int((bounds[1][1] - domain_min[1]) / dx)
    skull_zmin = int((bounds[0][2] - domain_min[2]) / dx)
    skull_zmax = int((bounds[1][2] - domain_min[2]) / dx)
    
    skull_mid_x = (skull_xmin + skull_xmax) // 2
    skull_mid_y = (skull_ymin + skull_ymax) // 2
    skull_mid_z = (skull_zmin + skull_zmax) // 2
    
    print(f"  Skull grid extent: X[{skull_xmin}:{skull_xmax}], Y[{skull_ymin}:{skull_ymax}], Z[{skull_zmin}:{skull_zmax}]")
    
    # --- SOFT TISSUE PLACEMENT ---
    # The sperm whale head extends ~5m anterior to the skull
    # We'll place tissues relative to the skull
    
    # Head envelope: elongated ellipsoid anterior to skull
    head_length_m = 5.5  # total head length in meters
    head_height_m = 2.0  # dorsal-ventral
    head_width_m = 1.8   # left-right
    
    # Head tip is anterior (negative X in mesh coords since skull is posterior)
    head_tip_x = skull_xmin - int(head_length_m / dx)  # far anterior
    head_tip_x = max(int(buffer * 0.3 / dx), head_tip_x)  # don't exceed domain
    
    # Build head envelope
    print("  Building soft tissue envelope...")
    tissue_mask = np.zeros((Nx, Ny, Nz), dtype=np.int8)
    # 0=water, 1=spermaceti, 2=junk, 3=frontal_air_sac, 4=distal_air_sac, 5=connective
    
    # Create coordinate arrays for vectorized operations
    xx = np.arange(Nx)
    yy = np.arange(Ny)
    zz = np.arange(Nz)
    X, Y, Z = np.meshgrid(xx, yy, zz, indexing='ij')
    
    # Head envelope (tapered ellipsoid from skull forward)
    # Anterior-posterior fraction: 0 at tip, 1 at skull
    x_frac = np.clip((X - head_tip_x) / max(skull_xmin - head_tip_x, 1), 0, 1)
    
    # Taper: narrow at anterior, full at posterior
    taper = 0.2 + 0.8 * x_frac  # 20% at tip, 100% at skull
    
    # Elliptical cross-section
    dy_norm = (Y - skull_mid_y) / (head_height_m / 2 / dx) / taper
    dz_norm = (Z - skull_mid_z) / (head_width_m / 2 / dx) / taper
    head_envelope = (x_frac > 0) & (x_frac <= 1) & ((dy_norm**2 + dz_norm**2) < 1)
    head_envelope = head_envelope & ~bone_mask
    
    # Dorsal/ventral split for spermaceti vs junk
    # Spermaceti organ: dorsal (Y > midline in our coords - depends on orientation)
    # In the OBJ, Y min=-1.52m, Y max=0.38m, so dorsal is likely positive Y
    # Spermaceti is dorsal and posterior
    dorsal = Y > skull_mid_y
    posterior_half = X > (head_tip_x + skull_xmin) // 2
    
    # Spermaceti organ: dorsal, posterior half of head
    spermaceti_region = head_envelope & dorsal & posterior_half
    
    # Junk: ventral and anterior
    junk_region = head_envelope & ~spermaceti_region & ~bone_mask
    
    # Assign spermaceti
    rho[spermaceti_region] = MATERIALS["spermaceti"]["rho"]
    c[spermaceti_region] = MATERIALS["spermaceti"]["c"]
    tissue_mask[spermaceti_region] = 1
    
    # Junk with gradient (anterior=junk_ant, posterior=junk_post)
    junk_x_frac = x_frac[junk_region]
    rho[junk_region] = (MATERIALS["junk_ant"]["rho"] * (1 - junk_x_frac) + 
                        MATERIALS["junk_post"]["rho"] * junk_x_frac)
    c[junk_region] = (MATERIALS["junk_ant"]["c"] * (1 - junk_x_frac) + 
                      MATERIALS["junk_post"]["c"] * junk_x_frac)
    tissue_mask[junk_region] = 2
    
    # Frontal air sac: thin layer on the concave rostral basin of the skull
    # This is at the anterior face of the skull
    frontal_sac_thickness = max(int(0.03 / dx), 2)  # 3cm
    frontal_sac_region = np.zeros_like(bone_mask)
    # Find anterior face of skull: bone voxels with no bone anterior to them
    bone_shifted = np.roll(bone_mask, 1, axis=0)
    bone_shifted[0, :, :] = False
    anterior_face = bone_mask & ~bone_shifted
    # Dilate anteriorly
    for i in range(frontal_sac_thickness):
        anterior_face_new = np.roll(anterior_face, -1, axis=0)
        anterior_face_new[-1, :, :] = False
        frontal_sac_region |= anterior_face_new
        anterior_face = anterior_face_new
    frontal_sac_region = frontal_sac_region & ~bone_mask
    
    rho[frontal_sac_region] = MATERIALS["air_sac"]["rho"]
    c[frontal_sac_region] = MATERIALS["air_sac"]["c"]
    tissue_mask[frontal_sac_region] = 3
    n_frontal = np.sum(frontal_sac_region)
    print(f"  Frontal air sac voxels: {n_frontal:,}")
    
    # Distal air sac: at anterior tip of spermaceti organ
    # Place a thin sac where spermaceti meets junk
    distal_x_center = (head_tip_x + skull_xmin) // 2
    distal_sac_thickness = max(int(0.02 / dx), 2)
    distal_sac_region = (
        (np.abs(X - distal_x_center) < distal_sac_thickness) &
        dorsal &
        head_envelope &
        ~bone_mask &
        ~frontal_sac_region
    )
    rho[distal_sac_region] = MATERIALS["air_sac"]["rho"]
    c[distal_sac_region] = MATERIALS["air_sac"]["c"]
    tissue_mask[distal_sac_region] = 4
    n_distal = np.sum(distal_sac_region)
    print(f"  Distal air sac voxels: {n_distal:,}")
    
    # Connective tissue: thin layer around spermaceti organ (case wall)
    case_thickness = max(int(0.05 / dx), 2)
    sperm_dilated = binary_dilation(spermaceti_region | (tissue_mask == 3), iterations=case_thickness)
    connective_region = sperm_dilated & ~spermaceti_region & ~bone_mask & ~frontal_sac_region & ~distal_sac_region & head_envelope
    rho[connective_region] = MATERIALS["connective"]["rho"]
    c[connective_region] = MATERIALS["connective"]["c"]
    tissue_mask[connective_region] = 5
    
    # Report tissue volumes
    print(f"\n  Tissue volumes:")
    print(f"    Bone:        {n_bone:>10,} voxels ({n_bone * dx**3:.4f} m^3)")
    n_sperm = np.sum(tissue_mask == 1)
    n_junk = np.sum(tissue_mask == 2)
    n_conn = np.sum(connective_region)
    print(f"    Spermaceti:  {n_sperm:>10,} voxels ({n_sperm * dx**3:.4f} m^3)")
    print(f"    Junk:        {n_junk:>10,} voxels ({n_junk * dx**3:.4f} m^3)")
    print(f"    Connective:  {n_conn:>10,} voxels ({n_conn * dx**3:.4f} m^3)")
    print(f"    Frontal sac: {n_frontal:>10,} voxels")
    print(f"    Distal sac:  {n_distal:>10,} voxels")
    
    # Gaussian smooth boundaries for stability + biological realism
    print("  Smoothing tissue boundaries (sigma=2 cells)...")
    rho = gaussian_filter(rho, sigma=2.0)
    c = gaussian_filter(c, sigma=2.0)
    # Clamp after smoothing
    rho = np.clip(rho, 40.0, 2500.0)
    c = np.clip(c, 700.0, 3500.0)
    
    # --- Source and receiver positions ---
    # Phonic lips: dorsal side, at the distal air sac, near anterior of spermaceti
    source_pos = (
        distal_x_center + distal_sac_thickness + 2,  # just posterior of distal sac
        skull_mid_y + int(0.3 / dx),  # dorsal
        skull_mid_z  # midline
    )
    # Clamp source to domain
    source_pos = tuple(max(5, min(s, n-6)) for s, n in zip(source_pos, (Nx, Ny, Nz)))
    
    # 6 receivers: forward, backward, up, down, left, right - 1m from head center
    recv_dist = int(1.0 / dx)
    head_center = (skull_mid_x - int(1.5/dx), skull_mid_y, skull_mid_z)  # approximate head center
    
    receivers = {
        "forward":  (max(5, head_center[0] - recv_dist), head_center[1], head_center[2]),
        "backward": (min(Nx-6, head_center[0] + recv_dist), head_center[1], head_center[2]),
        "up":       (head_center[0], min(Ny-6, head_center[1] + recv_dist), head_center[2]),
        "down":     (head_center[0], max(5, head_center[1] - recv_dist), head_center[2]),
        "left":     (head_center[0], head_center[1], max(5, head_center[2] - recv_dist)),
        "right":    (head_center[0], head_center[1], min(Nz-6, head_center[2] + recv_dist)),
    }
    
    # Ensure all receivers are in water (not inside head)
    for name, pos in receivers.items():
        pos = tuple(max(5, min(int(p), n-6)) for p, n in zip(pos, (Nx, Ny, Nz)))
        receivers[name] = pos
    
    grid_info = {
        "Nx": Nx, "Ny": Ny, "Nz": Nz, "dx": dx, "dt": dt,
        "domain_min": domain_min.tolist(),
        "domain_max": domain_max.tolist(),
        "skull_mid_x": skull_mid_x,
        "skull_mid_y": skull_mid_y,
        "skull_mid_z": skull_mid_z,
        "head_tip_x": head_tip_x,
    }
    
    print(f"\n  Source position (grid): {source_pos}")
    for name, pos in receivers.items():
        print(f"  Receiver '{name}': {pos}")
    
    return rho, c, tissue_mask, bone_mask, source_pos, receivers, grid_info


# ============================================================
# STEP 4: 3D FDTD Implementation
# ============================================================
def ricker_wavelet(freq, dt, n_samples):
    """Ricker wavelet (Mexican hat) source."""
    t = np.arange(n_samples) * dt
    t0 = 1.5 / freq
    t_shifted = t - t0
    pi_f_t = (np.pi * freq * t_shifted) ** 2
    wavelet = (1 - 2 * pi_f_t) * np.exp(-pi_f_t)
    return wavelet


def fdtd_3d(rho, c, source_pos, receivers, dx, dt, n_steps, source_signal,
            snapshot_times=None):
    """
    3D acoustic FDTD on a staggered grid.
    
    Equations (velocity-pressure formulation):
      v_x^{n+1/2} = v_x^{n-1/2} - (dt/rho) * dp/dx
      v_y^{n+1/2} = v_y^{n-1/2} - (dt/rho) * dp/dy
      v_z^{n+1/2} = v_z^{n-1/2} - (dt/rho) * dp/dz
      p^{n+1} = p^n - dt * rho * c^2 * div(v)
    
    With Mur first-order absorbing boundaries.
    """
    Nx, Ny, Nz = rho.shape
    print(f"\n{'='*60}")
    print(f"STEP 4: Running 3D FDTD")
    print(f"  Grid: {Nx}x{Ny}x{Nz} = {Nx*Ny*Nz:,} points")
    print(f"  Steps: {n_steps}, dt={dt*1e6:.2f}us")
    print(f"{'='*60}")
    
    # Precompute material fields
    c2 = c ** 2
    rho_c2 = rho * c2
    
    # Staggered grid density (harmonic average at half-grid points)
    rho_inv_x = 2.0 / (rho[1:, :, :] + rho[:-1, :, :])
    rho_inv_y = 2.0 / (rho[:, 1:, :] + rho[:, :-1, :])
    rho_inv_z = 2.0 / (rho[:, :, 1:] + rho[:, :, :-1])
    
    dt_dx = dt / dx
    
    # Fields
    p = np.zeros((Nx, Ny, Nz), dtype=np.float64)
    vx = np.zeros((Nx-1, Ny, Nz), dtype=np.float64)
    vy = np.zeros((Nx, Ny-1, Nz), dtype=np.float64)
    vz = np.zeros((Nx, Ny, Nz-1), dtype=np.float64)
    
    # PML absorbing boundary (simple damping)
    pml_width = 20
    damping = np.zeros((Nx, Ny, Nz), dtype=np.float64)
    for i in range(pml_width):
        d = ((pml_width - i) / pml_width) ** 3 * 0.25
        damping[i, :, :] = np.maximum(damping[i, :, :], d)
        damping[Nx-1-i, :, :] = np.maximum(damping[Nx-1-i, :, :], d)
        damping[:, i, :] = np.maximum(damping[:, i, :], d)
        damping[:, Ny-1-i, :] = np.maximum(damping[:, Ny-1-i, :], d)
        damping[:, :, i] = np.maximum(damping[:, :, i], d)
        damping[:, :, Nz-1-i] = np.maximum(damping[:, :, Nz-1-i], d)
    
    decay = 1.0 - damping
    
    # Sensor data
    recv_names = list(receivers.keys())
    recv_positions = [receivers[n] for n in recv_names]
    sensor_data = {name: np.zeros(n_steps, dtype=np.float64) for name in recv_names}
    
    # Snapshots
    snapshots = {}
    if snapshot_times is None:
        snapshot_times = []
    
    sx, sy, sz = source_pos
    
    t_start = time.time()
    
    for t_step in range(n_steps):
        # Inject source (soft source)
        if t_step < len(source_signal):
            p[sx, sy, sz] += source_signal[t_step]
        
        # Update velocity from pressure gradient
        # vx -= dt/rho * dp/dx
        vx -= dt_dx * rho_inv_x * (p[1:, :, :] - p[:-1, :, :])
        vy -= dt_dx * rho_inv_y * (p[:, 1:, :] - p[:, :-1, :])
        vz -= dt_dx * rho_inv_z * (p[:, :, 1:] - p[:, :, :-1])
        
        # Update pressure from velocity divergence
        # p -= dt * rho * c^2 * div(v)
        div_v = np.zeros_like(p)
        div_v[1:-1, :, :] += (vx[1:, :, :] - vx[:-1, :, :]) / dx
        div_v[:, 1:-1, :] += (vy[:, 1:, :] - vy[:, :-1, :]) / dx
        div_v[:, :, 1:-1] += (vz[:, :, 1:] - vz[:, :, :-1]) / dx
        p -= dt * rho_c2 * div_v
        
        # PML damping
        p *= decay
        vx *= decay[1:, :, :]  # approximate - use same decay
        vy *= decay[:, 1:, :]
        vz *= decay[:, :, 1:]
        
        # Record receivers
        for name, pos in zip(recv_names, recv_positions):
            ix, iy, iz = int(pos[0]), int(pos[1]), int(pos[2])
            if 0 <= ix < Nx and 0 <= iy < Ny and 0 <= iz < Nz:
                sensor_data[name][t_step] = p[ix, iy, iz]
        
        # Save snapshots
        if t_step in snapshot_times:
            snapshots[t_step] = p.copy()
        
        # Progress
        if (t_step + 1) % 500 == 0:
            max_p = float(np.max(np.abs(p)))
            elapsed = time.time() - t_start
            rate = (t_step + 1) / elapsed
            eta = (n_steps - t_step - 1) / rate
            print(f"  Step {t_step+1}/{n_steps}, max|p|={max_p:.4f}, "
                  f"{rate:.0f} steps/s, ETA {eta:.0f}s", flush=True)
            if max_p > 1e10 or np.isnan(max_p):
                print("  UNSTABLE - aborting!", flush=True)
                break
    
    total_time = time.time() - t_start
    print(f"\n  Simulation complete in {total_time:.1f}s ({n_steps/total_time:.0f} steps/s)")
    
    return sensor_data, snapshots, p


# ============================================================
# STEP 5: Analysis
# ============================================================
def analyze_results(sensor_data, dt, n_steps, grid_info):
    """Extract IPI, spectra, beam pattern from receiver data."""
    print(f"\n{'='*60}")
    print("STEP 5: Analyzing results")
    print(f"{'='*60}")
    
    results = {}
    time_axis = np.arange(n_steps) * dt * 1000  # ms
    
    # Forward signal analysis
    fwd = sensor_data["forward"]
    abs_fwd = np.abs(fwd)
    
    # Find peaks (P0, P1, P2...)
    threshold = np.max(abs_fwd) * 0.1 if np.max(abs_fwd) > 0 else 0
    peaks = []
    in_peak = False
    min_gap = int(0.5e-3 / dt)  # minimum 0.5ms between peaks
    
    for i in range(1, len(abs_fwd) - 1):
        if (abs_fwd[i] > threshold and 
            abs_fwd[i] > abs_fwd[i-1] and 
            abs_fwd[i] >= abs_fwd[i+1]):
            if not in_peak and (not peaks or (i - peaks[-1]) > min_gap):
                peaks.append(i)
                in_peak = True
        elif abs_fwd[i] < threshold * 0.3:
            in_peak = False
    
    peak_times_ms = [p * dt * 1000 for p in peaks]
    results["peak_times_ms"] = peak_times_ms
    results["n_pulses"] = len(peaks)
    
    if len(peak_times_ms) >= 2:
        ipis = [peak_times_ms[i+1] - peak_times_ms[i] for i in range(len(peak_times_ms)-1)]
        results["ipi_ms"] = ipis
        results["mean_ipi_ms"] = np.mean(ipis)
        print(f"  Detected {len(peaks)} pulses")
        print(f"  Peak times (ms): {[f'{t:.2f}' for t in peak_times_ms]}")
        print(f"  IPI (ms): {[f'{i:.2f}' for i in ipis]}")
        print(f"  Mean IPI: {np.mean(ipis):.2f} ms")
    else:
        results["ipi_ms"] = []
        results["mean_ipi_ms"] = 0
        print(f"  Only {len(peaks)} pulse(s) detected - cannot measure IPI")
    
    # FFT of forward signal
    fft_fwd = np.abs(np.fft.rfft(fwd))
    freqs = np.fft.rfftfreq(len(fwd), dt)
    results["fft_fwd"] = fft_fwd
    results["freqs"] = freqs
    
    # Peak frequency (above 500 Hz)
    mask = freqs > 500
    if np.any(mask) and np.any(fft_fwd[mask] > 0):
        peak_idx = np.argmax(fft_fwd[mask])
        results["peak_freq_hz"] = float(freqs[mask][peak_idx])
        print(f"  Peak frequency: {results['peak_freq_hz']:.0f} Hz")
    
    # Spectral centroid
    total_e = np.sum(fft_fwd**2)
    if total_e > 0:
        results["spectral_centroid_hz"] = float(np.sum(freqs * fft_fwd**2) / total_e)
        print(f"  Spectral centroid: {results['spectral_centroid_hz']:.0f} Hz")
    
    # Beam pattern: peak pressure at each receiver
    print(f"\n  Beam pattern (peak pressure at each receiver):")
    beam = {}
    for name, data in sensor_data.items():
        peak_amp = float(np.max(np.abs(data)))
        beam[name] = peak_amp
        print(f"    {name:>10s}: {peak_amp:.6f}")
    results["beam_pattern"] = beam
    
    # Front/back ratio
    if beam.get("backward", 0) > 0:
        fb_ratio = 20 * np.log10(beam["forward"] / beam["backward"])
        results["front_back_ratio_db"] = fb_ratio
        print(f"  Front/back ratio: {fb_ratio:.1f} dB")
    
    # Up/down ratio
    if beam.get("down", 0) > 0 and beam.get("up", 0) > 0:
        ud_ratio = 20 * np.log10(beam["up"] / beam["down"])
        results["up_down_ratio_db"] = ud_ratio
        print(f"  Up/down ratio: {ud_ratio:.1f} dB")
    
    # Left/right symmetry
    if beam.get("left", 0) > 0 and beam.get("right", 0) > 0:
        lr_ratio = 20 * np.log10(beam["left"] / beam["right"])
        results["left_right_ratio_db"] = lr_ratio
        print(f"  Left/right ratio: {lr_ratio:.1f} dB")
    
    return results


# ============================================================
# STEP 6: Visualization
# ============================================================
def visualize(rho, c, tissue_mask, bone_mask, sensor_data, snapshots,
              results, grid_info, source_pos, receivers):
    """4-panel figure: tissue map, pressure snapshot, time series, spectra+beam."""
    print(f"\n{'='*60}")
    print("STEP 6: Generating visualization")
    print(f"{'='*60}")
    
    Nx, Ny, Nz = rho.shape
    dx = grid_info["dx"]
    dt = grid_info["dt"]
    mid_x = grid_info["skull_mid_x"]
    mid_y = grid_info["skull_mid_y"]
    mid_z = grid_info["skull_mid_z"]
    
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # --- Panel A: 3D tissue map (three orthogonal slices) ---
    ax_a = fig.add_subplot(gs[0, 0])
    
    # Create a composite showing sagittal (XY at mid-Z), horizontal (XZ at mid-Y), coronal (YZ at mid-X)
    # Use a sub-gridspec
    gs_a = gs[0, 0].subgridspec(1, 3, wspace=0.1)
    ax_a.remove()
    
    # Tissue colormap: water=blue, spermaceti=yellow, junk=orange, bone=white, air_sac=black, connective=green
    from matplotlib.colors import ListedColormap, BoundaryNorm
    tissue_colors = ['#2196F3', '#FFD700', '#FF8C00', '#111111', '#333333', '#4CAF50', '#FFFFFF']
    tissue_cmap = ListedColormap(tissue_colors)
    tissue_labels = ['Water', 'Spermaceti', 'Junk', 'Frontal sac', 'Distal sac', 'Connective', 'Bone']
    
    # Build tissue display map (0-6)
    tissue_display = np.zeros_like(tissue_mask, dtype=np.float32)
    tissue_display[tissue_mask == 0] = 0  # water
    tissue_display[tissue_mask == 1] = 1  # spermaceti
    tissue_display[tissue_mask == 2] = 2  # junk
    tissue_display[tissue_mask == 3] = 3  # frontal air sac
    tissue_display[tissue_mask == 4] = 4  # distal air sac
    tissue_display[tissue_mask == 5] = 5  # connective
    tissue_display[bone_mask] = 6         # bone
    
    bounds_cm = [0, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
    norm = BoundaryNorm(bounds_cm, tissue_cmap.N)
    
    # Sagittal (XY at mid-Z)
    ax_sag = fig.add_subplot(gs_a[0])
    ax_sag.imshow(tissue_display[:, :, mid_z].T, origin='lower', aspect='auto',
                  cmap=tissue_cmap, norm=norm)
    ax_sag.set_title('Sagittal (mid-Z)', fontsize=10)
    ax_sag.set_xlabel('X (A-P)')
    ax_sag.set_ylabel('Y (D-V)')
    
    # Horizontal (XZ at mid-Y)
    ax_hor = fig.add_subplot(gs_a[1])
    ax_hor.imshow(tissue_display[:, mid_y, :].T, origin='lower', aspect='auto',
                  cmap=tissue_cmap, norm=norm)
    ax_hor.set_title('Horizontal (mid-Y)', fontsize=10)
    ax_hor.set_xlabel('X (A-P)')
    ax_hor.set_ylabel('Z (L-R)')
    
    # Coronal (YZ at mid-X)
    ax_cor = fig.add_subplot(gs_a[2])
    im = ax_cor.imshow(tissue_display[mid_x, :, :].T, origin='lower', aspect='auto',
                       cmap=tissue_cmap, norm=norm)
    ax_cor.set_title('Coronal (mid-X)', fontsize=10)
    ax_cor.set_xlabel('Y (D-V)')
    ax_cor.set_ylabel('Z (L-R)')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=tissue_colors[i], label=tissue_labels[i]) 
                      for i in range(len(tissue_labels))]
    ax_cor.legend(handles=legend_elements, loc='upper right', fontsize=7,
                  bbox_to_anchor=(1.0, 1.0))
    
    fig.text(0.25, 0.95, 'A: 3D Tissue Map (Orthogonal Slices)', fontsize=14, 
             fontweight='bold', ha='center')
    
    # --- Panel B: Pressure field snapshot ---
    ax_b_area = fig.add_subplot(gs[0, 1])
    
    if snapshots:
        # Use the snapshot closest to P1 arrival
        snap_key = list(snapshots.keys())[0]
        p_snap = snapshots[snap_key]
        
        gs_b = gs[0, 1].subgridspec(1, 2, wspace=0.15)
        ax_b_area.remove()
        
        # Sagittal pressure slice
        ax_bs = fig.add_subplot(gs_b[0])
        p_sag = p_snap[:, :, mid_z]
        vmax = np.max(np.abs(p_sag)) * 0.8
        if vmax == 0:
            vmax = 1
        ax_bs.imshow(p_sag.T, origin='lower', aspect='auto', cmap='RdBu_r',
                     vmin=-vmax, vmax=vmax)
        # Overlay bone outline
        bone_sag = bone_mask[:, :, mid_z].astype(float)
        ax_bs.contour(bone_sag.T, levels=[0.5], colors='black', linewidths=0.5)
        ax_bs.set_title(f'Sagittal, t={snap_key*dt*1000:.2f}ms', fontsize=10)
        ax_bs.set_xlabel('X')
        ax_bs.set_ylabel('Y')
        
        # Horizontal pressure slice
        ax_bh = fig.add_subplot(gs_b[1])
        p_hor = p_snap[:, mid_y, :]
        vmax2 = np.max(np.abs(p_hor)) * 0.8
        if vmax2 == 0:
            vmax2 = 1
        im_p = ax_bh.imshow(p_hor.T, origin='lower', aspect='auto', cmap='RdBu_r',
                            vmin=-vmax2, vmax=vmax2)
        bone_hor = bone_mask[:, mid_y, :].astype(float)
        ax_bh.contour(bone_hor.T, levels=[0.5], colors='black', linewidths=0.5)
        ax_bh.set_title(f'Horizontal, t={snap_key*dt*1000:.2f}ms', fontsize=10)
        ax_bh.set_xlabel('X')
        ax_bh.set_ylabel('Z')
        plt.colorbar(im_p, ax=ax_bh, label='Pressure')
        
        fig.text(0.75, 0.95, 'B: Pressure Field Snapshot', fontsize=14,
                 fontweight='bold', ha='center')
    else:
        ax_b_area.text(0.5, 0.5, 'No snapshots captured', transform=ax_b_area.transAxes,
                       ha='center', va='center')
        fig.text(0.75, 0.95, 'B: Pressure Field', fontsize=14, fontweight='bold', ha='center')
    
    # --- Panel C: Time series at all 6 receivers ---
    ax_c = fig.add_subplot(gs[1, 0])
    n_steps = len(list(sensor_data.values())[0])
    time_ms = np.arange(n_steps) * dt * 1000
    
    colors = {'forward': '#E53935', 'backward': '#1E88E5', 'up': '#43A047',
              'down': '#FB8C00', 'left': '#8E24AA', 'right': '#00ACC1'}
    
    for name, data in sensor_data.items():
        ax_c.plot(time_ms, data / (np.max(np.abs(data)) + 1e-30), 
                  label=name, color=colors.get(name, 'gray'), linewidth=0.8, alpha=0.85)
    
    ax_c.set_xlabel('Time (ms)', fontsize=11)
    ax_c.set_ylabel('Normalized Pressure', fontsize=11)
    ax_c.set_title('C: Time Series at 6 Receivers', fontsize=14, fontweight='bold')
    ax_c.legend(fontsize=9, loc='upper right')
    ax_c.set_xlim(0, time_ms[-1])
    ax_c.grid(True, alpha=0.3)
    
    # Mark detected peaks on forward signal
    if results.get("peak_times_ms"):
        for i, pt in enumerate(results["peak_times_ms"]):
            ax_c.axvline(pt, color='red', linestyle='--', alpha=0.4, linewidth=0.7)
            ax_c.text(pt, 0.95, f'P{i}', fontsize=8, color='red', ha='center')
    
    # --- Panel D: Spectral comparison + beam pattern ---
    gs_d = gs[1, 1].subgridspec(1, 2, wspace=0.35)
    
    # Spectrum
    ax_spec = fig.add_subplot(gs_d[0])
    freqs = results.get("freqs", np.array([]))
    fft_fwd = results.get("fft_fwd", np.array([]))
    
    if len(freqs) > 0 and len(fft_fwd) > 0:
        # Normalize
        fft_norm = fft_fwd / (np.max(fft_fwd) + 1e-30)
        mask = freqs < 40000
        ax_spec.plot(freqs[mask] / 1000, fft_norm[mask], color='#E53935', linewidth=1)
        ax_spec.set_xlabel('Frequency (kHz)', fontsize=11)
        ax_spec.set_ylabel('Normalized Amplitude', fontsize=11)
        ax_spec.set_title('Forward Spectrum', fontsize=11)
        ax_spec.grid(True, alpha=0.3)
        if results.get("peak_freq_hz"):
            ax_spec.axvline(results["peak_freq_hz"]/1000, color='blue', linestyle='--',
                           alpha=0.5, label=f'Peak: {results["peak_freq_hz"]/1000:.1f}kHz')
            ax_spec.legend(fontsize=9)
    
    # Beam pattern polar plot
    ax_beam = fig.add_subplot(gs_d[1], polar=True)
    beam = results.get("beam_pattern", {})
    if beam:
        # Map directions to angles (0=forward, pi=backward, pi/2=up, etc.)
        dir_angles = {
            "forward": 0, "right": np.pi/2, "backward": np.pi, "left": 3*np.pi/2,
            "up": np.pi/4, "down": 7*np.pi/4  # approximate
        }
        angles = []
        amps = []
        for name, amp in beam.items():
            if name in dir_angles:
                angles.append(dir_angles[name])
                amps.append(amp)
        
        # Close the polygon
        if angles:
            order = np.argsort(angles)
            angles_sorted = [angles[i] for i in order] + [angles[order[0]] + 2*np.pi]
            amps_sorted = [amps[i] for i in order] + [amps[order[0]]]
            max_amp = max(amps_sorted) if amps_sorted else 1
            amps_norm = [a / (max_amp + 1e-30) for a in amps_sorted]
            
            ax_beam.plot(angles_sorted, amps_norm, 'o-', color='#E53935', linewidth=2)
            ax_beam.fill(angles_sorted, amps_norm, alpha=0.2, color='#E53935')
            ax_beam.set_title('Beam Pattern', fontsize=11, pad=15)
            ax_beam.set_rticks([0.25, 0.5, 0.75, 1.0])
    
    fig.suptitle('3D FDTD Acoustic Simulation - Sperm Whale Head (Real Skull Scan)',
                 fontsize=16, fontweight='bold', y=0.98)
    
    out_path = f"{OUTPUT_DIR}/skull_3d_results.png"
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n  Figure saved: {out_path}")
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================
def main():
    total_start = time.time()
    
    # Step 1: Load skull
    mesh, dims = load_skull()
    
    # Step 2+3: Build domain and tissue map
    dx = 0.02  # Start with 2cm - good balance of resolution and speed
    rho, c, tissue_mask, bone_mask, source_pos, receivers, grid_info = build_3d_domain(mesh, dx=dx)
    
    dt = grid_info["dt"]
    Nx, Ny, Nz = grid_info["Nx"], grid_info["Ny"], grid_info["Nz"]
    
    # Step 4: Run FDTD
    duration_ms = 12.0
    n_steps = int(duration_ms / 1000.0 / dt)
    center_freq = 12000  # 12 kHz
    
    # Source signal
    n_source = int(0.5e-3 / dt)  # 0.5ms wavelet
    source_signal = ricker_wavelet(center_freq, dt, n_source) * 5000
    
    # Determine snapshot time (estimate P1 arrival ~ 2-4ms)
    # P1 travels: source -> frontal sac -> reflect -> spermaceti -> skull -> reflect -> forward
    # Rough: spermaceti length ~1.5m, c=1370 -> 1.1ms one way, round trip ~2.2ms plus water travel
    p1_time_step = int(3.0e-3 / dt)  # ~3ms
    snapshot_times = [p1_time_step]
    
    print(f"\n  Duration: {duration_ms}ms = {n_steps} steps")
    print(f"  Source: {center_freq}Hz Ricker wavelet, {n_source} samples")
    print(f"  Snapshot at step {p1_time_step} ({p1_time_step*dt*1000:.2f}ms)")
    
    sensor_data, snapshots, final_p = fdtd_3d(
        rho, c, source_pos, receivers, dx, dt, n_steps, source_signal,
        snapshot_times=snapshot_times
    )
    
    # Step 5: Analyze
    results = analyze_results(sensor_data, dt, n_steps, grid_info)
    
    # Step 6: Visualize
    visualize(rho, c, tissue_mask, bone_mask, sensor_data, snapshots,
              results, grid_info, source_pos, receivers)
    
    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"COMPLETE - Total time: {total_time:.1f}s")
    print(f"{'='*60}")
    
    # Summary
    print(f"\nSUMMARY:")
    print(f"  Grid: {Nx}x{Ny}x{Nz} ({Nx*Ny*Nz:,} points), dx={dx*100:.1f}cm")
    print(f"  Duration: {duration_ms}ms, {n_steps} steps, dt={dt*1e6:.2f}us")
    print(f"  Pulses detected: {results.get('n_pulses', 0)}")
    if results.get('mean_ipi_ms', 0) > 0:
        print(f"  Mean IPI: {results['mean_ipi_ms']:.2f} ms")
    if results.get('peak_freq_hz'):
        print(f"  Peak frequency: {results['peak_freq_hz']:.0f} Hz")
    if results.get('spectral_centroid_hz'):
        print(f"  Spectral centroid: {results['spectral_centroid_hz']:.0f} Hz")
    if results.get('front_back_ratio_db'):
        print(f"  Front/back ratio: {results['front_back_ratio_db']:.1f} dB")
    if results.get('up_down_ratio_db'):
        print(f"  Up/down ratio: {results['up_down_ratio_db']:.1f} dB")
    if results.get('left_right_ratio_db'):
        print(f"  Left/right ratio: {results['left_right_ratio_db']:.1f} dB")
    
    beam = results.get('beam_pattern', {})
    if beam:
        print(f"\n  Beam pattern (peak pressure):")
        for name, amp in sorted(beam.items(), key=lambda x: -x[1]):
            print(f"    {name:>10s}: {amp:.6f}")


if __name__ == "__main__":
    main()

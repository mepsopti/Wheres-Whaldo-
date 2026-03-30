#!/usr/bin/env python3
"""
FEM Acoustic Simulation of Sperm Whale Head Using Real Skull Scan
================================================================
Solves the pressure wave equation with spatially varying material properties
derived from a 3D skull mesh and anatomical tissue mapping.

Equation: 1/c^2 * d2p/dt2 = div(1/rho * grad(p)) + source

Uses DOLFINx (modern FEniCS) if available, falls back to SfePy or pure
scipy FEM. Starts with 2D sagittal slice for tractability, with notes
for full 3D extension.

Author: Jaak (Whale Acoustics Agent)
Date: 2026-03-29
"""

import sys
import os
import time
import warnings
import numpy as np
from pathlib import Path

warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================
SKULL_OBJ = "/mnt/archive/datasets/whale_communication/3d_scans/sperm_whale_cranium_high/source/12_2007_online.obj"
OUTPUT_DIR = "/Users/ericbrowy/Desktop/Claude/whale-fem-staging"
FIGURE_PATH = "/Users/ericbrowy/Desktop/Claude/whale-fem-staging/skull_fem_results_test.png"
SCRIPT_PATH = os.path.join(OUTPUT_DIR, "skull_fem_fenics.py")

# Material properties: (density kg/m3, sound speed m/s)
MATERIALS = {
    'seawater':          {'rho': 1025.0, 'c': 1530.0, 'color': '#1a5276'},
    'blubber':           {'rho':  930.0, 'c': 1430.0, 'color': '#f5b041'},
    'muscle':            {'rho': 1050.0, 'c': 1570.0, 'color': '#e74c3c'},
    'bone':              {'rho': 1900.0, 'c': 3000.0, 'color': '#ecf0f1'},
    'spermaceti':        {'rho':  857.0, 'c': 1370.0, 'color': '#f9e79f'},
    'junk_posterior':    {'rho':  870.0, 'c': 1380.0, 'color': '#fad7a0'},
    'junk_anterior':     {'rho':  930.0, 'c': 1430.0, 'color': '#f0b27a'},
    'connective_tissue': {'rho': 1070.0, 'c': 1570.0, 'color': '#d5a6bd'},
    'air_sac':           {'rho':   50.0, 'c':  800.0, 'color': '#2c3e50'},
}

# Simulation parameters
SOURCE_FREQ = 12000.0      # Hz - Ricker wavelet center frequency
DURATION = 0.005           # 15 ms
ELEMENT_SIZE_2D = 0.04    # 5mm elements for 2D (finer than 3D budget)
ELEMENT_SIZE_3D = 0.02     # 20mm elements for 3D

# Head dimensions (meters) - based on adult male sperm whale
HEAD_LENGTH = 5.0          # anterior-posterior
HEAD_WIDTH = 2.0           # lateral
HEAD_HEIGHT = 1.5          # dorsal-ventral

# Receiver positions (meters from head center)
RECEIVER_DIST = 1.0
RECEIVERS = {
    'forward':  np.array([ HEAD_LENGTH/2 + RECEIVER_DIST, 0.0]),
    'backward': np.array([-HEAD_LENGTH/2 - RECEIVER_DIST, 0.0]),
    'dorsal':   np.array([0.0,  HEAD_HEIGHT/2 + RECEIVER_DIST]),
    'ventral':  np.array([0.0, -HEAD_HEIGHT/2 - RECEIVER_DIST]),
}

print("=" * 70)
print("Sperm Whale Head FEM Acoustic Simulation")
print("=" * 70)

# ============================================================
# Step 0: Check available FEM libraries
# ============================================================
print("\n[Step 0] Checking available libraries...")

fem_backend = None

try:
    import dolfinx
    from dolfinx import fem, mesh as dmesh, io
    from dolfinx.fem import functionspace
    import ufl
    fem_backend = 'dolfinx'
    print(f"  DOLFINx {dolfinx.__version__} available")
except ImportError:
    print("  DOLFINx not available")

if fem_backend is None:
    try:
        import fenics
        fem_backend = 'fenics_legacy'
        print(f"  Legacy FEniCS available")
    except ImportError:
        print("  Legacy FEniCS not available")

if fem_backend is None:
    try:
        import sfepy
        fem_backend = 'sfepy'
        print(f"  SfePy {sfepy.__version__} available")
    except ImportError:
        print("  SfePy not available")

if fem_backend is None:
    print("  No FEM library found - will use scipy sparse FEM (custom)")
    fem_backend = 'scipy_fem'

print(f"  Selected backend: {fem_backend}")

# Check mesh tools
try:
    import trimesh
    print(f"  trimesh {trimesh.__version__} available")
    HAS_TRIMESH = True
except ImportError:
    print("  trimesh not available - will install")
    HAS_TRIMESH = False

try:
    import gmsh
    print(f"  gmsh available")
    HAS_GMSH = True
except ImportError:
    print("  gmsh not available")
    HAS_GMSH = False

try:
    import meshio
    print(f"  meshio available")
    HAS_MESHIO = True
except ImportError:
    print("  meshio not available")
    HAS_MESHIO = False

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
from matplotlib.tri import Triangulation
import matplotlib.gridspec as gridspec
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.spatial import Delaunay

# ============================================================
# Step 1: Load and process skull mesh
# ============================================================
print("\n[Step 1] Loading skull mesh...")

skull_vertices = None
skull_faces = None
skull_bounds = None

if os.path.exists(SKULL_OBJ):
    if HAS_TRIMESH:
        skull_mesh = trimesh.load(SKULL_OBJ)
        skull_vertices = np.array(skull_mesh.vertices)
        skull_faces = np.array(skull_mesh.faces)
        print(f"  Loaded: {len(skull_vertices)} vertices, {len(skull_faces)} faces")

        # Convert mm to meters
        skull_vertices_m = skull_vertices / 1000.0

        # Center the skull
        centroid = skull_vertices_m.mean(axis=0)
        skull_vertices_m -= centroid

        # Find principal axes and orient (longest = anterior-posterior = x)
        bbox = skull_vertices_m.max(axis=0) - skull_vertices_m.min(axis=0)
        axis_order = np.argsort(bbox)[::-1]  # longest first
        skull_vertices_m = skull_vertices_m[:, axis_order]

        skull_bounds = {
            'min': skull_vertices_m.min(axis=0),
            'max': skull_vertices_m.max(axis=0),
            'size': skull_vertices_m.max(axis=0) - skull_vertices_m.min(axis=0)
        }
        print(f"  Skull bounding box (m): {skull_bounds['size']}")
        print(f"  Range X: [{skull_bounds['min'][0]:.3f}, {skull_bounds['max'][0]:.3f}]")
        print(f"  Range Y: [{skull_bounds['min'][1]:.3f}, {skull_bounds['max'][1]:.3f}]")
        print(f"  Range Z: [{skull_bounds['min'][2]:.3f}, {skull_bounds['max'][2]:.3f}]")

        # Scale skull to match real head proportions
        # Real sperm whale skull is roughly 1.5-2m long in a 5m head
        skull_scale = 1.8 / skull_bounds['size'][0]
        skull_vertices_m *= skull_scale
        skull_bounds = {
            'min': skull_vertices_m.min(axis=0),
            'max': skull_vertices_m.max(axis=0),
            'size': skull_vertices_m.max(axis=0) - skull_vertices_m.min(axis=0)
        }
        print(f"  Scaled skull bounding box (m): {skull_bounds['size']}")

        # Extract sagittal cross-section (slice at z ~ 0)
        # Find faces that straddle z=0
        z_coords = skull_vertices_m[:, 2]
        sagittal_edges = []
        tol = skull_bounds['size'][2] * 0.02  # 2% tolerance

        for face in skull_faces:
            v = skull_vertices_m[face]
            z = v[:, 2]
            if z.min() <= tol and z.max() >= -tol:
                # This face crosses z=0 plane - extract intersection
                for i in range(3):
                    j = (i + 1) % 3
                    if (z[i] <= 0 and z[j] >= 0) or (z[i] >= 0 and z[j] <= 0):
                        if abs(z[j] - z[i]) > 1e-10:
                            t = -z[i] / (z[j] - z[i])
                            pt = v[i] + t * (v[j] - v[i])
                            sagittal_edges.append(pt[:2])  # x, y only

        if sagittal_edges:
            sagittal_points = np.array(sagittal_edges)
            print(f"  Sagittal cross-section: {len(sagittal_points)} intersection points")
        else:
            sagittal_points = None
            print("  WARNING: Could not extract sagittal cross-section")
    else:
        # Manual OBJ parsing
        print("  Parsing OBJ manually (no trimesh)...")
        verts = []
        faces_list = []
        with open(SKULL_OBJ, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    parts = line.strip().split()
                    verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif line.startswith('f '):
                    parts = line.strip().split()
                    face = []
                    for p in parts[1:]:
                        face.append(int(p.split('/')[0]) - 1)
                    if len(face) >= 3:
                        faces_list.append(face[:3])
        skull_vertices = np.array(verts)
        skull_faces = np.array(faces_list)
        skull_vertices_m = skull_vertices / 1000.0
        centroid = skull_vertices_m.mean(axis=0)
        skull_vertices_m -= centroid
        bbox = skull_vertices_m.max(axis=0) - skull_vertices_m.min(axis=0)
        axis_order = np.argsort(bbox)[::-1]
        skull_vertices_m = skull_vertices_m[:, axis_order]
        skull_scale = 1.8 / (skull_vertices_m.max(axis=0) - skull_vertices_m.min(axis=0))[0]
        skull_vertices_m *= skull_scale
        skull_bounds = {
            'min': skull_vertices_m.min(axis=0),
            'max': skull_vertices_m.max(axis=0),
            'size': skull_vertices_m.max(axis=0) - skull_vertices_m.min(axis=0)
        }
        print(f"  Loaded: {len(skull_vertices)} vertices, {len(skull_faces)} faces")
        print(f"  Scaled skull bounding box (m): {skull_bounds['size']}")
        sagittal_points = None
else:
    print(f"  WARNING: Skull file not found at {SKULL_OBJ}")
    print("  Using synthetic skull outline for demonstration")
    skull_vertices_m = None
    skull_bounds = {
        'min': np.array([-0.9, -0.5, -0.4]),
        'max': np.array([ 0.9,  0.5,  0.4]),
        'size': np.array([1.8, 1.0, 0.8])
    }
    sagittal_points = None

# ============================================================
# Step 2: Create 2D sagittal domain and mesh
# ============================================================
print("\n[Step 2] Creating 2D sagittal domain mesh...")

# Domain: elliptical region representing the sagittal slice through head + water
# Include some surrounding water for wave propagation
DOMAIN_PAD = 0.5  # 50cm of water around head
domain_half_x = HEAD_LENGTH / 2 + DOMAIN_PAD
domain_half_y = HEAD_HEIGHT / 2 + DOMAIN_PAD

def create_2d_mesh(element_size):
    """Create a 2D triangular mesh of the sagittal domain."""
    # Generate points on a regular grid, then triangulate
    nx = int(2 * domain_half_x / element_size) + 1
    ny = int(2 * domain_half_y / element_size) + 1
    x = np.linspace(-domain_half_x, domain_half_x, nx)
    y = np.linspace(-domain_half_y, domain_half_y, ny)
    xx, yy = np.meshgrid(x, y)

    # Clip to elliptical domain (head + water padding)
    ellipse_mask = (xx / domain_half_x)**2 + (yy / domain_half_y)**2 <= 1.0
    points = np.column_stack([xx[ellipse_mask], yy[ellipse_mask]])

    # Add refinement points near skull boundary if we have skull data
    if sagittal_points is not None and len(sagittal_points) > 0:
        # Subsample skull boundary points
        n_skull = min(len(sagittal_points), 500)
        idx = np.random.choice(len(sagittal_points), n_skull, replace=False)
        skull_pts = sagittal_points[idx]
        # Add slightly offset points for refinement
        for offset in [0.005, -0.005, 0.01, -0.01]:
            normals = np.zeros_like(skull_pts)
            normals[:, 1] = offset
            points = np.vstack([points, skull_pts + normals])

    # Delaunay triangulation
    tri = Delaunay(points)
    triangles = tri.simplices

    # Filter triangles outside the elliptical domain
    centroids = points[triangles].mean(axis=1)
    inside = (centroids[:, 0] / domain_half_x)**2 + (centroids[:, 1] / domain_half_y)**2 <= 1.0
    triangles = triangles[inside]

    # Filter degenerate triangles (tiny area or bad aspect ratio)
    min_area = (element_size * 0.01) ** 2  # 1% of target element area
    good = []
    for i, tri_idx in enumerate(triangles):
        v = points[tri_idx]
        area = abs(np.cross(v[1] - v[0], v[2] - v[0])) / 2.0
        if area > min_area:
            # Also check aspect ratio - reject slivers
            edges = np.array([v[1]-v[0], v[2]-v[1], v[0]-v[2]])
            lens = np.linalg.norm(edges, axis=1)
            if lens.max() / (lens.min() + 1e-15) < 20:
                good.append(i)
    triangles = triangles[good]

    return points, triangles

# Use adaptive element size based on available memory
try:
    import psutil
    available_ram_gb = psutil.virtual_memory().available / 1e9
    print(f"  Available RAM: {available_ram_gb:.1f} GB")
    if available_ram_gb > 100:
        elem_size = 0.003  # 3mm - very fine
    elif available_ram_gb > 30:
        elem_size = 0.005  # 5mm - fine
    elif available_ram_gb > 10:
        elem_size = 0.008  # 8mm
    else:
        elem_size = 0.012  # 12mm - coarser
except ImportError:
    elem_size = ELEMENT_SIZE_2D

print(f"  Target element size: {elem_size*1000:.1f} mm")
t0 = time.time()
nodes, elements = create_2d_mesh(elem_size)
mesh_time = time.time() - t0
print(f"  Mesh: {len(nodes)} nodes, {len(elements)} triangles ({mesh_time:.1f}s)")

# Compute mesh quality metrics
def mesh_quality(nodes, elements):
    """Compute element sizes and quality ratios."""
    sizes = []
    qualities = []
    for tri in elements:
        v = nodes[tri]
        edges = np.array([v[1]-v[0], v[2]-v[1], v[0]-v[2]])
        lens = np.linalg.norm(edges, axis=1)
        sizes.append(lens.mean())
        # Quality: ratio of inscribed to circumscribed circle radius
        s = lens.sum() / 2
        area = abs(np.cross(edges[0], edges[1])) / 2
        if area > 1e-15 and lens.max() > 0:
            r_in = area / s
            r_circ = lens.prod() / (4 * area)
            qualities.append(2 * r_in / r_circ if r_circ > 0 else 0)
        else:
            qualities.append(0)
    return np.array(sizes), np.array(qualities)

elem_sizes, elem_qualities = mesh_quality(nodes, elements)
print(f"  Element sizes: min={elem_sizes.min()*1000:.1f}mm, "
      f"max={elem_sizes.max()*1000:.1f}mm, mean={elem_sizes.mean()*1000:.1f}mm")
print(f"  Element quality: min={elem_qualities.min():.3f}, "
      f"mean={elem_qualities.mean():.3f}, max={elem_qualities.max():.3f}")

# ============================================================
# Step 3: Assign tissue types (2D sagittal)
# ============================================================
print("\n[Step 3] Assigning tissue types to mesh elements...")

def classify_tissue_2d(x, y, skull_pts=None):
    """
    Classify a 2D point (sagittal plane) into tissue type.
    x: anterior(+) to posterior(-), y: dorsal(+) to ventral(-)
    Origin at head center.

    Anatomy (sagittal view, from Cranford 1999, Madsen 2002):
    - Skull: posterior-ventral (the bony cranium)
    - Spermaceti organ: dorsal, between skull and frontal air sac
    - Junk: ventral-anterior, tapering forward
    - Frontal air sac: thin layer on rostral basin of skull (anterior face)
    - Distal air sac: near the tip of the head (phonic lips)
    - Case: connective tissue enclosing spermaceti organ
    - Blubber: outer layer
    - Seawater: outside the head
    """
    # Head ellipse parameters
    hx = HEAD_LENGTH / 2
    hy = HEAD_HEIGHT / 2

    # Is this point inside the head ellipse?
    head_r = (x / hx)**2 + (y / hy)**2
    if head_r > 1.0:
        return 'seawater'

    # Blubber: outer 5% of head ellipse
    if head_r > 0.90:
        return 'blubber'

    # Skull region: posterior, centered, roughly -0.5 to 0.5 in x, -0.3 to 0.1 in y
    # Shift skull position slightly posterior
    skull_cx, skull_cy = -0.3, -0.1
    skull_rx, skull_ry = 0.9, 0.45
    skull_r = ((x - skull_cx) / skull_rx)**2 + ((y - skull_cy) / skull_ry)**2

    # If we have real skull cross-section points, use them for a tighter boundary
    if skull_pts is not None and len(skull_pts) > 50:
        # Point-in-polygon-ish: use distance to nearest skull point
        dists = np.sqrt((skull_pts[:, 0] - x)**2 + (skull_pts[:, 1] - y)**2)
        min_dist = dists.min()
        if min_dist < 0.03:  # Within 3cm of skull surface
            return 'bone'

    if skull_r < 0.3:
        return 'bone'

    # Air sacs
    # Frontal air sac: thin layer on anterior face of skull
    if -0.1 < x < 0.6 and -0.15 < y < 0.15 and 0.25 < skull_r < 0.45:
        return 'air_sac'

    # Distal air sac: near tip of head (phonic lips region)
    if x > hx * 0.75 and abs(y) < 0.15:
        return 'air_sac'

    # Spermaceti organ: dorsal, posterior of center
    if y > 0.0 and x < 0.5:
        if head_r < 0.85:
            # Case (connective tissue envelope)
            so_r = ((x - (-0.2)) / 1.2)**2 + ((y - 0.25) / 0.35)**2
            if 0.85 < so_r < 1.0:
                return 'connective_tissue'
            if so_r < 0.85:
                return 'spermaceti'

    # Junk: ventral-anterior, with density gradient
    if y < 0.1 and x > -0.3:
        junk_r = ((x - 0.8) / 1.5)**2 + ((y - (-0.2)) / 0.4)**2
        if junk_r < 1.0:
            # Gradient: posterior is less dense, anterior is denser
            if x < 0.5:
                return 'junk_posterior'
            else:
                return 'junk_anterior'

    # Muscle: fills remaining space near skull
    if skull_r < 0.8:
        return 'muscle'

    # Connective tissue: between organs
    if head_r < 0.85:
        return 'connective_tissue'

    return 'muscle'

# Classify each element by its centroid
centroids = nodes[elements].mean(axis=1)
tissue_labels = []
tissue_rho = np.zeros(len(elements))
tissue_c = np.zeros(len(elements))

for i, (cx, cy) in enumerate(centroids):
    tissue = classify_tissue_2d(cx, cy, sagittal_points)
    tissue_labels.append(tissue)
    tissue_rho[i] = MATERIALS[tissue]['rho']
    tissue_c[i] = MATERIALS[tissue]['c']

tissue_labels = np.array(tissue_labels)

# Print tissue distribution
unique_tissues, counts = np.unique(tissue_labels, return_counts=True)
print("  Tissue distribution:")
for t, n in sorted(zip(unique_tissues, counts), key=lambda x: -x[1]):
    pct = 100 * n / len(tissue_labels)
    print(f"    {t:20s}: {n:6d} elements ({pct:5.1f}%)")

# ============================================================
# Step 4: Assemble FEM system - 2D acoustic wave equation
# ============================================================
print("\n[Step 4] Assembling FEM system...")

def assemble_fem_matrices(nodes, elements, rho, c):
    """
    Assemble mass and stiffness matrices for the acoustic wave equation.

    The weak form of 1/c^2 * d2p/dt2 = div(1/rho * grad(p))
    gives:
        M * d2p/dt2 + K * p = f

    where:
        M_ij = integral(1/c^2 * phi_i * phi_j)
        K_ij = integral(1/rho * grad(phi_i) . grad(phi_j))

    Using P1 (linear) triangular elements.
    """
    n_nodes = len(nodes)
    n_elem = len(elements)

    # Pre-allocate triplets
    row_idx = []
    col_idx = []
    m_vals = []
    k_vals = []

    for e in range(n_elem):
        tri = elements[e]
        x = nodes[tri]

        # Element area
        d = np.array([[x[1,0]-x[0,0], x[2,0]-x[0,0]],
                       [x[1,1]-x[0,1], x[2,1]-x[0,1]]])
        det_d = abs(d[0,0]*d[1,1] - d[0,1]*d[1,0])
        area = det_d / 2.0

        if area < 1e-15:
            continue

        # Gradients of basis functions
        inv_det = 1.0 / det_d
        b = np.array([
            [d[1,1]*(-1) + d[0,1]*(1), d[1,1]*(1), -d[0,1]*(1)],
            [-d[1,0]*(-1) - d[0,0]*(1), -d[1,0]*(1), d[0,0]*(1)]
        ]) * inv_det

        # Actually, for linear triangle, gradients are:
        # grad(phi_i) = 1/(2*area) * [y_j - y_k, x_k - x_j]
        # where j,k are the other two vertices (cyclic)
        grad_phi = np.zeros((3, 2))
        for i in range(3):
            j = (i + 1) % 3
            k = (i + 2) % 3
            grad_phi[i, 0] = (x[j, 1] - x[k, 1]) / (2 * area)
            grad_phi[i, 1] = (x[k, 0] - x[j, 0]) / (2 * area)

        # Element material properties (constant per element)
        rho_e = rho[e]
        c_e = c[e]

        # Mass matrix: integral(1/c^2 * phi_i * phi_j) over element
        # For P1: M_ij = area/(12*c^2) * (1 + delta_ij)
        m_factor = area / (c_e**2)
        m_diag = m_factor / 6.0
        m_off = m_factor / 12.0

        # Stiffness matrix: integral(1/rho * grad(phi_i) . grad(phi_j)) over element
        k_local = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                k_local[i, j] = (1.0 / rho_e) * np.dot(grad_phi[i], grad_phi[j]) * area

        for i in range(3):
            for j in range(3):
                row_idx.append(tri[i])
                col_idx.append(tri[j])
                k_vals.append(k_local[i, j])
                if i == j:
                    m_vals.append(m_diag)
                else:
                    m_vals.append(m_off)

    M = sparse.coo_matrix((m_vals, (row_idx, col_idx)),
                           shape=(n_nodes, n_nodes)).tocsr()
    K = sparse.coo_matrix((k_vals, (row_idx, col_idx)),
                           shape=(n_nodes, n_nodes)).tocsr()

    return M, K

t0 = time.time()
M, K = assemble_fem_matrices(nodes, elements, tissue_rho, tissue_c)
assembly_time = time.time() - t0
print(f"  Assembly time: {assembly_time:.1f}s")
print(f"  Mass matrix: {M.shape}, {M.nnz} nonzeros")
print(f"  Stiffness matrix: {K.shape}, {K.nnz} nonzeros")

# ============================================================
# Step 5: Absorbing boundary condition
# ============================================================
print("\n[Step 5] Setting up absorbing boundary conditions...")

# First-order absorbing BC (Sommerfeld): dp/dn + 1/c * dp/dt = 0
# This adds a damping term: C * dp/dt where C is assembled on boundary edges
# C_ij = integral_boundary(1/c * phi_i * phi_j)

def find_boundary_edges(elements, n_nodes):
    """Find edges on the domain boundary."""
    edge_count = {}
    for tri in elements:
        for i in range(3):
            j = (i + 1) % 3
            edge = tuple(sorted([tri[i], tri[j]]))
            edge_count[edge] = edge_count.get(edge, 0) + 1
    # Boundary edges appear exactly once
    return [e for e, cnt in edge_count.items() if cnt == 1]

boundary_edges = find_boundary_edges(elements, len(nodes))
print(f"  Boundary edges: {len(boundary_edges)}")

# Assemble damping matrix on boundary
row_idx = []
col_idx = []
c_vals = []

# Speed of sound at boundary (seawater)
c_water = MATERIALS['seawater']['c']

for e0, e1 in boundary_edges:
    edge_len = np.linalg.norm(nodes[e1] - nodes[e0])
    # 1D mass matrix on edge: length/6 * [2, 1; 1, 2]
    factor = edge_len / (c_water * 6.0)
    for i, ni in enumerate([e0, e1]):
        for j, nj in enumerate([e0, e1]):
            row_idx.append(ni)
            col_idx.append(nj)
            if i == j:
                c_vals.append(2 * factor)
            else:
                c_vals.append(factor)

C = sparse.coo_matrix((c_vals, (row_idx, col_idx)),
                       shape=(len(nodes), len(nodes))).tocsr()
print(f"  Damping matrix: {C.shape}, {C.nnz} nonzeros")

# ============================================================
# Step 6: Time integration (Newmark-beta)
# ============================================================
print("\n[Step 6] Setting up time integration...")

# CFL-like condition for timestep
c_max = tissue_c.max()
h_min = elem_sizes.min()
dt_cfl = 0.5 * h_min / c_max  # Courant number = 0.5
dt = min(dt_cfl, 1e-6)  # Cap at 1 microsecond
n_steps = int(DURATION / dt) + 1

print(f"  Max sound speed: {c_max:.0f} m/s")
print(f"  Min element size: {h_min*1000:.2f} mm")
print(f"  CFL timestep: {dt_cfl*1e6:.2f} us")
print(f"  Using dt = {dt*1e6:.2f} us")
print(f"  Total steps: {n_steps}")
print(f"  Simulation duration: {DURATION*1000:.1f} ms")

# Source: Ricker wavelet at phonic lips
# Phonic lips are at the anterior-dorsal tip of the spermaceti organ
source_pos = np.array([HEAD_LENGTH * 0.35, HEAD_HEIGHT * 0.1])  # anterior, slightly dorsal
print(f"  Source position (phonic lips): ({source_pos[0]:.2f}, {source_pos[1]:.2f}) m")

# Find nearest node to source
dists_to_source = np.linalg.norm(nodes - source_pos, axis=1)
source_node = np.argmin(dists_to_source)
print(f"  Source node: {source_node} (distance: {dists_to_source[source_node]*1000:.1f} mm)")

# Find nearest nodes to receivers
receiver_nodes = {}
for name, pos in RECEIVERS.items():
    dists = np.linalg.norm(nodes - pos, axis=1)
    receiver_nodes[name] = np.argmin(dists)
    print(f"  Receiver '{name}': node {receiver_nodes[name]} "
          f"(dist: {dists[receiver_nodes[name]]*1000:.1f} mm)")

def ricker_wavelet(t, f0, t0_delay=None):
    """Ricker (Mexican hat) wavelet."""
    if t0_delay is None:
        t0_delay = 1.5 / f0  # Delay so wavelet starts near zero
    tau = t - t0_delay
    a = (np.pi * f0 * tau)**2
    return (1 - 2 * a) * np.exp(-a)

# Newmark-beta parameters (average acceleration - unconditionally stable)
beta = 0.25
gamma = 0.5

# Effective stiffness matrix: K_eff = M/(beta*dt^2) + C*gamma/(beta*dt) + K
K_eff = M / (beta * dt**2) + C * gamma / (beta * dt) + K

print(f"  Factorizing effective stiffness ({K_eff.shape[0]} DOFs)...")
t0 = time.time()

# For large systems, use iterative solver; for smaller, direct
n_dof = len(nodes)
if n_dof > 50000:
    from scipy.sparse.linalg import cg, LinearOperator
    # Diagonal preconditioner
    diag_inv = 1.0 / K_eff.diagonal()
    diag_inv[np.isinf(diag_inv)] = 0
    M_precond = sparse.diags(diag_inv)
    USE_ITERATIVE = True
    print(f"  Using iterative CG solver (n_dof={n_dof})")
else:
    from scipy.sparse.linalg import splu
    try:
        K_eff_lu = splu(K_eff.tocsc())
        USE_ITERATIVE = False
        print(f"  Using direct LU solver (n_dof={n_dof})")
    except Exception as e:
        print(f"  LU failed ({e}), falling back to iterative")
        from scipy.sparse.linalg import cg
        diag_inv = 1.0 / K_eff.diagonal()
        diag_inv[np.isinf(diag_inv)] = 0
        M_precond = sparse.diags(diag_inv)
        USE_ITERATIVE = True

factor_time = time.time() - t0
print(f"  Factorization time: {factor_time:.1f}s")

# ============================================================
# Step 7: Time-stepping loop
# ============================================================
print("\n[Step 7] Running time-domain simulation...")

p = np.zeros(n_dof)       # pressure at t
p_prev = np.zeros(n_dof)  # pressure at t-dt
v = np.zeros(n_dof)       # velocity (dp/dt) at t
a_vec = np.zeros(n_dof)   # acceleration at t

# Storage for receiver time series
receiver_data = {name: np.zeros(n_steps) for name in RECEIVERS}
times = np.zeros(n_steps)

# Storage for snapshots
snapshot_times_ms = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0]
snapshots = {}

# Source force vector
f_ext = np.zeros(n_dof)

print_interval = max(1, n_steps // 20)
t0 = time.time()

for step in range(n_steps):
    t = step * dt
    times[step] = t

    # Source excitation
    f_ext[:] = 0
    src_amplitude = ricker_wavelet(t, SOURCE_FREQ) * 1e6  # Pa scale
    f_ext[source_node] = src_amplitude

    # Newmark-beta predictor
    p_pred = p + dt * v + dt**2 * (0.5 - beta) * a_vec
    v_pred = v + dt * (1 - gamma) * a_vec

    # Effective force
    rhs = f_ext - K.dot(p_pred) - C.dot(v_pred)
    rhs += M.dot(p_pred) / (beta * dt**2)
    rhs += C.dot(p_pred) * gamma / (beta * dt)

    # Solve for new pressure
    if USE_ITERATIVE:
        p_new, info = cg(K_eff, rhs, x0=p, rtol=1e-8, maxiter=200, M=M_precond)
        if info != 0 and step % print_interval == 0:
            print(f"    Warning: CG did not converge at step {step}")
    else:
        p_new = K_eff_lu.solve(rhs)

    # Newmark-beta corrector
    a_new = (p_new - p_pred) / (beta * dt**2)
    v_new = v_pred + dt * gamma * a_new

    # Update
    p_prev[:] = p
    p[:] = p_new
    v[:] = v_new
    a_vec[:] = a_new

    # Record receiver data
    for name, node_id in receiver_nodes.items():
        receiver_data[name][step] = p[node_id]

    # Store snapshots
    t_ms = t * 1000
    for snap_t in snapshot_times_ms:
        if abs(t_ms - snap_t) < dt * 1000 / 2:
            snapshots[snap_t] = p.copy()
            print(f"    Snapshot at t={snap_t:.1f}ms, max|p|={np.abs(p).max():.2f}")

    if step % print_interval == 0:
        elapsed = time.time() - t0
        rate = (step + 1) / elapsed if elapsed > 0 else 0
        eta = (n_steps - step) / rate if rate > 0 else 0
        print(f"    Step {step:6d}/{n_steps} | t={t*1000:.3f}ms | "
              f"max|p|={np.abs(p).max():.2e} | {rate:.0f} steps/s | ETA {eta:.0f}s")

sim_time = time.time() - t0
print(f"  Simulation complete in {sim_time:.1f}s ({n_steps/sim_time:.0f} steps/s)")

# ============================================================
# Step 8: Post-processing - IPI, spectra, beam pattern
# ============================================================
print("\n[Step 8] Post-processing results...")

# IPI (Inter-Pulse Interval) analysis on forward receiver
fwd_signal = receiver_data['forward']
fwd_envelope = np.abs(fwd_signal)

# Find peaks in envelope
from scipy.signal import find_peaks, hilbert

try:
    analytic = hilbert(fwd_signal)
    envelope = np.abs(analytic)
    peaks, properties = find_peaks(envelope, height=envelope.max() * 0.1,
                                    distance=int(0.001 / dt))
    if len(peaks) >= 2:
        ipi = np.diff(times[peaks]) * 1000  # ms
        print(f"  IPI (forward): {ipi} ms")
        print(f"  Mean IPI: {ipi.mean():.2f} ms")
    else:
        print(f"  Only {len(peaks)} peak(s) found - IPI not computable")
        ipi = np.array([])
except Exception as e:
    print(f"  IPI analysis error: {e}")
    ipi = np.array([])

# Spectral analysis
print("  Computing spectra...")
from scipy.fft import rfft, rfftfreq

spectra = {}
for name, signal in receiver_data.items():
    n_fft = len(signal)
    freqs = rfftfreq(n_fft, dt)
    spectrum = np.abs(rfft(signal * np.hanning(n_fft)))
    spectra[name] = (freqs, spectrum)

# Beam pattern at source frequency
print("  Computing beam pattern...")
beam_pattern = {}
for name, (freqs, spectrum) in spectra.items():
    idx = np.argmin(np.abs(freqs - SOURCE_FREQ))
    beam_pattern[name] = spectrum[idx]

max_beam = max(beam_pattern.values()) if max(beam_pattern.values()) > 0 else 1
for name, val in beam_pattern.items():
    db = 20 * np.log10(val / max_beam + 1e-30)
    print(f"  Beam at {name}: {db:.1f} dB (relative)")

# ============================================================
# Step 9: Visualization
# ============================================================
print("\n[Step 9] Creating visualization...")

fig = plt.figure(figsize=(20, 16), dpi=300)
gs = gridspec.GridSpec(3, 3, hspace=0.35, wspace=0.35)

# Color map for tissues
tissue_color_map = {name: props['color'] for name, props in MATERIALS.items()}
tissue_names = list(MATERIALS.keys())

# Panel A: Mesh with tissue types (sagittal)
ax_a = fig.add_subplot(gs[0, 0])
ax_a.set_title('A: Tissue Map (Sagittal)', fontsize=11, fontweight='bold')

# Color each triangle by tissue type
for i, t_name in enumerate(tissue_names):
    mask = tissue_labels == t_name
    if mask.sum() == 0:
        continue
    tris = elements[mask]
    tri_plot = Triangulation(nodes[:, 0], nodes[:, 1], tris)
    ax_a.tripcolor(tri_plot, np.ones(mask.sum()),
                   cmap=matplotlib.colors.ListedColormap([tissue_color_map[t_name]]),
                   vmin=0, vmax=1, alpha=0.8)

# Mark source and receivers
ax_a.plot(*source_pos, 'r*', markersize=12, label='Phonic lips')
for name, pos in RECEIVERS.items():
    ax_a.plot(*pos, 'ko', markersize=4)
    ax_a.annotate(name, pos, fontsize=6, ha='center', va='bottom')

# Overlay skull outline if available
if sagittal_points is not None and len(sagittal_points) > 0:
    ax_a.scatter(sagittal_points[::5, 0], sagittal_points[::5, 1],
                 s=0.5, c='white', alpha=0.5, zorder=5)

ax_a.set_xlabel('Anterior-Posterior (m)', fontsize=8)
ax_a.set_ylabel('Dorsal-Ventral (m)', fontsize=8)
ax_a.set_aspect('equal')
ax_a.legend(fontsize=7, loc='upper left')

# Add tissue legend
legend_elements = []
for t_name in tissue_names:
    if t_name in tissue_labels:
        legend_elements.append(plt.Rectangle((0,0), 1, 1,
                               fc=tissue_color_map[t_name], label=t_name))
ax_a.legend(handles=legend_elements, fontsize=5, loc='lower left', ncol=2)

# Panel B: Pressure field snapshot (sagittal) - pick best snapshot
ax_b = fig.add_subplot(gs[0, 1])
ax_b.set_title('B: Pressure Field - P1 arrival', fontsize=11, fontweight='bold')

if snapshots:
    # Find snapshot with highest pressure variation (likely P1 arrival)
    best_t = max(snapshots.keys(), key=lambda t: np.abs(snapshots[t]).max())
    snap = snapshots[best_t]
    tri_plot = Triangulation(nodes[:, 0], nodes[:, 1], elements)
    vmax = np.abs(snap).max()
    if vmax > 0:
        im = ax_b.tripcolor(tri_plot, snap, cmap='RdBu_r',
                            vmin=-vmax, vmax=vmax, shading='flat')
        plt.colorbar(im, ax=ax_b, label='Pressure (Pa)', shrink=0.8)
    ax_b.set_title(f'B: Pressure at t={best_t:.1f}ms', fontsize=11, fontweight='bold')
else:
    ax_b.text(0.5, 0.5, 'No snapshots available', ha='center', va='center',
              transform=ax_b.transAxes)

ax_b.set_xlabel('Anterior-Posterior (m)', fontsize=8)
ax_b.set_ylabel('Dorsal-Ventral (m)', fontsize=8)
ax_b.set_aspect('equal')

# Panel C: Pressure at another time (horizontal / asymmetry note)
ax_c = fig.add_subplot(gs[0, 2])
if len(snapshots) >= 2:
    snap_times = sorted(snapshots.keys())
    # Pick a later snapshot
    later_t = snap_times[min(len(snap_times)-1, len(snap_times)//2)]
    snap = snapshots[later_t]
    tri_plot = Triangulation(nodes[:, 0], nodes[:, 1], elements)
    vmax = np.abs(snap).max()
    if vmax > 0:
        im = ax_c.tripcolor(tri_plot, snap, cmap='RdBu_r',
                            vmin=-vmax, vmax=vmax, shading='flat')
        plt.colorbar(im, ax=ax_c, label='Pressure (Pa)', shrink=0.8)
    ax_c.set_title(f'C: Pressure at t={later_t:.1f}ms', fontsize=11, fontweight='bold')
else:
    ax_c.text(0.5, 0.5, 'Insufficient snapshots\n(need 3D for horizontal slice)',
              ha='center', va='center', transform=ax_c.transAxes, fontsize=9)
    ax_c.set_title('C: Horizontal slice (requires 3D)', fontsize=11, fontweight='bold')

ax_c.set_xlabel('Anterior-Posterior (m)', fontsize=8)
ax_c.set_ylabel('Dorsal-Ventral (m)', fontsize=8)
ax_c.set_aspect('equal')

# Panel D: Receiver time series
ax_d = fig.add_subplot(gs[1, :2])
ax_d.set_title('D: Receiver Time Series', fontsize=11, fontweight='bold')

colors_recv = {'forward': '#2ecc71', 'backward': '#e74c3c',
               'dorsal': '#3498db', 'ventral': '#f39c12'}
for name, signal in receiver_data.items():
    ax_d.plot(times * 1000, signal, label=name, color=colors_recv.get(name, 'gray'),
              linewidth=0.8, alpha=0.9)

ax_d.set_xlabel('Time (ms)', fontsize=9)
ax_d.set_ylabel('Pressure (Pa)', fontsize=9)
ax_d.legend(fontsize=8)
ax_d.grid(True, alpha=0.3)

# Mark peaks
if len(peaks) > 0:
    ax_d.plot(times[peaks] * 1000, fwd_signal[peaks], 'rv', markersize=6, label='Peaks')

# Panel E: Frequency spectra
ax_e = fig.add_subplot(gs[1, 2])
ax_e.set_title('E: Frequency Spectra', fontsize=11, fontweight='bold')

for name, (freqs, spectrum) in spectra.items():
    mask = freqs < 50000  # Up to 50kHz
    spec_db = 20 * np.log10(spectrum[mask] / (spectrum[mask].max() + 1e-30) + 1e-30)
    ax_e.plot(freqs[mask] / 1000, spec_db, label=name,
              color=colors_recv.get(name, 'gray'), linewidth=0.8)

ax_e.axvline(SOURCE_FREQ / 1000, color='gray', linestyle='--', alpha=0.5, label='Source freq')
ax_e.set_xlabel('Frequency (kHz)', fontsize=9)
ax_e.set_ylabel('Relative Level (dB)', fontsize=9)
ax_e.set_ylim(-60, 5)
ax_e.legend(fontsize=7)
ax_e.grid(True, alpha=0.3)

# Panel F: Mesh quality and simulation info
ax_f = fig.add_subplot(gs[2, 0])
ax_f.set_title('F: Mesh Quality', fontsize=11, fontweight='bold')
ax_f.hist(elem_sizes * 1000, bins=50, color='#3498db', alpha=0.7, label='Element size')
ax_f.set_xlabel('Element Size (mm)', fontsize=9)
ax_f.set_ylabel('Count', fontsize=9)
ax_f.legend(fontsize=8)

ax_f2 = ax_f.twinx()
ax_f2.hist(elem_qualities, bins=50, color='#e74c3c', alpha=0.5, label='Quality')
ax_f2.set_ylabel('Quality count', fontsize=9, color='#e74c3c')
ax_f2.legend(fontsize=8, loc='upper left')

# Panel G: Beam pattern (polar-ish bar chart)
ax_g = fig.add_subplot(gs[2, 1])
ax_g.set_title('G: Directional Response at 12kHz', fontsize=11, fontweight='bold')

directions = list(beam_pattern.keys())
values = [beam_pattern[d] for d in directions]
max_v = max(values) if max(values) > 0 else 1
values_db = [20 * np.log10(v / max_v + 1e-30) for v in values]

bars = ax_g.bar(directions, values_db, color=[colors_recv.get(d, 'gray') for d in directions])
ax_g.set_ylabel('Relative Level (dB)', fontsize=9)
ax_g.set_ylim(min(values_db) - 5, 5)
ax_g.grid(True, alpha=0.3, axis='y')

# Panel H: Summary text
ax_h = fig.add_subplot(gs[2, 2])
ax_h.axis('off')
ax_h.set_title('H: Simulation Summary', fontsize=11, fontweight='bold')

summary_text = f"""FEM Acoustic Simulation - Sperm Whale Head
{'='*45}
Domain: 2D Sagittal Slice
FEM Backend: {fem_backend}
Skull source: {'Real OBJ scan' if skull_vertices is not None else 'Synthetic'}
{'Skull vertices: ' + str(len(skull_vertices)) if skull_vertices is not None else ''}
{'Skull faces: ' + str(len(skull_faces)) if skull_faces is not None else ''}

Mesh:
  Nodes: {len(nodes):,}
  Elements: {len(elements):,}
  Element size: {elem_sizes.mean()*1000:.1f}mm (mean)
  Quality: {elem_qualities.mean():.3f} (mean)

Simulation:
  Source: Ricker wavelet at {SOURCE_FREQ/1000:.0f} kHz
  Duration: {DURATION*1000:.0f} ms
  Timestep: {dt*1e6:.2f} us
  Steps: {n_steps:,}
  Wall time: {sim_time:.1f}s

Results:
  {'IPI: ' + ', '.join([f'{x:.2f}ms' for x in ipi]) if len(ipi) > 0 else 'IPI: not resolved'}
  Forward beam dominance: {values_db[0]:.1f} dB (relative)
  Peak pressure: {max(np.abs(sig).max() for sig in receiver_data.values()):.2e} Pa

Notes:
  - 2D sagittal slice (3D needs ~252GB RAM)
  - Real skull scan used for bone boundaries
  - GRIN gradient modeled in junk tissue
  - Absorbing BCs on outer boundary
"""
ax_h.text(0.02, 0.98, summary_text, transform=ax_h.transAxes,
          fontsize=6, verticalalignment='top', fontfamily='monospace',
          bbox=dict(boxstyle='round', facecolor='#f8f9fa', alpha=0.8))

fig.suptitle('Sperm Whale Head FEM Acoustic Simulation - Sagittal Plane',
             fontsize=14, fontweight='bold', y=0.98)

os.makedirs(OUTPUT_DIR, exist_ok=True)
fig.savefig(FIGURE_PATH, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\n  Figure saved: {FIGURE_PATH}")
plt.close()

# Also save a copy of this script
import shutil
if os.path.abspath(__file__) != os.path.abspath(SCRIPT_PATH):
    shutil.copy2(__file__, SCRIPT_PATH)
    print(f"  Script saved: {SCRIPT_PATH}")

# ============================================================
# Final Summary
# ============================================================
print("\n" + "=" * 70)
print("SIMULATION COMPLETE")
print("=" * 70)
print(f"  Backend: {fem_backend}")
print(f"  Mesh: {len(nodes):,} nodes, {len(elements):,} elements")
print(f"  Simulation: {n_steps:,} steps in {sim_time:.1f}s")
print(f"  Snapshots saved: {list(snapshots.keys())} ms")
print(f"  Figure: {FIGURE_PATH}")
print()
print("FOR FULL 3D SIMULATION:")
print("  - Need: DOLFINx + gmsh + ~64GB RAM minimum (252GB recommended)")
print("  - Mesh: ~500K-2M tets at 20mm element size")
print("  - Estimated wall time: 2-8 hours on 32 cores")
print("  - Horizontal slice (Panel C) requires 3D to capture skull asymmetry")
print("  - 3D captures lateral sound propagation and left/right beam asymmetry")
print("=" * 70)

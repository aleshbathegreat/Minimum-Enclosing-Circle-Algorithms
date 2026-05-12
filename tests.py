# -*- coding: utf-8 -*-
# Force UTF-8 output on Windows
import sys, io
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
"""
tests.py — Correctness + Performance tests for MEC algorithms
  - Welzl (exact, 2D)
  - Skyum (exact, 2D)
  - Approx / Coreset MEB (any dimension)
  - Matoušek–Sharir–Welzl MSW (exact, any dimension)
"""

import os, time, math, random, sys, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless – no display required
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── project imports ──────────────────────────────────────────────────────────
sys.setrecursionlimit(10_000_000)
from welzl_mec  import minimum_enclosing_circle, is_inside
from approx     import meb_approximation
from skynum     import skyum_algo
from HighDWelzl import minimum_enclosing_ball  as msw_minimum_enclosing_ball
from HighDWelzl import minimum_enclosing_circle as msw_minimum_enclosing_circle

warnings.filterwarnings("ignore")   # suppress CVXPY verbosity

# ── output directory ─────────────────────────────────────────────────────────
IMG = "images"
os.makedirs(IMG, exist_ok=True)

# =============================================================================
# CONFIGURATION  ← tweak everything here
# =============================================================================
CFG = dict(
    # --- correctness ---
    EPS_DEFAULT   = 0.01,   # error tolerance used in basic containment checks
    EPS_SWEEP     = [0.1, 0.05, 0.01],   # eps values tested in the sweep

    # --- timing: Welzl vs n ---
    WELZL_SIZES   = [10, 50, 100, 250, 500, 750, 1000, 1500, 2000, 100000],
    WELZL_REPEATS = 5,

    # --- timing: Skyum vs n ---
    SKYUM_SIZES   = [10, 50, 100, 250, 500, 750, 1000, 1500, 2000, 100000],
    SKYUM_REPEATS = 5,

    # --- timing: Approx vs n ---
    APPROX_N_SIZES   = [10, 30, 60, 100, 200, 400, 700, 1000, 100000],
    APPROX_N_EPS     = 0.01,  # fixed eps for the n-scaling run
    APPROX_N_REPEATS = 3,

    # --- timing: Approx vs dimension ---
    APPROX_DIM_DIMS    = [2, 5, 10, 20, 50, 100, 200, 500, 1000],
    APPROX_DIM_N       = 100,   # fixed n for the dimension-scaling run
    APPROX_DIM_EPS     = 0.01,
    APPROX_DIM_REPEATS = 3,

    # --- timing: Approx vs eps ---
    APPROX_EPS_LIST    = [0.2, 0.1, 0.07, 0.05, 0.03, 0.02, 0.01, 0.007, 0.005],
    APPROX_EPS_N       = 200,   # fixed n for the eps-scaling run
    APPROX_EPS_REPEATS = 3,

    # --- comparison Welzl vs Approx ---
    CMP_SIZES   = [10, 50, 100, 250, 500, 750, 1000],
    CMP_EPS     = 0.01,
    CMP_REPEATS = 3,

    # --- MSW vs Approx across dimensions ---
    MSW_DIM_DIMS    = [2, 3, 5, 50, 200],
    MSW_DIM_N       = 100,   # fixed n for dimension scaling
    MSW_DIM_EPS     = 0.01,
    MSW_DIM_REPEATS = 3,

    # --- circle plot ---
    PLOT_EPS = 0.01,
)
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# STYLE
# ─────────────────────────────────────────────────────────────────────────────
DARK   = "#0f1117"
PANEL  = "#1a1d27"
ACCENT = "#7c6af7"
PINK   = "#f06292"
TEAL   = "#26c6da"
LIME   = "#a5d6a7"
GOLD   = "#ffd54f"
WHITE  = "#e8eaf6"
GREY   = "#546e7a"

def _style(fig, axes=None):
    fig.patch.set_facecolor(DARK)
    for ax in (axes if axes is not None else [fig.gca()]):
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=WHITE, labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor(GREY)
        ax.xaxis.label.set_color(WHITE)
        ax.yaxis.label.set_color(WHITE)
        ax.title.set_color(WHITE)
        ax.grid(True, linestyle=":", alpha=0.3, color=GREY)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
PASS = "[PASS]"
FAIL = "[FAIL]"
_results = []

def _check(name, cond, detail=""):
    ok = bool(cond)
    _results.append((name, ok, detail))
    sym = PASS if ok else FAIL
    print(f"  {sym} {name}" + (f"  [{detail}]" if detail else ""))
    return ok

def _section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

def _all_inside_welzl(points, circle, tol=1e-7):
    c, r = circle
    return all(np.linalg.norm(np.array(p) - c) <= r + tol for p in points)

def _all_inside_approx(points_np, center, radius, tol=1e-7):
    dists = np.linalg.norm(points_np - center, axis=1)
    return bool(np.all(dists <= radius + tol))

def _all_inside_skyum(points, center, radius, tol=1e-7):
    return all(np.linalg.norm(np.array(p) - center) <= radius + tol for p in points)

def _all_inside_msw(points, center_tuple, radius, tol=1e-7):
    c = np.array(center_tuple)
    return all(np.linalg.norm(np.array(p) - c) <= radius + tol for p in points)

def approx_2d_to_welzl_compare(pts_list, eps=0.05):
    """Run both algorithms on the same 2-D point set, return radii."""
    pts_np = np.array(pts_list, dtype=float)
    _, r_exact = minimum_enclosing_circle(pts_list)
    _, r_approx, _ = meb_approximation(pts_np, eps)
    return r_exact, r_approx

# ─────────────────────────────────────────────────────────────────────────────
# 1. CORRECTNESS – WELZL
# ─────────────────────────────────────────────────────────────────────────────
def test_welzl_correctness():
    _section("WELZL — Correctness & Edge Cases")

    # 0 points
    circle = minimum_enclosing_circle([])
    _check("0 points → radius 0", circle[1] == 0, str(circle))

    # 1 point
    p = (3.0, 7.0)
    circle = minimum_enclosing_circle([p])
    _check("1 point → center==point, radius 0",
           np.allclose(circle[0], p) and circle[1] == 0, str(circle))

    # 2 points
    a, b = (0.0, 0.0), (4.0, 0.0)
    circle = minimum_enclosing_circle([a, b])
    c, r = circle
    _check("2 points → midpoint center",
           abs(c[0]-2) < 1e-9 and abs(c[1]) < 1e-9, str(c))
    _check("2 points → radius = half-distance", abs(r - 2.0) < 1e-9, f"r={r:.6f}")

    # 3 collinear points
    pts = [(0,0),(1,0),(2,0)]
    circle = minimum_enclosing_circle(pts)
    _check("3 collinear → all inside", _all_inside_welzl(pts, circle), str(circle))
    _check("3 collinear → radius = 1", abs(circle[1]-1.0) < 1e-7, f"r={circle[1]:.6f}")

    # 3 points forming equilateral triangle
    pts = [(0,0),(1,0),(0.5, math.sqrt(3)/2)]
    circle = minimum_enclosing_circle(pts)
    expected_r = 1/math.sqrt(3)
    _check("equilateral triangle → all inside", _all_inside_welzl(pts, circle))
    _check("equilateral triangle → circumradius",
           abs(circle[1] - expected_r) < 1e-6, f"r={circle[1]:.6f} vs {expected_r:.6f}")

    # Points on a circle (known circumradius = 5)
    angles = [i * 2*math.pi/8 for i in range(8)]
    pts = [(5*math.cos(a), 5*math.sin(a)) for a in angles]
    circle = minimum_enclosing_circle(pts)
    _check("8 pts on circle r=5 → all inside", _all_inside_welzl(pts, circle))
    _check("8 pts on circle r=5 → radius≈5", abs(circle[1]-5) < 1e-5, f"r={circle[1]:.6f}")

    # Duplicate points
    pts = [(1,1),(1,1),(1,1)]
    circle = minimum_enclosing_circle(pts)
    _check("duplicate points → radius 0", circle[1] < 1e-9, f"r={circle[1]}")

    # Single cluster far from origin
    pts = [(1000+random.gauss(0,0.1), 1000+random.gauss(0,0.1)) for _ in range(50)]
    random.seed(0)
    circle = minimum_enclosing_circle(pts)
    _check("50 pts far from origin → all inside", _all_inside_welzl(pts, circle))

    # Large random set
    random.seed(42)
    pts = [(random.uniform(-100,100), random.uniform(-100,100)) for _ in range(500)]
    circle = minimum_enclosing_circle(pts)
    _check("500 random pts → all inside", _all_inside_welzl(pts, circle))

# ─────────────────────────────────────────────────────────────────────────────
# 1.5 CORRECTNESS – SKYUM
# ─────────────────────────────────────────────────────────────────────────────
def test_skyum_correctness():
    _section("SKYUM — Correctness & Edge Cases")

    # 1 point
    p = (3.0, 7.0)
    center, radius = skyum_algo([p])
    _check("1 point → center==point, radius 0",
           np.allclose(center, p) and radius == 0, f"c={center}, r={radius}")

    # 2 points
    a, b = (0.0, 0.0), (4.0, 0.0)
    center, radius = skyum_algo([a, b])
    _check("2 points → midpoint center",
           abs(center[0]-2) < 1e-9 and abs(center[1]) < 1e-9, str(center))
    _check("2 points → radius = half-distance", abs(radius - 2.0) < 1e-9, f"r={radius:.6f}")

    # 3 collinear points
    pts = [(0,0),(1,0),(2,0)]
    center, radius = skyum_algo(pts)
    _check("3 collinear → all inside", _all_inside_skyum(pts, center, radius), f"c={center}, r={radius}")
    _check("3 collinear → radius = 1", abs(radius-1.0) < 1e-7, f"r={radius:.6f}")

    # 3 points forming equilateral triangle
    pts = [(0,0),(1,0),(0.5, math.sqrt(3)/2)]
    center, radius = skyum_algo(pts)
    expected_r = 1/math.sqrt(3)
    _check("equilateral triangle → all inside", _all_inside_skyum(pts, center, radius))
    _check("equilateral triangle → circumradius",
           abs(radius - expected_r) < 1e-6, f"r={radius:.6f} vs {expected_r:.6f}")

    # 8 pts on circle r=5
    angles = [i * 2*math.pi/8 for i in range(8)]
    pts = [(5*math.cos(a), 5*math.sin(a)) for a in angles]
    center, radius = skyum_algo(pts)
    _check("8 pts on circle r=5 → all inside", _all_inside_skyum(pts, center, radius))
    _check("8 pts on circle r=5 → radius≈5", abs(radius-5) < 1e-5, f"r={radius:.6f}")

    # Large random set
    random.seed(42)
    pts = [(random.uniform(-100,100), random.uniform(-100,100)) for _ in range(500)]
    center, radius = skyum_algo(pts)
    _check("500 random pts → all inside", _all_inside_skyum(pts, center, radius))

# ─────────────────────────────────────────────────────────────────────────────
# 1.7 CORRECTNESS – MSW (N-Dimensional)
# ─────────────────────────────────────────────────────────────────────────────
def test_msw_correctness():
    _section("MSW (Matou\u0161ek\u2013Sharir\u2013Welzl) — Correctness & Edge Cases")

    # ---- 2-D tests (compare against Welzl) ----
    pts2d = [(0,0),(4,0),(4,4),(0,4)]
    c, r = msw_minimum_enclosing_ball(pts2d, dim=2)
    cw, rw = minimum_enclosing_circle(pts2d)
    _check("2D square → all inside", _all_inside_msw(pts2d, c, r))
    _check("2D square → radius matches Welzl", abs(r - rw) < 1e-4, f"msw={r:.4f} welzl={rw:.4f}")

    pts2d = [(0,0),(1,0),(0.5, math.sqrt(3)/2)]
    c, r = msw_minimum_enclosing_ball(pts2d, dim=2)
    expected_r = 1/math.sqrt(3)
    _check("2D equilateral tri → all inside", _all_inside_msw(pts2d, c, r))
    _check("2D equilateral tri → circumradius", abs(r - expected_r) < 1e-5,
           f"r={r:.5f} vs {expected_r:.5f}")

    pts2d = [(0,0),(3,0),(7,0),(10,0)]
    c, r = msw_minimum_enclosing_ball(pts2d, dim=2)
    _check("2D collinear → all inside", _all_inside_msw(pts2d, c, r))
    _check("2D collinear → radius = 5", abs(r - 5.0) < 1e-5, f"r={r:.5f}")

    random.seed(42)
    pts2d = [(random.uniform(-100,100), random.uniform(-100,100)) for _ in range(300)]
    c, r = msw_minimum_enclosing_ball(pts2d, dim=2)
    cw, rw = minimum_enclosing_circle(pts2d)
    _check("2D 300 random pts → all inside", _all_inside_msw(pts2d, c, r))
    _check("2D 300 random pts → radius matches Welzl", abs(r - rw) < 1e-4,
           f"msw={r:.4f} welzl={rw:.4f}")

    # ---- 3-D tests ----
    cube = [(x,y,z) for x in (0,1) for y in (0,1) for z in (0,1)]
    c, r = msw_minimum_enclosing_ball(cube, dim=3)
    _check("3D unit cube corners → all inside", _all_inside_msw(cube, c, r))
    _check("3D unit cube → radius = sqrt(3)/2",
           abs(r - math.sqrt(3)/2) < 1e-5, f"r={r:.5f}")

    np.random.seed(7)
    pts3d = [tuple(p) for p in np.random.randn(200, 3).tolist()]
    c, r = msw_minimum_enclosing_ball(pts3d, dim=3)
    _check("3D 200 random pts → all inside", _all_inside_msw(pts3d, c, r))

    # ---- 5-D tests ----
    np.random.seed(13)
    pts5d = [tuple(p) for p in np.random.randn(80, 5).tolist()]
    c, r = msw_minimum_enclosing_ball(pts5d, dim=5)
    _check("5D 80 random pts → all inside", _all_inside_msw(pts5d, c, r))
    _check("5D center has 5 components", len(c) == 5, str(len(c)))

    # ---- Approx containment bound check (2D) ----
    np.random.seed(99)
    pts2d_np = np.random.randn(100, 2)
    pts2d_l  = [tuple(p) for p in pts2d_np.tolist()]
    c_msw, r_msw = msw_minimum_enclosing_ball(pts2d_l, dim=2)
    _, r_approx, _ = meb_approximation(pts2d_np, eps=0.01)
    _check("MSW 2D exact ≤ Approx radius",
           r_msw <= r_approx + 1e-4,
           f"msw={r_msw:.4f} approx={r_approx:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. CORRECTNESS – APPROX
# ─────────────────────────────────────────────────────────────────────────────
def test_approx_correctness():
    _section("APPROX — Correctness & Edge Cases")
    EPS = CFG["EPS_DEFAULT"]

    # 1 point (2-D)
    pts = np.array([[3.0, 4.0]])
    c, r, _ = meb_approximation(pts, EPS)
    _check("approx 1 pt → all inside", _all_inside_approx(pts, c, r))

    # 2 points (2-D)
    pts = np.array([[0.0, 0.0],[4.0, 0.0]])
    c, r, _ = meb_approximation(pts, EPS)
    _check("approx 2 pts → all inside", _all_inside_approx(pts, c, r))
    _check("approx 2 pts → r ≥ exact/2", r >= 2.0 - 1e-6, f"r={r:.4f}")

    # 2-D random, containment
    np.random.seed(7)
    pts = np.random.randn(100, 2)
    c, r, _ = meb_approximation(pts, EPS)
    _check("approx 2D 100pts → all inside", _all_inside_approx(pts, c, r))

    # (1+eps) bound vs Welzl exact
    pts_list = list(map(tuple, pts.tolist()))
    _, r_exact = minimum_enclosing_circle(pts_list)
    _check(f"approx 2D → r <= (1+e)*r_exact",
           r <= r_exact*(1+EPS) + 1e-5,
           f"r_approx={r:.4f}  r_exact={r_exact:.4f}  bound={(1+EPS)*r_exact:.4f}")

    # 3-D
    pts = np.random.randn(80, 3)
    c, r, _ = meb_approximation(pts, EPS)
    _check("approx 3D 80pts → all inside", _all_inside_approx(pts, c, r))
    _check("approx 3D → center shape (3,)", c.shape == (3,), str(c.shape))

    # 10-D
    pts = np.random.randn(60, 10)
    c, r, _ = meb_approximation(pts, EPS)
    _check("approx 10D 60pts → all inside", _all_inside_approx(pts, c, r))

    # 100-D
    pts = np.random.randn(50, 100)
    c, r, _ = meb_approximation(pts, EPS)
    _check("approx 100D 50pts → all inside", _all_inside_approx(pts, c, r))

    # 1000-D (from approx.py __main__)
    pts = np.random.rand(100, 1000)
    c, r, _ = meb_approximation(pts, EPS)
    _check("approx 1000D 100pts → all inside", _all_inside_approx(pts, c, r))
    _check("approx 1000D → center shape (1000,)", c.shape == (1000,), str(c.shape))

    # tighter eps
    np.random.seed(99)
    pts2d = np.random.randn(80, 2)
    pts_list = list(map(tuple, pts2d.tolist()))
    _, r_exact = minimum_enclosing_circle(pts_list)
    for eps in CFG["EPS_SWEEP"]:
        c, r, _ = meb_approximation(pts2d, eps)
        ok_cont = _all_inside_approx(pts2d, c, r)
        ok_bound = r <= r_exact*(1+eps) + 1e-4
        _check(f"approx eps={eps} → containment", ok_cont)
        _check(f"approx eps={eps} → (1+e) bound",  ok_bound,
               f"r={r:.4f} bound={(1+eps)*r_exact:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. VISUALISE CIRCLES (2-D)
# ─────────────────────────────────────────────────────────────────────────────
def plot_circles():
    _section("Plotting MEC examples → images/")
    np.random.seed(42)
    random.seed(42)

    configs = [
        ("Small cluster",   np.random.randn(30,2)*2),
        ("Uniform square",  np.random.uniform(-5,5,(60,2))),
        ("Ring of pts",     np.array([(3*math.cos(t),3*math.sin(t))
                                      for t in np.linspace(0,2*math.pi,40,endpoint=False)])),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor(DARK)
    fig.suptitle(f"Minimum Enclosing Circle — Welzl vs Approx (e={CFG['PLOT_EPS']}) vs Skyum",
                 color=WHITE, fontsize=14, fontweight="bold", y=1.02)

    for ax, (title, pts_np) in zip(axes, configs):
        pts_list = [tuple(p) for p in pts_np]
        cw, rw   = minimum_enclosing_circle(pts_list)
        ca, ra, coreset = meb_approximation(pts_np, eps=CFG["PLOT_EPS"])
        cs, rs   = skyum_algo(pts_list)

        ax.set_facecolor(PANEL)
        ax.scatter(pts_np[:,0], pts_np[:,1], color=WHITE,  s=18, alpha=0.7, zorder=3)
        ax.scatter(coreset[:,0], coreset[:,1], color=GOLD, s=60,
                   edgecolors=DARK, linewidths=0.8, zorder=4, label="Coreset")

        circ_exact  = plt.Circle(cw, rw, color=TEAL,  fill=False, lw=2,   linestyle="-",  label=f"Welzl r={rw:.2f}")
        circ_approx = plt.Circle(ca, ra, color=PINK,  fill=False, lw=2,   linestyle="--", label=f"Approx r={ra:.2f}")
        circ_skyum  = plt.Circle(cs, rs, color=LIME,  fill=False, lw=2,   linestyle=":",  label=f"Skyum r={rs:.2f}")
        ax.add_patch(circ_exact)
        ax.add_patch(circ_approx)
        ax.add_patch(circ_skyum)
        ax.plot(*cw, "x", color=TEAL, ms=10, mew=2)
        ax.plot(*ca, "+", color=PINK, ms=10, mew=2)
        ax.plot(*cs, "*", color=LIME, ms=10, mew=2)

        ax.set_aspect("equal")
        ax.set_title(title, color=WHITE, fontsize=10)
        ax.tick_params(colors=WHITE, labelsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor(GREY)
        ax.grid(True, ls=":", alpha=0.25, color=GREY)
        ax.legend(fontsize=7, facecolor=PANEL, labelcolor=WHITE, edgecolor=GREY)

    plt.tight_layout()
    out = f"{IMG}/mec_examples.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. TIMING — Welzl vs n
# ─────────────────────────────────────────────────────────────────────────────
def bench_welzl_vs_n():
    _section("Timing: Welzl vs n  (2-D)")
    sizes   = CFG["WELZL_SIZES"]
    repeats = CFG["WELZL_REPEATS"]
    times   = []

    for n in sizes:
        t_total = 0
        for _ in range(repeats):
            random.seed(random.randint(0, 10**6))
            pts = [(random.uniform(-100,100), random.uniform(-100,100)) for _ in range(n)]
            t0  = time.perf_counter()
            minimum_enclosing_circle(pts)
            t_total += time.perf_counter() - t0
        avg = t_total / repeats
        times.append(avg)
        print(f"    n={n:5d}  avg={avg*1000:.3f} ms")

    fig, ax = plt.subplots(figsize=(9,5))
    _style(fig, [ax])
    ax.plot(sizes, [t*1000 for t in times], color=TEAL, lw=2.5, marker="o",
            markersize=6, markerfacecolor=DARK)
    ax.fill_between(sizes, [t*1000 for t in times], alpha=0.12, color=TEAL)
    ax.set_xlabel("Number of Points (n)")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Welzl MEC — Runtime vs n")
    plt.tight_layout()
    out = f"{IMG}/welzl_timing_vs_n.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")
    return sizes, times

# ─────────────────────────────────────────────────────────────────────────────
# 4.5 TIMING — Skyum vs n
# ─────────────────────────────────────────────────────────────────────────────
def bench_skyum_vs_n():
    _section("Timing: Skyum vs n  (2-D)")
    sizes   = CFG["SKYUM_SIZES"]
    repeats = CFG["SKYUM_REPEATS"]
    times   = []

    for n in sizes:
        t_total = 0
        for _ in range(repeats):
            random.seed(random.randint(0, 10**6))
            pts = [(random.uniform(-100,100), random.uniform(-100,100)) for _ in range(n)]
            t0  = time.perf_counter()
            skyum_algo(pts)
            t_total += time.perf_counter() - t0
        avg = t_total / repeats
        times.append(avg)
        print(f"    n={n:5d}  avg={avg*1000:.3f} ms")

    fig, ax = plt.subplots(figsize=(9,5))
    _style(fig, [ax])
    ax.plot(sizes, [t*1000 for t in times], color=LIME, lw=2.5, marker="o",
            markersize=6, markerfacecolor=DARK)
    ax.fill_between(sizes, [t*1000 for t in times], alpha=0.12, color=LIME)
    ax.set_xlabel("Number of Points (n)")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Skyum MEC — Runtime vs n")
    plt.tight_layout()
    out = f"{IMG}/skyum_timing_vs_n.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")
    return sizes, times

# ─────────────────────────────────────────────────────────────────────────────
# 5. TIMING — Approx vs n  (2-D)
# ─────────────────────────────────────────────────────────────────────────────
def bench_approx_vs_n():
    EPS = CFG["APPROX_N_EPS"]
    _section(f"Timing: Approx vs n  (2-D, e={EPS})")
    sizes   = CFG["APPROX_N_SIZES"]
    repeats = CFG["APPROX_N_REPEATS"]
    times   = []

    for n in sizes:
        t_total = 0
        for s in range(repeats):
            np.random.seed(s)
            pts = np.random.randn(n, 2)
            t0  = time.perf_counter()
            meb_approximation(pts, EPS)
            t_total += time.perf_counter() - t0
        avg = t_total / repeats
        times.append(avg)
        print(f"    n={n:5d}  avg={avg*1000:.3f} ms")

    fig, ax = plt.subplots(figsize=(9,5))
    _style(fig, [ax])
    ax.plot(sizes, [t*1000 for t in times], color=PINK, lw=2.5, marker="s",
            markersize=6, markerfacecolor=DARK)
    ax.fill_between(sizes, [t*1000 for t in times], alpha=0.12, color=PINK)
    ax.set_xlabel("Number of Points (n)")
    ax.set_ylabel("Time (ms)")
    ax.set_title(f"Approx MEB — Runtime vs n  (e={EPS}, d=2)")
    plt.tight_layout()
    out = f"{IMG}/approx_timing_vs_n.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")
    return sizes, times

# ─────────────────────────────────────────────────────────────────────────────
# 6. TIMING — Approx vs dimension
# ─────────────────────────────────────────────────────────────────────────────
def bench_approx_vs_dim():
    EPS = CFG["APPROX_DIM_EPS"]
    N   = CFG["APPROX_DIM_N"]
    _section(f"Timing: Approx vs dimension  (n={N}, e={EPS})")
    dims    = CFG["APPROX_DIM_DIMS"]
    repeats = CFG["APPROX_DIM_REPEATS"]
    times   = []

    for d in dims:
        t_total = 0
        for s in range(repeats):
            np.random.seed(s)
            pts = np.random.randn(N, d)
            t0  = time.perf_counter()
            meb_approximation(pts, EPS)
            t_total += time.perf_counter() - t0
        avg = t_total / repeats
        times.append(avg)
        print(f"    d={d:5d}  avg={avg*1000:.3f} ms")

    fig, ax = plt.subplots(figsize=(9,5))
    _style(fig, [ax])
    ax.plot(dims, [t*1000 for t in times], color=ACCENT, lw=2.5, marker="D",
            markersize=6, markerfacecolor=DARK)
    ax.fill_between(dims, [t*1000 for t in times], alpha=0.12, color=ACCENT)
    ax.set_xlabel("Dimension (d)")
    ax.set_ylabel("Time (ms)")
    ax.set_title(f"Approx MEB — Runtime vs Dimension  (n={N}, e={EPS})")
    plt.tight_layout()
    out = f"{IMG}/approx_timing_vs_dim.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")
    return dims, times

# ─────────────────────────────────────────────────────────────────────────────
# 7. TIMING — Approx vs ε
# ─────────────────────────────────────────────────────────────────────────────
def bench_approx_vs_eps():
    N_eps = CFG["APPROX_EPS_N"]
    _section(f"Timing: Approx vs eps  (n={N_eps}, d=2)")
    epsilons = CFG["APPROX_EPS_LIST"]
    repeats  = CFG["APPROX_EPS_REPEATS"]
    times    = []
    coresizes= []

    np.random.seed(0)
    pts = np.random.randn(N_eps, 2)

    for eps in epsilons:
        t_total = 0
        cs = 0
        for _ in range(repeats):
            t0 = time.perf_counter()
            _, _, coreset = meb_approximation(pts, eps)
            t_total += time.perf_counter() - t0
            cs = len(coreset)
        avg = t_total / repeats
        times.append(avg)
        coresizes.append(cs)
        print(f"    ε={eps:.3f}  avg={avg*1000:.3f} ms  coreset={cs}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    _style(fig, [ax1, ax2])

    ax1.plot(epsilons, [t*1000 for t in times], color=LIME, lw=2.5,
             marker="o", markersize=6, markerfacecolor=DARK)
    ax1.fill_between(epsilons, [t*1000 for t in times], alpha=0.12, color=LIME)
    ax1.invert_xaxis()
    ax1.set_xlabel("ε (smaller = tighter)")
    ax1.set_ylabel("Time (ms)")
    ax1.set_title("Approx MEB — Runtime vs ε")

    ax2.plot(epsilons, coresizes, color=GOLD, lw=2.5,
             marker="s", markersize=6, markerfacecolor=DARK)
    ax2.invert_xaxis()
    ax2.set_xlabel("ε (smaller = tighter)")
    ax2.set_ylabel("Coreset size")
    ax2.set_title("Approx MEB — Coreset Size vs ε")

    plt.tight_layout()
    out = f"{IMG}/approx_timing_vs_eps.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")
    return epsilons, times, coresizes

# ─────────────────────────────────────────────────────────────────────────────
# 8.5 TIMING — MSW vs Approx across dimensions
# ─────────────────────────────────────────────────────────────────────────────
def bench_msw_vs_approx_dim():
    EPS = CFG["MSW_DIM_EPS"]
    N   = CFG["MSW_DIM_N"]
    _section(f"Timing: MSW vs Approx — dimension scaling  (n={N}, e={EPS})")
    dims    = CFG["MSW_DIM_DIMS"]
    repeats = CFG["MSW_DIM_REPEATS"]
    t_msw, t_approx = [], []

    for d in dims:
        tm, ta = 0.0, 0.0
        for s in range(repeats):
            np.random.seed(s)
            pts_np  = np.random.randn(N, d)
            pts_lst = [tuple(p) for p in pts_np.tolist()]

            t0 = time.perf_counter()
            msw_minimum_enclosing_ball(pts_lst, dim=d)
            tm += time.perf_counter() - t0

            t0 = time.perf_counter()
            meb_approximation(pts_np, EPS)
            ta += time.perf_counter() - t0

        t_msw.append(tm / repeats)
        t_approx.append(ta / repeats)
        print(f"    d={d:5d}  MSW={tm/repeats*1000:.2f}ms  Approx={ta/repeats*1000:.2f}ms")

    fig, ax = plt.subplots(figsize=(10, 5))
    _style(fig, [ax])
    ax.plot(dims, [t*1000 for t in t_msw],    color=GOLD,   lw=2.5, marker="o",
            markersize=6, markerfacecolor=DARK, label="MSW (exact)")
    ax.plot(dims, [t*1000 for t in t_approx], color=PINK,   lw=2.5, marker="s",
            markersize=6, markerfacecolor=DARK, label=f"Approx (\u03b5={EPS})")
    ax.fill_between(dims, [t*1000 for t in t_msw],    alpha=0.10, color=GOLD)
    ax.fill_between(dims, [t*1000 for t in t_approx], alpha=0.10, color=PINK)
    ax.set_xlabel("Dimension (d)")
    ax.set_ylabel("Time (ms)")
    ax.set_title(f"MSW vs Approx MEB — Runtime vs Dimension  (n={N}, \u03b5={EPS})")
    legend = ax.legend(facecolor=PANEL, edgecolor=GREY, labelcolor=WHITE)
    plt.tight_layout()
    out = f"{IMG}/msw_vs_approx_dim.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")
    return dims, t_msw, t_approx

# ─────────────────────────────────────────────────────────────────────────────
# 8. COMPARISON — Welzl vs Approx on shared 2-D sets
# ─────────────────────────────────────────────────────────────────────────────
def bench_comparison():
    EPS = CFG["CMP_EPS"]
    _section(f"Comparison: Welzl vs Approx vs Skyum vs MSW  (2-D, e={EPS})")
    sizes   = CFG["CMP_SIZES"]
    repeats = CFG["CMP_REPEATS"]
    t_welzl, t_approx, t_skyum, t_msw = [], [], [], []

    for n in sizes:
        tw, ta, ts, tm = 0, 0, 0, 0
        for s in range(repeats):
            random.seed(s); np.random.seed(s)
            pts_list = [(random.uniform(-50,50), random.uniform(-50,50)) for _ in range(n)]
            pts_np   = np.array(pts_list)

            t0 = time.perf_counter(); minimum_enclosing_circle(pts_list);                               tw += time.perf_counter()-t0
            t0 = time.perf_counter(); meb_approximation(pts_np, EPS);                                  ta += time.perf_counter()-t0
            t0 = time.perf_counter(); skyum_algo(pts_list);                                            ts += time.perf_counter()-t0
            t0 = time.perf_counter(); msw_minimum_enclosing_ball(pts_list, dim=2, method="iterative"); tm += time.perf_counter()-t0

        t_welzl.append(tw/repeats); t_approx.append(ta/repeats)
        t_skyum.append(ts/repeats); t_msw.append(tm/repeats)
        print(f"    n={n:5d}  Welzl={tw/repeats*1000:.2f}ms  Approx={ta/repeats*1000:.2f}ms  Skyum={ts/repeats*1000:.2f}ms  MSW={tm/repeats*1000:.2f}ms")

    fig, ax = plt.subplots(figsize=(11,5))
    _style(fig, [ax])
    ax.plot(sizes, [t*1000 for t in t_welzl],  color=TEAL,  lw=2.5, marker="o", markersize=6, markerfacecolor=DARK, label="Welzl (exact, 2D)")
    ax.plot(sizes, [t*1000 for t in t_approx], color=PINK,  lw=2.5, marker="s", markersize=6, markerfacecolor=DARK, label=f"Approx (e={EPS})")
    ax.plot(sizes, [t*1000 for t in t_skyum],  color=LIME,  lw=2.5, marker="D", markersize=6, markerfacecolor=DARK, label="Skyum (exact, 2D)")
    ax.plot(sizes, [t*1000 for t in t_msw],    color=GOLD,  lw=2.5, marker="^", markersize=6, markerfacecolor=DARK, label="MSW (exact, N-D)")
    ax.fill_between(sizes, [t*1000 for t in t_welzl],  alpha=0.08, color=TEAL)
    ax.fill_between(sizes, [t*1000 for t in t_approx], alpha=0.08, color=PINK)
    ax.fill_between(sizes, [t*1000 for t in t_skyum],  alpha=0.08, color=LIME)
    ax.fill_between(sizes, [t*1000 for t in t_msw],    alpha=0.08, color=GOLD)
    ax.set_xlabel("Number of Points (n)")
    ax.set_ylabel("Time (ms)")
    ax.set_title(f"All Algorithms -- Runtime Comparison (2-D, e={EPS})")
    ax.legend(facecolor=PANEL, edgecolor=GREY, labelcolor=WHITE)
    plt.tight_layout()
    out = f"{IMG}/comparison_all_algorithms.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
def print_summary():
    _section("TEST SUMMARY")
    passed = sum(1 for _, ok, _ in _results if ok)
    total  = len(_results)
    colour = ""
    print(f"\n  {passed}/{total} tests passed")
    if passed < total:
        print("  Failed:")
        for name, ok, detail in _results:
            if not ok:
                print(f"    {FAIL} {name}  {detail}")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_welzl_correctness()
    test_skyum_correctness()
    test_approx_correctness()
    test_msw_correctness()
    plot_circles()
    bench_welzl_vs_n()
    bench_skyum_vs_n()
    bench_approx_vs_n()
    bench_approx_vs_dim()
    bench_approx_vs_eps()
    bench_msw_vs_approx_dim()
    bench_comparison()
    print_summary()
    print(f"\n  All images saved to  ./{IMG}/\n")

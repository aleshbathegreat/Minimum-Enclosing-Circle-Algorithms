import numpy as np
import math
import matplotlib.pyplot as plt
from numba import njit

# ---------------------------------------------------------------------------
# Numba-JIT helpers
# ---------------------------------------------------------------------------

@njit(cache=True)
def _max_dist_sq(pts, cx, cy):
    """Return the squared maximum distance from (cx, cy) to any point in pts."""
    best = 0.0
    for i in range(pts.shape[0]):
        dx = pts[i, 0] - cx
        dy = pts[i, 1] - cy
        d2 = dx * dx + dy * dy
        if d2 > best:
            best = d2
    return best


@njit(cache=True)
def _ternary_search_y(pts, x, low, high, iters=60):
    """Golden-section search for the y that minimises max squared distance at fixed x."""
    for _ in range(iters):
        m1 = low + (high - low) / 3.0
        m2 = high - (high - low) / 3.0
        if _max_dist_sq(pts, x, m1) < _max_dist_sq(pts, x, m2):
            high = m2
        else:
            low = m1
    return (low + high) * 0.5


@njit(cache=True)
def _ternary_search_x(pts, x_low, x_high, y_low, y_high, iters=60):
    """
    Nested ternary search: outer over x, inner over y.
    Returns (cx, cy, max_r²) — the Megiddo-style minimax centre.
    """
    for _ in range(iters):
        m1 = x_low + (x_high - x_low) / 3.0
        m2 = x_high - (x_high - x_low) / 3.0

        y1 = _ternary_search_y(pts, m1, y_low, y_high)
        y2 = _ternary_search_y(pts, m2, y_low, y_high)

        f1 = _max_dist_sq(pts, m1, y1)
        f2 = _max_dist_sq(pts, m2, y2)

        if f1 < f2:
            x_high = m2
        else:
            x_low = m1

    cx = (x_low + x_high) * 0.5
    cy = _ternary_search_y(pts, cx, y_low, y_high)
    return cx, cy, _max_dist_sq(pts, cx, cy)


@njit(cache=True)
def _circumcircle(x1, y1, x2, y2, x3, y3):
    """
    Circumcircle of three non-collinear points.
    Returns (cx, cy, r²).  r² = 1e18 if degenerate.
    """
    D = 2.0 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if math.fabs(D) < 1e-9:
        return 0.0, 0.0, 1e18
    a2 = x1 * x1 + y1 * y1
    b2 = x2 * x2 + y2 * y2
    c2 = x3 * x3 + y3 * y3
    ux = (a2 * (y2 - y3) + b2 * (y3 - y1) + c2 * (y1 - y2)) / D
    uy = (a2 * (x3 - x2) + b2 * (x1 - x3) + c2 * (x2 - x1)) / D
    r2 = (x1 - ux) ** 2 + (y1 - uy) ** 2
    return ux, uy, r2


@njit(cache=True)
def _all_inside(pts, cx, cy, r2, tol=1e-7):
    """True iff every point in pts lies within radius² r2 (+ tol)."""
    for i in range(pts.shape[0]):
        dx = pts[i, 0] - cx
        dy = pts[i, 1] - cy
        if dx * dx + dy * dy > r2 + tol:
            return False
    return True


@njit(cache=True)
def _brute_force_mec(pts):
    """
    Exact O(n³) MEC for small point sets.
    Enumerates all diameter pairs and circumcircles.
    Returns (cx, cy, r²).
    """
    n = pts.shape[0]
    best_r2 = 1e18
    best_cx = 0.0
    best_cy = 0.0

    if n == 0:
        return 0.0, 0.0, 0.0
    if n == 1:
        return pts[0, 0], pts[0, 1], 0.0

    # Diameter pairs
    for i in range(n):
        for j in range(i + 1, n):
            cx = (pts[i, 0] + pts[j, 0]) * 0.5
            cy = (pts[i, 1] + pts[j, 1]) * 0.5
            dx = pts[i, 0] - cx
            dy = pts[i, 1] - cy
            r2 = dx * dx + dy * dy
            if r2 < best_r2 and _all_inside(pts, cx, cy, r2):
                best_r2 = r2
                best_cx = cx
                best_cy = cy

    # Circumcircles of triples
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                cx, cy, r2 = _circumcircle(
                    pts[i, 0], pts[i, 1],
                    pts[j, 0], pts[j, 1],
                    pts[k, 0], pts[k, 1],
                )
                if r2 < best_r2 and _all_inside(pts, cx, cy, r2):
                    best_r2 = r2
                    best_cx = cx
                    best_cy = cy

    return best_cx, best_cy, best_r2


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def solve_mec_megiddo(points):
    """
    Minimum Enclosing Circle via a Megiddo-style prune-and-search.

    For ≤ 3 points the exact brute-force answer is returned.
    For larger sets a nested ternary search (JIT-compiled) locates the
    minimax centre — equivalent to Megiddo's linear-time 1-D search
    applied successively over x and y.

    Parameters
    ----------
    points : array-like of shape (n, 2)

    Returns
    -------
    cx, cy : float   — centre coordinates
    r      : float   — radius
    """
    pts = np.unique(np.asarray(points, dtype=np.float64), axis=0)

    if len(pts) == 0:
        return 0.0, 0.0, 0.0
    if len(pts) <= 3:
        cx, cy, r2 = _brute_force_mec(pts)
        return cx, cy, math.sqrt(r2)

    # Bounding box defines the search domain
    x_low, x_high = pts[:, 0].min(), pts[:, 0].max()
    y_low, y_high = pts[:, 1].min(), pts[:, 1].max()

    cx, cy, r2 = _ternary_search_x(pts, x_low, x_high, y_low, y_high)
    return cx, cy, math.sqrt(r2)


# ---------------------------------------------------------------------------
# Warm-up JIT (called once at import so benchmarks don't pay for compilation)
# ---------------------------------------------------------------------------

def _warmup():
    dummy = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]], dtype=np.float64)
    _brute_force_mec(dummy)
    _ternary_search_x(dummy, 0.0, 1.0, 0.0, 1.0, iters=2)

_warmup()

# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize_mec(points, cx, cy, r, title, filename):
    pts = np.asarray(points)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(pts[:, 0], pts[:, 1], color='royalblue', label='Points', zorder=3)

    circle = plt.Circle((cx, cy), r, color='crimson', fill=False, linewidth=2, label='MEC')
    ax.add_patch(circle)
    ax.plot(cx, cy, 'rx', markersize=10, markeredgewidth=2, label='Center')

    ax.set_aspect('equal', adjustable='datalim')
    plt.legend()
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.6)
    import os
    os.makedirs("images", exist_ok=True)
    plt.savefig(f"images/{filename}.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved images/{filename}.png")


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time
    np.random.seed(42)

    def run_and_plot(points, title, filename):
        t0 = time.perf_counter()
        cx, cy, r = solve_mec_megiddo(points)
        t1 = time.perf_counter()
        print(f"--- {title} ---")
        print(f"MEC Centre : ({cx:.6f}, {cy:.6f})")
        print(f"MEC Radius : {r:.6f}")
        print(f"Runtime    : {(t1-t0)*1000:.3f} ms\n")
        visualize_mec(points, cx, cy, r, title, filename)

    # 1. Wide Coordinate Spread (Stresses the Ternary Search grid)
    # Points clustered near origin, but with extreme outliers at +/- 1,000,000
    tight = np.random.randn(50, 2) * 10
    outliers = np.array([[-1000000, 1000000], [1000000, -1000000], [1000000, 1000000], [-1000000, -1000000]])
    wide_spread = np.vstack([tight, outliers])
    run_and_plot(wide_spread, "Megiddo: Wide Coordinate Spread", "megiddo_wide_spread")

    # 2. Perfectly Collinear Line (Robustness Check)
    # 100 points exactly on a diagonal line. Exact math algorithms often divide by zero here.
    x_vals = np.linspace(-50, 50, 100)
    collinear_line = np.column_stack((x_vals, x_vals))
    run_and_plot(collinear_line, "Megiddo: Perfectly Collinear Points", "megiddo_collinear")
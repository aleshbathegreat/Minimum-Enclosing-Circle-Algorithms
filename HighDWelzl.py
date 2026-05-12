import numpy as np
import random
import sys
import math
from itertools import combinations
from typing import List, Tuple
from numba import njit

EPS = 1e-10
sys.setrecursionlimit(100_000)

@njit
def _solve_system_njit(M_in, b_in):
    n = M_in.shape[0]
    M = np.empty((n, n + 1), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            M[i, j] = M_in[i, j]
        M[i, n] = b_in[i]

    for col in range(n):
        pivot = col
        max_val = abs(M[col, col])
        for row in range(col + 1, n):
            val = abs(M[row, col])
            if val > max_val:
                max_val = val
                pivot = row
                
        if max_val < 1e-10:
            return np.zeros(0, dtype=np.float64), False
            
        if pivot != col:
            for j in range(col, n + 1):
                temp = M[col, j]
                M[col, j] = M[pivot, j]
                M[pivot, j] = temp

        for row in range(col + 1, n):
            f = M[row, col] / M[col, col]
            for j in range(col, n + 1):
                M[row, j] -= f * M[col, j]

    x = np.zeros(n, dtype=np.float64)
    for i in range(n - 1, -1, -1):
        x[i] = M[i, n]
        for j in range(i + 1, n):
            x[i] -= M[i, j] * x[j]
        x[i] /= M[i, i]
        
    return x, True

@njit
def circumsphere_njit(boundary, b_size, dim):
    center = np.zeros(dim, dtype=np.float64)
    if b_size == 0:
        return center, 0.0
    if b_size == 1:
        for i in range(dim):
            center[i] = boundary[0, i]
        return center, 0.0

    if b_size == 2:
        for i in range(dim):
            center[i] = (boundary[0, i] + boundary[1, i]) / 2.0
        r = 0.0
        for i in range(dim):
            r += (center[i] - boundary[0, i]) ** 2
        return center, math.sqrt(r)

    p0 = boundary[0]
    vs = np.empty((b_size - 1, dim), dtype=np.float64)
    for i in range(1, b_size):
        for j in range(dim):
            vs[i - 1, j] = boundary[i, j] - p0[j]

    G = np.dot(vs, vs.T)
    
    rhs = np.zeros(b_size - 1, dtype=np.float64)
    for i in range(b_size - 1):
        s = 0.0
        for j in range(dim):
            s += vs[i, j] ** 2
        rhs[i] = s / 2.0

    alpha, success = _solve_system_njit(G, rhs)

    if not success:
        best_r = -1.0
        for i in range(b_size):
            for j in range(i + 1, b_size):
                d = 0.0
                for k in range(dim):
                    d += (boundary[i, k] - boundary[j, k]) ** 2
                d = math.sqrt(d) / 2.0
                if d > best_r:
                    best_r = d
                    for k in range(dim):
                        center[k] = (boundary[i, k] + boundary[j, k]) / 2.0
        return center, best_r

    center = p0 + np.dot(alpha, vs)
    r = 0.0
    for j in range(dim):
        r += (center[j] - p0[j]) ** 2
    return center, math.sqrt(r)

@njit
def point_in_ball_njit(p, center, radius, dim):
    d = 0.0
    for i in range(dim):
        d += (p[i] - center[i]) ** 2
    d = math.sqrt(d)
    tol = max(1e-10, radius * 1e-7)
    return d <= radius + tol

@njit
def msw_miniball(points_array, points_size, boundary, b_size, dim):
    c, r = circumsphere_njit(boundary, b_size, dim)
    if points_size == 0 or b_size == dim + 1:
        return c, r

    for i in range(points_size):
        p = points_array[i]
        if not point_in_ball_njit(p, c, r, dim):
            for j in range(dim):
                boundary[b_size, j] = p[j]
            c, r = msw_miniball(points_array, i, boundary, b_size + 1, dim)
    return c, r



def dist(a, b):
    return float(np.linalg.norm(np.array(a) - np.array(b)))


def circumsphere(points):
    k = len(points)
    if k == 0:
        return None, 0.0
    if k == 1:
        return tuple(float(x) for x in points[0]), 0.0

    pts = np.array(points, dtype=np.float64)
    p0 = pts[0]

    if k == 2:
        center = (p0 + pts[1]) / 2.0
        return tuple(float(x) for x in center), float(np.linalg.norm(center - p0))

    # Displacement vectors from p0
    vs = pts[1:] - p0

    # Gram matrix and RHS
    G   = vs @ vs.T
    rhs = np.sum(vs ** 2, axis=1) / 2.0

    try:
        alpha = np.linalg.solve(G, rhs)          # fast LAPACK call — O(d^3) but in C
    except np.linalg.LinAlgError:
        # Degenerate (affinely dependent) — fall back to best diameter pair
        best_c, best_r = None, -1.0
        for i in range(k):
            for j in range(i + 1, k):
                c = (pts[i] + pts[j]) / 2.0
                r = float(np.linalg.norm(pts[i] - pts[j])) / 2.0
                if r > best_r:
                    best_c, best_r = tuple(float(x) for x in c), r
        return best_c, best_r

    center = p0 + alpha @ vs
    return tuple(float(x) for x in center), float(np.linalg.norm(center - p0))


def make_ball(basis_pts):
    if len(basis_pts) == 0:
        return None, 0.0
    return circumsphere(basis_pts)


def point_in_ball(p, center, radius):
    if center is None:
        return False
    tol = max(EPS, radius * 1e-7)
    return dist(p, center) <= radius + tol



def violates(x, basis):
    center, radius = make_ball(basis)
    return not point_in_ball(x, center, radius)


def compute_basis(W, x, dim):
    """Find the new basis of W ∪ {x} knowing x must be on the boundary.
    Enumerates all subsets containing x, up to size dim+1."""
    all_pts = list(W) + [x]
    best_basis = [x]
    best_radius = float('inf')

    c, r = circumsphere([x])
    if all(point_in_ball(p, c, r) for p in all_pts):
        return [x]

    for size in range(1, min(len(W), dim) + 1):
        for combo in combinations(W, size):
            subset = [x] + list(combo)
            c, r = circumsphere(subset)
            if c is not None and all(point_in_ball(p, c, r) for p in all_pts):
                if r < best_radius:
                    best_radius = r
                    best_basis = subset
    return best_basis




def _msw_recursive(U, V, dim):
    # The second recursive call (on violation) is a tail call,
    # so we convert it to a loop.  Only the first call is true recursion.
    while True:
        if len(U) <= len(V):
            V_set = set(V)
            if set(U) <= V_set:
                return V

        V_set = set(V)
        candidates = [p for p in U if p not in V_set]
        if not candidates:
            return V

        x = random.choice(candidates)

        removed = False
        U_minus_x = []
        for p in U:
            if not removed and p is x:
                removed = True
                continue
            U_minus_x.append(p)

        W = _msw_recursive(U_minus_x, V, dim)

        if violates(x, W):
            V = compute_basis(W, x, dim)
            continue
        else:
            return W




def _miniball_with_boundary(pts, n, boundary, dim):

    if n == 0 or len(boundary) == dim + 1:
        return make_ball(boundary)

    p = pts[n - 1]
    center, radius = _miniball_with_boundary(pts, n - 1, boundary, dim)

    if point_in_ball(p, center, radius):
        return center, radius

    return _miniball_with_boundary(pts, n - 1, boundary + [p], dim)


def _msw_iterative(points, dim):
    pts = list(points)
    n = len(pts)
    if n == 0:
        return tuple([0.0]*dim), 0.0
    if n <= 1:
        return circumsphere(pts)

    random.shuffle(pts)
    pts_arr = np.array(pts, dtype=np.float64)
    boundary = np.zeros((dim + 1, dim), dtype=np.float64)
    
    c, r = msw_miniball(pts_arr, n, boundary, 0, dim)
    return tuple(float(x) for x in c), r




def minimum_enclosing_ball(points, dim=None):
    pts = list(set(points))
    n = len(pts)

    if n == 0:
        d = dim if dim else 2
        return tuple([0.0] * d), 0.0

    if dim is None:
        dim = len(pts[0])

    if n <= dim + 1:
        return make_ball(pts)

    # Always use the JIT-compiled iterative path (no recursive)
    return _msw_iterative(pts, dim)


# Backward-compatible wrapper for 2-D callers
def minimum_enclosing_circle(points):
    return minimum_enclosing_ball(points, dim=2)




def welzl_reference(points, dim=None):
    pts = list(set(points))
    random.shuffle(pts)
    n = len(pts)
    if n == 0:
        return (0.0, 0.0), 0.0
    if dim is None:
        dim = len(pts[0])
    if n <= dim + 1:
        return make_ball(pts)

    def _welzl(P, R):
        if len(P) == 0 or len(R) == dim + 1:
            return make_ball(R)
        p = P[0]
        rest = P[1:]
        center, radius = _welzl(rest, R)
        if point_in_ball(p, center, radius):
            return center, radius
        return _welzl(rest, R + [p])

    # For large n, use the iterative nested-loop approach
    if n > 200:
        return _miniball_with_boundary(pts, n, [], dim)

    return _welzl(pts, [])



if __name__ == "__main__":
    import math
    import time

    print("=" * 65)
    print("  MSW Algorithm — Minimum Enclosing Ball (any dimension)")
    print("=" * 65)

    # ---- 2-D tests ----
    tests_2d = {
        "Square":          [(0,0), (4,0), (4,4), (0,4)],
        "Equilateral tri": [(0,0), (6,0), (3, 3*math.sqrt(3))],
        "Collinear":       [(0,0), (3,0), (7,0), (10,0)],
        "Cluster+outlier": [(0,0),(1,0),(0,1),(1,1),(10,0)],
        "Random 100 (2D)": [(random.uniform(-50,50),
                             random.uniform(-50,50)) for _ in range(100)],
    }

    for name, pts in tests_2d.items():
        t0 = time.perf_counter()
        c, r = minimum_enclosing_ball(pts, dim=2)
        dt = time.perf_counter() - t0
        c_ref, r_ref = welzl_reference(pts, dim=2)
        ok = "PASS" if abs(r - r_ref) < 1e-4 else "FAIL"
        print(f"\n[2D] {name} (n={len(pts)})  [{ok}]")
        print(f"  center={tuple(round(x,4) for x in c)}  r={r:.4f}  ({dt*1000:.2f} ms)")

    # ---- 3-D tests ----
    tests_3d = {
        "Cube corners": [(0,0,0),(1,0,0),(0,1,0),(0,0,1),
                         (1,1,0),(1,0,1),(0,1,1),(1,1,1)],
        "Random 200 (3D)": [(random.uniform(-50,50),
                             random.uniform(-50,50),
                             random.uniform(-50,50)) for _ in range(200)],
    }

    for name, pts in tests_3d.items():
        t0 = time.perf_counter()
        c, r = minimum_enclosing_ball(pts, dim=3)
        dt = time.perf_counter() - t0
        c_ref, r_ref = welzl_reference(pts, dim=3)
        ok = "PASS" if abs(r - r_ref) < 1e-4 else "FAIL"
        print(f"\n[3D] {name} (n={len(pts)})  [{ok}]")
        print(f"  center={tuple(round(x,4) for x in c)}  r={r:.4f}  ({dt*1000:.2f} ms)")

    # ---- 5-D test ----
    pts_5d = [tuple(random.uniform(-10,10) for _ in range(5)) for _ in range(80)]
    t0 = time.perf_counter()
    c5, r5 = minimum_enclosing_ball(pts_5d, dim=5)
    dt5 = time.perf_counter() - t0
    c5r, r5r = welzl_reference(pts_5d, dim=5)
    ok5 = "PASS" if abs(r5 - r5r) < 1e-3 else "FAIL"
    print(f"\n[5D] Random 80 (n=80)  [{ok5}]")
    print(f"  radius={r5:.4f}  ref={r5r:.4f}  ({dt5*1000:.2f} ms)")

    print("\n" + "=" * 65)
    print("  Done.")
    print("=" * 65)

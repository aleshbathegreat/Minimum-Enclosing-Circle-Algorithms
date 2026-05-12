import math
import numpy as np
from numba import njit

@njit
def partition(arr, low, high):
    pivot_x = arr[high, 0]
    pivot_y = arr[high, 1]
    i = low - 1
    for j in range(low, high):
        if arr[j, 0] < pivot_x or (arr[j, 0] == pivot_x and arr[j, 1] < pivot_y):
            i += 1
            tx, ty = arr[i, 0], arr[i, 1]
            arr[i, 0], arr[i, 1] = arr[j, 0], arr[j, 1]
            arr[j, 0], arr[j, 1] = tx, ty
            
    i += 1
    tx, ty = arr[i, 0], arr[i, 1]
    arr[i, 0], arr[i, 1] = arr[high, 0], arr[high, 1]
    arr[high, 0], arr[high, 1] = tx, ty
    return i

@njit
def quicksort(arr, low, high):
    if low < high:
        pi = partition(arr, low, high)
        quicksort(arr, low, pi - 1)
        quicksort(arr, pi + 1, high)

@njit
def lex_sort_2d(arr):
    np.random.shuffle(arr)
    quicksort(arr, 0, len(arr) - 1)

@njit
def remove_duplicates(arr):
    if len(arr) == 0:
        return arr
    n = len(arr)
    unique_count = 1
    for i in range(1, n):
        if arr[i, 0] != arr[i-1, 0] or arr[i, 1] != arr[i-1, 1]:
            arr[unique_count, 0] = arr[i, 0]
            arr[unique_count, 1] = arr[i, 1]
            unique_count += 1
    return arr[:unique_count]

@njit
def cross_2d(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

@njit
def monotone_chain_hull(points_array):
    n = len(points_array)
    if n < 3:
        res = np.zeros((n, 2), dtype=np.float64)
        for i in range(n):
            res[i] = points_array[i]
        return res

    lex_sort_2d(points_array)
    points_array = remove_duplicates(points_array)
    n = len(points_array)
    
    if n < 3:
        return points_array

    lower = np.zeros((n, 2), dtype=np.float64)
    l_size = 0
    for i in range(n):
        while l_size >= 2 and cross_2d(lower[l_size-2], lower[l_size-1], points_array[i]) <= 0:
            l_size -= 1
        lower[l_size] = points_array[i]
        l_size += 1

    upper = np.zeros((n, 2), dtype=np.float64)
    u_size = 0
    for i in range(n - 1, -1, -1):
        while u_size >= 2 and cross_2d(upper[u_size-2], upper[u_size-1], points_array[i]) <= 0:
            u_size -= 1
        upper[u_size] = points_array[i]
        u_size += 1

    hull_size = l_size - 1 + u_size - 1
    hull = np.zeros((hull_size, 2), dtype=np.float64)
    idx = 0
    for i in range(l_size - 1):
        hull[idx] = lower[i]
        idx += 1
    for i in range(u_size - 1):
        hull[idx] = upper[i]
        idx += 1
        
    return hull

@njit
def get_metrics_numba(prev_px, prev_py, px, py, next_px, next_py):
    bax = prev_px - px
    bay = prev_py - py
    bcx = next_px - px
    bcy = next_py - py
    acx = next_px - prev_px
    acy = next_py - prev_py
    
    dot = bax*bcx + bay*bcy
    norm_ba = math.hypot(bax, bay)
    norm_bc = math.hypot(bcx, bcy)
    
    if norm_ba == 0 or norm_bc == 0:
        cosine_angle = 1.0
    else:
        cosine_angle = dot / (norm_ba * norm_bc)
    
    cosine_angle = max(-1.0, min(1.0, cosine_angle))
    angle = math.acos(cosine_angle)
    
    a = norm_bc
    b = math.hypot(acx, acy)
    c = norm_ba
    area = 0.5 * abs(bax*bcy - bay*bcx)
    
    if area == 0:
        radius = np.inf
    else:
        radius = (a * b * c) / (4.0 * area)
    return radius, angle

@njit
def get_circumcircle_numba(x1, y1, x2, y2, x3, y3):
    D = 2.0 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if D == 0:
        return 0.0, 0.0, np.inf
    ux = ((x1**2 + y1**2) * (y2 - y3) + (x2**2 + y2**2) * (y3 - y1) + (x3**2 + y3**2) * (y1 - y2)) / D
    uy = ((x1**2 + y1**2) * (x3 - x2) + (x2**2 + y2**2) * (x1 - x3) + (x3**2 + y3**2) * (x2 - x1)) / D
    radius = math.hypot(x1 - ux, y1 - uy)
    return ux, uy, radius

@njit
def skyum_core(hull):
    n = len(hull)
    if n == 1:
        return hull[0][0], hull[0][1], 0.0
    if n == 2:
        cx = (hull[0][0] + hull[1][0]) / 2.0
        cy = (hull[0][1] + hull[1][1]) / 2.0
        r = math.hypot(hull[0][0] - hull[1][0], hull[0][1] - hull[1][1]) / 2.0
        return cx, cy, r

    S_x = np.zeros(n, dtype=np.float64)
    S_y = np.zeros(n, dtype=np.float64)
    for i in range(n):
        S_x[i] = hull[i][0]
        S_y[i] = hull[i][1]

    metrics_r = np.zeros(n, dtype=np.float64)
    metrics_a = np.zeros(n, dtype=np.float64)
    
    n_S = n
    for i in range(n_S):
        prv = (i - 1) % n_S
        nxt = (i + 1) % n_S
        r, a = get_metrics_numba(S_x[prv], S_y[prv], S_x[i], S_y[i], S_x[nxt], S_y[nxt])
        metrics_r[i] = r
        metrics_a[i] = a

    finished = False
    while not finished and n_S > 2:
        best_idx = -1
        max_r = -1.0
        max_a = -1.0
        for i in range(n_S):
            r = metrics_r[i]
            a = metrics_a[i]
            if r > max_r or (r == max_r and a > max_a):
                max_r = r
                max_a = a
                best_idx = i
                
        if max_a <= math.pi / 2.0:
            finished = True
        else:
            for i in range(best_idx, n_S - 1):
                S_x[i] = S_x[i+1]
                S_y[i] = S_y[i+1]
                metrics_r[i] = metrics_r[i+1]
                metrics_a[i] = metrics_a[i+1]
            n_S -= 1
            
            prv = (best_idx - 1) % n_S
            curr = best_idx % n_S
            
            pprv = (prv - 1) % n_S
            nxt = (prv + 1) % n_S
            r, a = get_metrics_numba(S_x[pprv], S_y[pprv], S_x[prv], S_y[prv], S_x[nxt], S_y[nxt])
            metrics_r[prv] = r
            metrics_a[prv] = a
            
            pprv2 = (curr - 1) % n_S
            nxt2 = (curr + 1) % n_S
            r, a = get_metrics_numba(S_x[pprv2], S_y[pprv2], S_x[curr], S_y[curr], S_x[nxt2], S_y[nxt2])
            metrics_r[curr] = r
            metrics_a[curr] = a

    if n_S == 2:
        cx = (S_x[0] + S_x[1]) / 2.0
        cy = (S_y[0] + S_y[1]) / 2.0
        r = math.hypot(S_x[0] - S_x[1], S_y[0] - S_y[1]) / 2.0
        return cx, cy, r
    else:
        cx, cy, r = get_circumcircle_numba(S_x[0], S_y[0], S_x[1], S_y[1], S_x[2], S_y[2])
        return cx, cy, r

def convex_hull(points):
    if len(points) == 0:
        return []
    pts = np.array(points, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(-1, 2)
    hull = monotone_chain_hull(pts)
    return [(p[0], p[1]) for p in hull]

def skyum_algo(points):
    if len(points) == 0:
        return ((0.0, 0.0), 0.0)
    pts = np.array(points, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(-1, 2)
    hull = monotone_chain_hull(pts)
    cx, cy, r = skyum_core(hull)
    return (cx, cy), r

import matplotlib.pyplot as plt
import numpy as np

def main():
    # 1. Generate random points
    np.random.seed(42) # For reproducibility
    num_points = 200
    points = np.random.rand(num_points, 2) * 100 

    # 2. Run Skyum Algorithm
    # Note: Make sure your skyum_algo returns (center_tuple, radius)
    center, radius = skyum_algo(points)
    hull = np.array(convex_hull([tuple(p) for p in points]))

    # 3. Setup Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal') # Crucial for circles to look like circles

    # Plot original points
    ax.scatter(points[:, 0], points[:, 1], color='gray', alpha=0.5, label='Points')
    
    # Plot Convex Hull
    hull_plot = np.vstack([hull, hull[0]]) # Close the loop
    ax.plot(hull_plot[:, 0], hull_plot[:, 1], 'r--', alpha=0.7, label='Convex Hull')

    # Plot the Smallest Enclosing Circle
    circle_patch = plt.Circle(center, radius, color='blue', fill=False, linewidth=2, label='SEC')
    ax.add_patch(circle_patch)
    
    # Plot the center
    ax.scatter([center[0]], [center[1]], color='blue', marker='x', s=100, label='Center')

    # Formatting
    plt.title(f"Skyum's Algorithm: Smallest Enclosing Circle\nRadius: {radius:.2f}")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Adjust limits so circle isn't cut off
    margin = radius * 0.2
    ax.set_xlim(center[0] - radius - margin, center[0] + radius + margin)
    ax.set_ylim(center[1] - radius - margin, center[1] + radius + margin)

    plt.show()

if __name__ == "__main__":
    main()
import random
import time
import numpy as np
import matplotlib.pyplot as plt
import math
from numba import njit

@njit
def circle_from_2_points(p1, p2):
    center = ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)
    radius = math.hypot(p1[0] - p2[0], p1[1] - p2[1]) / 2.0
    return center, radius

@njit
def circle_from_3_points(p1, p2, p3):
    ax, ay = p1
    bx, by = p2
    cx, cy = p3

    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-9:
        return (0.0, 0.0), -1.0

    a2 = ax**2 + ay**2
    b2 = bx**2 + by**2
    c2 = cx**2 + cy**2

    ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d
    uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d

    center = (ux, uy)
    radius = math.hypot(p1[0] - ux, p1[1] - uy)
    return center, radius

@njit
def is_inside(center, radius, p):
    return math.hypot(p[0] - center[0], p[1] - center[1]) <= radius + 1e-9

@njit
def trivial_circle(boundary, b_size):
    if b_size == 0:
        return (0.0, 0.0), 0.0
    if b_size == 1:
        return (boundary[0][0], boundary[0][1]), 0.0
    if b_size == 2:
        return circle_from_2_points(boundary[0], boundary[1])

    # try all pairs first
    for i in range(3):
        for j in range(i + 1, 3):
            c, r = circle_from_2_points(boundary[i], boundary[j])
            valid = True
            for k in range(3):
                if not is_inside(c, r, boundary[k]):
                    valid = False
                    break
            if valid:
                return c, r

    return circle_from_3_points(boundary[0], boundary[1], boundary[2])

@njit
def welzl(points_array, points_size, boundary, b_size):
    c, r = trivial_circle(boundary, b_size)
    if points_size == 0 or b_size == 3:
        return c, r

    for i in range(points_size):
        p = points_array[i]
        if not is_inside(c, r, p):
            boundary[b_size][0] = p[0]
            boundary[b_size][1] = p[1]
            c, r = welzl(points_array, i, boundary, b_size + 1)
    return c, r

def minimum_enclosing_circle(points):
    if len(points) == 0:
        return ((0.0, 0.0), 0.0)
    pts = np.array(points, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(-1, 2)
    np.random.shuffle(pts)
    boundary = np.zeros((3, 2), dtype=np.float64)
    return welzl(pts, len(pts), boundary, 0)


def plot_circle(points, center, radius):
    pts = np.array(points)
    fig, ax = plt.subplots()
    ax.scatter(pts[:, 0], pts[:, 1], label="Input points")
    ax.add_patch(plt.Circle(center, radius, fill=False, linewidth=2, label="MEC"))
    ax.scatter(*center, marker="x", s=100, label="Center")
    ax.set_aspect("equal")
    ax.set_title("Minimum enclosing circle — Welzl (numpy)")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.6)
    padding = radius * 0.3 + 1
    ax.set_xlim(center[0] - radius - padding, center[0] + radius + padding)
    ax.set_ylim(center[1] - radius - padding, center[1] + radius + padding)
    import os
    os.makedirs("images", exist_ok=True)
    plt.savefig("images/welzl_average_case.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved images/welzl_average_case.png")

if __name__ == "__main__":
    np.random.seed(42)
    number_of_points = 100

    points = np.random.randn(number_of_points, 2) * 20

    start = time.perf_counter()
    center, radius = minimum_enclosing_circle(points)
    elapsed = time.perf_counter() - start

    print(f"Center: {center}")
    print(f"Radius: {radius:.4f}")
    print(f"Runtime: {elapsed*1000:.2f}ms")

    plot_circle(points, center, radius)
import sys
import tracemalloc
import time
import random
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Increase recursion depth for Welzl and MSW on large inputs
sys.setrecursionlimit(10_000_000)

from welzl_mec import minimum_enclosing_circle as welzl_algo
from approx import meb_approximation as approx_algo
from skyum import skyum_algo
from HighDWelzl import minimum_enclosing_ball as msw_algo
from megiddo import solve_mec_megiddo as megiddo_algo

DARK   = "#0f1117"
PANEL  = "#1a1d27"
ACCENT = "#7c6af7"
PINK   = "#f06292"
TEAL   = "#26c6da"
LIME   = "#a5d6a7"
GOLD   = "#ffd54f"
WHITE  = "#e8eaf6"
GREY   = "#546e7a"

def measure_peak_memory(func, args):
    tracemalloc.start()
    try:
        func(*args)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak / 1024.0  # Return in KB

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

def plot_individual(sizes, mem_data, name, color, filename):
    fig, ax = plt.subplots(figsize=(10, 6))
    _style(fig, [ax])
    ax.plot(sizes, mem_data, color=color, lw=2.5, marker="o", markersize=6, markerfacecolor=DARK, label=name)
    ax.set_xlabel("Number of Points (n)")
    ax.set_ylabel("Peak Memory (KB)")
    ax.set_title(f"{name} Space Complexity -- Peak Memory vs n")
    ax.legend(facecolor=PANEL, edgecolor=GREY, labelcolor=WHITE)
    plt.tight_layout()
    plt.savefig(f"images/{filename}.png", dpi=150, bbox_inches="tight")
    plt.close()

def warmup_jit():
    # Warm up JIT compilers so tracemalloc doesn't track their memory overhead
    dummy_pts = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.5, 0.5)]
    dummy_np = np.array(dummy_pts)
    welzl_algo(dummy_pts)
    approx_algo(dummy_np, 0.1)
    skyum_algo(dummy_pts)
    msw_algo(dummy_pts, 2)
    megiddo_algo(dummy_pts)

def main():
    os.makedirs("images", exist_ok=True)
    warmup_jit()
    sizes = [0, 1, 100, 1000, 10000, 100000]
    repeats = 3
    
    mem_welzl = []
    mem_approx = []
    mem_skyum = []
    mem_msw = []
    mem_megiddo = []

    for n in sizes:
        w_peak, a_peak, s_peak, msw_peak, meg_peak = 0, 0, 0, 0, 0
        for r in range(repeats):
            random.seed(r)
            np.random.seed(r)
            pts_list = [(random.uniform(-50, 50), random.uniform(-50, 50)) for _ in range(n)]
            pts_np = np.array(pts_list).reshape(-1, 2)

            # Welzl
            w_peak += measure_peak_memory(welzl_algo, (pts_list,))
            # Approx
            a_peak += measure_peak_memory(approx_algo, (pts_np, 0.01))
            # Skyum
            s_peak += measure_peak_memory(skyum_algo, (pts_list,))
            # MSW
            msw_peak += measure_peak_memory(msw_algo, (pts_list, 2))
            # Megiddo
            meg_peak += measure_peak_memory(megiddo_algo, (pts_list,))

        mem_welzl.append(w_peak / repeats)
        mem_approx.append(a_peak / repeats)
        mem_skyum.append(s_peak / repeats)
        mem_msw.append(msw_peak / repeats)
        mem_megiddo.append(meg_peak / repeats)

        print(f"n={n:6d} | Welzl: {w_peak/repeats:7.1f} KB | Approx: {a_peak/repeats:7.1f} KB | Skyum: {s_peak/repeats:7.1f} KB | MSW: {msw_peak/repeats:7.1f} KB | Megiddo: {meg_peak/repeats:7.1f} KB")

    # Plot individual
    plot_individual(sizes, mem_welzl, "Welzl", TEAL, "space_complexity_welzl")
    plot_individual(sizes, mem_approx, "Approx", PINK, "space_complexity_approx")
    plot_individual(sizes, mem_skyum, "Skyum", LIME, "space_complexity_skyum")
    plot_individual(sizes, mem_msw, "MSW", GOLD, "space_complexity_msw")
    plot_individual(sizes, mem_megiddo, "Megiddo", ACCENT, "space_complexity_megiddo")

    # Plot Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    _style(fig, [ax])
    
    ax.plot(sizes, mem_welzl, color=TEAL, lw=2.5, marker="o", markersize=6, markerfacecolor=DARK, label="Welzl")
    ax.plot(sizes, mem_approx, color=PINK, lw=2.5, marker="s", markersize=6, markerfacecolor=DARK, label="Approx")
    ax.plot(sizes, mem_skyum, color=LIME, lw=2.5, marker="D", markersize=6, markerfacecolor=DARK, label="Skyum")
    ax.plot(sizes, mem_msw, color=GOLD, lw=2.5, marker="^", markersize=6, markerfacecolor=DARK, label="MSW")
    ax.plot(sizes, mem_megiddo, color=ACCENT, lw=2.5, marker="p", markersize=6, markerfacecolor=DARK, label="Megiddo")

    ax.set_xlabel("Number of Points (n)")
    ax.set_ylabel("Peak Memory (KB)")
    ax.set_title("Algorithms Space Complexity -- Peak Memory vs n")
    ax.legend(facecolor=PANEL, edgecolor=GREY, labelcolor=WHITE)
    plt.tight_layout()
    out = "images/space_complexity_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved space complexity graphs to images/")

if __name__ == "__main__":
    main()

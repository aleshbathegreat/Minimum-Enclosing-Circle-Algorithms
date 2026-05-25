import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

def diameter_approximation(points):
    p = points[np.random.choice(points.shape[0])]
    q = points[np.argmax(np.sum((points - p) ** 2, axis=1))]
    q_ = points[np.argmax(np.sum((points - q) ** 2, axis=1))]
    return q, q_

def meb_approximation(points, eps):
    if len(points) == 0:
        return np.zeros(2), 0.0, np.empty((0, 2))
    if len(points) == 1:
        return points[0], 0.0, points
    
    d = points.shape[1]
    q, q_ = diameter_approximation(points)
    X = np.array([q, q_])
    delta = (eps ** 2) / 163  

    while True:
        c = cp.Variable(d)
        r = cp.Variable()
        constraints = [cp.norm(X - c[None, :], axis=1) <= r]
        prob = cp.Problem(cp.Minimize(r), constraints)
        prob.solve(solver=cp.CLARABEL)
        
        c_val, r_opt = c.value, r.value
        r_approx = r_opt * (1 + delta)
        expanded_r = (1 + eps / 2) * r_approx
        
        dists = np.linalg.norm(points - c_val, axis=1)
        farthest_idx = np.argmax(dists)
        
        if dists[farthest_idx] <= expanded_r + 1e-8:
            return c_val, expanded_r, X

        X = np.vstack([X, points[farthest_idx]])
if __name__ == "__main__":
    import os
    os.makedirs("images", exist_ok=True)
    
    def plot_scenario(points, eps, title, filename):
        np.random.seed(42)
        import time
        t0 = time.perf_counter()
        center, radius, coreset = meb_approximation(points, eps=eps)
        t1 = time.perf_counter()
        
        plt.figure(figsize=(8, 8))
        plt.scatter(points[:, 0], points[:, 1], c='gray', alpha=0.5, label='Input Points $S$')
        plt.scatter(coreset[:, 0], coreset[:, 1], c='red', edgecolors='black', s=80, label='Coreset $X$')
        
        circle = plt.Circle(center, radius, color='blue', fill=False, linewidth=2, label='Approx MEB')
        plt.gca().add_patch(circle)
        plt.plot(center[0], center[1], 'bx', markersize=10, label='Center')
        
        plt.axhline(0, color='black', lw=0.5, ls='--')
        plt.axvline(0, color='black', lw=0.5, ls='--')
        plt.title(f"{title}\nCoreset Size: {len(coreset)} | Time: {(t1-t0)*1000:.1f}ms")
        plt.legend(loc='upper right')
        plt.axis('equal')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.savefig(f"images/{filename}.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved {filename}.png | Coreset size: {len(coreset)} | Time: {(t1-t0)*1000:.1f}ms")

    eps_test = 0.01
    
    # 1. Best Case (Massive Outliers)
    # 95 tightly packed points at origin, 5 extreme outliers
    tight = np.random.randn(95, 2) * 2
    outliers = np.array([[-50, 50], [50, 50], [50, -50], [-50, -50], [0, 60]])
    best_case_pts = np.vstack([tight, outliers])
    plot_scenario(best_case_pts, eps_test, "Best Case: Dense Center with Outliers", "approx_best_case")

    # 2. Worst Case 1: The Bumpy Circle (Adversarial)
    # Points clustered on a circle but slightly asymmetrical to trick the diameter guess
    angles = np.linspace(0, np.pi * 1.5, 100) # Only 3/4ths of a circle
    worst_case_circle = np.column_stack((np.cos(angles)*50, np.sin(angles)*50))
    # Add noise to prevent perfect symmetry
    worst_case_circle += np.random.randn(100, 2) * 2
    plot_scenario(worst_case_circle, eps_test, "Worst Case: Asymmetrical Bumpy Circle", "approx_worst_circle")

    # 3. Worst Case 2: The Triangle (Forces center shifting)
    # A massive cluster of points shaped like an obtuse triangle
    worst_case_dumbbell = np.random.rand(100, 2) * 50
    # Filter to only keep points inside an obtuse triangle
    worst_case_dumbbell = np.array([p for p in worst_case_dumbbell if p[1] < p[0] and p[1] > -0.2*p[0]])
    if len(worst_case_dumbbell) < 3: worst_case_dumbbell = np.random.rand(50, 2) * 50 # Fallback
    plot_scenario(worst_case_dumbbell, eps_test, "Worst Case: Obtuse Asymmetry", "approx_worst_dumbbell")
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

def diameter_approximation(points):
    p = points[np.random.choice(points.shape[0])]
    q = points[np.argmax(np.sum((points - p) ** 2, axis=1))]
    q_ = points[np.argmax(np.sum((points - q) ** 2, axis=1))]
    return q, q_

def meb_approximation(points, eps):
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
    for dim in [2, 3, 1000]:
        test_points = np.random.rand(100, dim)
        center, radius, coreset = meb_approximation(test_points, eps=0.1)
        print(f"Dim: {dim} | Coreset size: {len(coreset)} | Center shape: {center.shape}")
    # --- Execution ---
    np.random.seed(42)
    points = np.random.randn(100, 2)  # Using Gaussian clusters for better visuals
    center, radius, coreset = meb_approximation(points, eps=0.01)

    # --- Plotting ---
    plt.figure(figsize=(8, 8))
    plt.scatter(points[:, 0], points[:, 1], c='gray', alpha=0.5, label='Input Points $S$')
    plt.scatter(coreset[:, 0], coreset[:, 1], c='red', edgecolors='black', s=80, label='Coreset $X$')

    # Draw the resulting ball
    circle = plt.Circle(center, radius, color='blue', fill=False, linewidth=2, label='Approx MEB')
    plt.gca().add_patch(circle)

    # Draw the center
    plt.plot(center[0], center[1], 'bx', markersize=10, label='Center')

    plt.axhline(0, color='black', lw=0.5, ls='--')
    plt.axvline(0, color='black', lw=0.5, ls='--')
    plt.title(f"MEB Approximation (ε=0.2)\nCoreset Size: {len(coreset)}")
    plt.legend()
    plt.axis('equal')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()
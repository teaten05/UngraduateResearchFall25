import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# ---------------------------
# SIMULATION PARAMETERS
# ---------------------------
N = 250                 # number of oscillators
K = 0.6                 # coupling strength
dt = 0.02               # smaller time step for better accuracy
T = 5000               # iterations per trial
delta = 0.05            # parameter perturbation range
sync_threshold = 0.8    # synchronization threshold
num_trials = 300        # total independent trials (adjust as needed)

# ---------------------------
# NETWORK CONSTRUCTION (RING TOPOLOGY)
# ---------------------------
G = nx.cycle_graph(N)
A = nx.to_numpy_array(G, dtype=np.float64)

# ---------------------------
# HELPER FUNCTIONS
# ---------------------------
def order_param(theta):
    """Compute Kuramoto-style order parameter r(t)."""
    z = np.exp(1j * theta)
    return np.abs(np.mean(z))

def normalize_rows(M):
    """Normalize rows to make valid probability distributions (no NaNs)."""
    M = np.maximum(M, 0)
    row_sums = M.sum(axis=1, keepdims=True)
    safe_row_sums = np.where(row_sums == 0, 1, row_sums)
    M = np.divide(M, safe_row_sums, out=np.zeros_like(M), where=safe_row_sums != 0)
    zero_rows = (row_sums.flatten() == 0)
    if np.any(zero_rows):
        M[zero_rows] = np.ones(M.shape[1]) / M.shape[1]
    M = np.clip(M, 0, 1)
    M = M / M.sum(axis=1, keepdims=True)
    return M

# ---------------------------
# SINGLE TRIAL SIMULATION (RK2 Integrator)
# ---------------------------
def run_trial(seed):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2*np.pi, N).astype(np.float64)
    omega = rng.uniform(0.9, 1.1, N).astype(np.float64)
    r_vals = np.empty(T, dtype=np.float64)

    for n in range(T):
        r_vals[n] = order_param(theta)

        # Influence matrix
        delta_theta = theta[np.newaxis, :] - theta[:, np.newaxis]
        influence = np.maximum(0, np.sin(delta_theta)) * A

        # Stochastic perturbations
        P = influence * rng.uniform(1 - delta, 1 + delta, size=(N, N))
        P = normalize_rows(P)

        # Choose neighbors based on probabilities
        cumP = np.cumsum(P, axis=1)
        r_rand = rng.random(N)
        j_idx = (cumP < r_rand[:, None]).sum(axis=1)

        # RK2 (Midpoint method)
        perturb = rng.uniform(1 - delta, 1 + delta, size=N)
        k1 = omega + K * perturb * np.sin(theta[j_idx] - theta)
        theta_half = theta + 0.5 * dt * k1
        k2 = omega + K * perturb * np.sin(theta_half[j_idx] - theta_half)
        theta = np.mod(theta + dt * k2, 2*np.pi)

    return r_vals

# ---------------------------
# PARALLEL EXECUTION
# ---------------------------
def parallel_simulation():
    num_workers = min(num_trials, multiprocessing.cpu_count())
    print(f"Running {num_trials} trials across {num_workers} workers...")

    all_results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(run_trial, seed) for seed in range(num_trials)]
        for f in as_completed(futures):
            all_results.append(f.result())

    return np.array(all_results, dtype=np.float64)

# ---------------------------
# MAIN EXECUTION
# ---------------------------
if __name__ == "__main__":
    results = parallel_simulation()

    avg_r = np.mean(results, axis=0)
    final_r = results[:, -1]
    mean_final = np.mean(final_r)
    std_final = np.std(final_r)
    min_final = np.min(final_r)
    max_final = np.max(final_r)
    ci95 = 1.96 * std_final / np.sqrt(num_trials)

    np.savetxt("markov_kuramoto_ring_highprecision.csv", results, delimiter=",")
    print(f"\n✅ Saved all {num_trials} trials to 'markov_kuramoto_ring_highprecision.csv'")

    # ---------------------------
    # STATISTICS
    # ---------------------------
    print("\n=== Synchronization Statistics ===")
    print(f"Final mean r(T):      {mean_final:.4f}")
    print(f"Standard deviation:   {std_final:.4f}")
    print(f"95% CI:               {mean_final - ci95:.4f} to {mean_final + ci95:.4f}")
    print(f"Min / Max r(T):       {min_final:.4f} / {max_final:.4f}")
    print("=================================\n")

    # ---------------------------
    # PLOTTING
    # ---------------------------
    plt.figure(figsize=(10, 5))
    for trial in results:
        plt.plot(trial, color='gray', alpha=0.15, linewidth=0.8)
    plt.plot(avg_r, color='blue', lw=2.5, label="Average r(t)")
    plt.axhline(sync_threshold, color='r', linestyle='--', label='Sync threshold (0.8)')
    plt.title("High-Precision RK2: Ring Topology Synchronization")
    plt.xlabel("Iteration (n)")
    plt.ylabel("r(t)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("markov_kuramoto_ring_highprecision_plot.png", dpi=200)
    plt.close()

    print(f"Done. Final mean r(T) = {mean_final:.4f}")

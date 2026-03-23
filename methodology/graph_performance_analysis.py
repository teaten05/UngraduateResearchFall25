"""
Graph Performance Analysis: Geometric Random Graph vs Other Topologies
======================================================================
Loads the latest simulation output and generates:
  1. Plot 1 (primary): <r_final> vs density — shows sharp synchronization transition
  2. Plot 2: Global network efficiency vs density for multiple graph types
  3. Diagram: Sparse vs dense geometric network comparison
  4. Table: Performance metrics for geometric vs other graph topologies

Kuramoto synchronization model (stochastic RK2) — N=150, K=1.0
"""

import os
import csv
import math
import random
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import to_rgba
import matplotlib.patches as mpatches
from scipy.stats import binned_statistic

# ── output directory ──────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "analysis_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── data paths (latest run first, dataset fallback) ───────────────────────────
DATA_PATHS = [
    os.path.join(os.path.dirname(__file__),
                 "research_output", "dataset", "data.csv"),
    os.path.join(os.path.dirname(__file__),
                 "research_output", "run_20251113_120451", "data.csv"),
]

# ── Kuramoto / comparison simulation params ───────────────────────────────────
SIM_N       = 50          # smaller N for fast cross-topology comparisons
SIM_K       = 1.0
SIM_DT      = 0.02
SIM_T       = 3000        # iterations
SIM_TRIALS  = 8           # trials per topology instance
SIM_DELTA   = 0.05        # stochastic perturbation range
RNG_SEED    = 42

# =============================================================================
#  DATA LOADING
# =============================================================================

def load_latest_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Return (density, r_final, efficiency, extra_cols) from the newest CSV."""
    for path in DATA_PATHS:
        if os.path.isfile(path):
            densities, r_finals, efficiencies = [], [], []
            extras: dict[str, list] = {}
            with open(path, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    densities.append(float(row["density"]))
                    r_finals.append(float(row["r_final"]))
                    efficiencies.append(float(row["efficiency"]))
                    for k in ("avg_clustering", "avg_path_length",
                              "spectral_gap", "modularity",
                              "avg_degree", "num_nodes", "num_edges"):
                        extras.setdefault(k, []).append(float(row[k]))
            print(f"Loaded {len(densities)} samples from:\n  {path}")
            return (np.array(densities), np.array(r_finals),
                    np.array(efficiencies),
                    {k: np.array(v) for k, v in extras.items()})
    raise FileNotFoundError("No data CSV found in expected locations.")


# =============================================================================
#  KURAMOTO SIMULATION (stochastic RK2, same model as project)
# =============================================================================

def order_param(theta: np.ndarray) -> float:
    return float(abs(np.mean(np.exp(1j * theta))))


def normalize_rows(M: np.ndarray) -> np.ndarray:
    M = np.maximum(M, 0.0)
    row_sums = M.sum(axis=1, keepdims=True)
    safe = np.where(row_sums == 0, 1.0, row_sums)
    M = M / safe
    zero_mask = (row_sums.flatten() == 0)
    if zero_mask.any():
        M[zero_mask] = 1.0 / M.shape[1]
    return np.clip(M, 0.0, 1.0) / M.sum(axis=1, keepdims=True)


def run_kuramoto_trial(A: np.ndarray, seed: int,
                       K: float = SIM_K, dt: float = SIM_DT,
                       T: int = SIM_T, delta: float = SIM_DELTA) -> float:
    """Single stochastic RK2 Kuramoto trial. Returns final order parameter."""
    rng = np.random.default_rng(seed)
    n = A.shape[0]
    theta = rng.uniform(0, 2 * math.pi, n)
    omega = rng.uniform(0.9, 1.1, n)

    for _ in range(T):
        d_theta = theta[np.newaxis, :] - theta[:, np.newaxis]
        influence = np.maximum(0.0, np.sin(d_theta)) * A
        P = influence * rng.uniform(1 - delta, 1 + delta, (n, n))
        P = normalize_rows(P)
        cumP = np.cumsum(P, axis=1)
        j_idx = (cumP < rng.random(n)[:, None]).sum(axis=1)
        j_idx = np.clip(j_idx, 0, n - 1)
        perturb = rng.uniform(1 - delta, 1 + delta, n)
        k1 = omega + K * perturb * np.sin(theta[j_idx] - theta)
        th_half = theta + 0.5 * dt * k1
        k2 = omega + K * perturb * np.sin(th_half[j_idx] - th_half)
        theta = np.mod(theta + dt * k2, 2 * math.pi)

    return order_param(theta)


def mean_r_final(G: nx.Graph, n_trials: int = SIM_TRIALS,
                 base_seed: int = 0) -> tuple[float, float]:
    """Return (mean r_final, std r_final) over n_trials."""
    A = nx.to_numpy_array(G, dtype=np.float64)
    rs = [run_kuramoto_trial(A, base_seed + i) for i in range(n_trials)]
    return float(np.mean(rs)), float(np.std(rs))


def global_efficiency(G: nx.Graph) -> float:
    """Latora–Marchiori global efficiency = mean(1/d_ij)."""
    n = G.number_of_nodes()
    if n < 2:
        return 0.0
    total = 0.0
    for src in G.nodes():
        lengths = nx.single_source_shortest_path_length(G, src)
        total += sum(1.0 / d for v, d in lengths.items() if v != src and d > 0)
    return total / (n * (n - 1))


# =============================================================================
#  COMPARISON GRAPH GENERATION  (N = SIM_N)
# =============================================================================

def build_comparison_graphs(n: int = SIM_N, rng_seed: int = RNG_SEED
                            ) -> list[dict]:
    """
    Build a representative set of graph types across a density range.
    Returns list of dicts with keys: label, graph, density, topology_type.
    """
    rng = random.Random(rng_seed)
    graphs = []

    # 1. Ring (cycle)
    G = nx.cycle_graph(n)
    graphs.append(dict(label="Ring", graph=G, topology_type="ring"))

    # 2. Path graph
    G = nx.path_graph(n)
    graphs.append(dict(label="Path", graph=G, topology_type="path"))

    # 3. Watts-Strogatz small-world (sparse)
    G = nx.watts_strogatz_graph(n, k=4, p=0.1, seed=rng_seed)
    graphs.append(dict(label="Small-world (k=4)", graph=G,
                       topology_type="small_world"))

    # 4. Watts-Strogatz small-world (denser)
    G = nx.watts_strogatz_graph(n, k=12, p=0.1, seed=rng_seed)
    graphs.append(dict(label="Small-world (k=12)", graph=G,
                       topology_type="small_world"))

    # 5. Erdős-Rényi at several densities
    for p in [0.08, 0.20, 0.40]:
        G = nx.erdos_renyi_graph(n, p, seed=rng_seed)
        if not nx.is_connected(G):          # ensure connectivity
            largest = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest).copy()
            G = nx.convert_node_labels_to_integers(G)
        graphs.append(dict(label=f"Erdős-Rényi (p={p})",
                           graph=G, topology_type="erdos_renyi"))

    # 6. Barabási-Albert scale-free
    G = nx.barabasi_albert_graph(n, 3, seed=rng_seed)
    graphs.append(dict(label="Scale-free (BA)", graph=G,
                       topology_type="scale_free"))

    # 7. Geometric random graph at low / medium / high density
    pos = {i: (random.Random(rng_seed + i).random(),
               random.Random(rng_seed + i + 1000).random())
           for i in range(n)}
    for r_val, label in [(0.18, "Geometric (sparse)"),
                          (0.30, "Geometric (medium)"),
                          (0.45, "Geometric (dense)")]:
        G = nx.random_geometric_graph(n, r_val, seed=rng_seed)
        if not nx.is_connected(G):
            comps = list(nx.connected_components(G))
            for ci in range(len(comps) - 1):
                a = list(comps[ci])[0]
                b = list(comps[ci + 1])[0]
                G.add_edge(a, b)
        graphs.append(dict(label=label, graph=G,
                           topology_type="geometric"))

    # 8. All-to-all (complete)
    G = nx.complete_graph(n)
    graphs.append(dict(label="All-to-all (complete)", graph=G,
                       topology_type="complete"))

    # annotate density
    max_e = n * (n - 1) / 2
    for d in graphs:
        d["density"] = d["graph"].number_of_edges() / max_e

    return graphs


# =============================================================================
#  PLOT 1 — ⟨r_final⟩ vs DENSITY  (MOST IMPORTANT)
# =============================================================================

def plot_r_final_vs_density(density: np.ndarray,
                             r_final: np.ndarray,
                             comparison: list[dict]) -> str:
    """
    Scatter + binned mean/CI of geometric data.
    Annotates comparison topology reference lines.
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    # ── geometric random graph data (scatter + binned mean) ──────────────────
    ax.scatter(density, r_final, s=4, alpha=0.25, color="#4C72B0",
               label="Geometric (individual runs)", zorder=2)

    n_bins = 30
    bin_means, bin_edges, _ = binned_statistic(
        density, r_final, statistic='mean', bins=n_bins)
    bin_stds, _, _ = binned_statistic(
        density, r_final, statistic='std', bins=n_bins)
    bin_counts, _, _ = binned_statistic(
        density, r_final, statistic='count', bins=n_bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # 95% CI
    ci95 = 1.96 * bin_stds / np.sqrt(np.maximum(bin_counts, 1))

    valid = ~np.isnan(bin_means)
    ax.plot(bin_centers[valid], bin_means[valid],
            color="#C44E52", lw=2.5, zorder=4,
            label=r"Geometric $\langle r_{\mathrm{final}} \rangle$ (binned)")
    ax.fill_between(bin_centers[valid],
                    bin_means[valid] - ci95[valid],
                    bin_means[valid] + ci95[valid],
                    alpha=0.25, color="#C44E52", label="95% CI")

    # ── comparison topologies (horizontal reference lines) ───────────────────
    palette = {
        "ring":        "#2ca02c",
        "path":        "#8c564b",
        "small_world": "#9467bd",
        "erdos_renyi": "#e377c2",
        "scale_free":  "#7f7f7f",
        "geometric":   "#bcbd22",
        "complete":    "#17becf",
    }
    print("\nRunning comparison simulations …")
    annotated_types = set()
    for item in comparison:
        ttype = item["topology_type"]
        color = palette.get(ttype, "gray")
        mr, sr = item.get("mean_r", np.nan), item.get("std_r", np.nan)
        if np.isnan(mr):
            continue
        # only add one legend entry per topology type
        lbl = item["label"] if ttype not in annotated_types else "_nolegend_"
        annotated_types.add(ttype)
        ax.errorbar(item["density"], mr, yerr=sr,
                    fmt="D", ms=8, color=color, capsize=4,
                    label=lbl, zorder=5)

    # ── all-to-all reference (theoretical ≈ 1) ───────────────────────────────
    ax.axhline(y=1.0, color="#17becf", lw=1.4, ls="--", alpha=0.7,
               label="All-to-all theoretical limit (r→1)")

    ax.set_xlabel("Network Density  (edges / max edges)", fontsize=13)
    ax.set_ylabel(r"$\langle r_{\mathrm{final}} \rangle$  (mean sync. order param.)",
                  fontsize=13)
    ax.set_title(
        r"Synchronization vs Network Density — Geometric Random Graph"
        "\n(Stochastic Kuramoto, N=150, K=1.0)",
        fontsize=13)
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(0.0, 1.05)
    ax.legend(fontsize=8, loc="lower right", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # annotate transition region
    ax.axvspan(density.min(), density.max(), alpha=0.07, color="#4C72B0",
               label="_nolegend_")
    ax.text(density.mean(), 0.04,
            f"Geometric data range\n[{density.min():.2f} – {density.max():.2f}]",
            ha="center", va="bottom", fontsize=8, color="#4C72B0",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "plot1_r_final_vs_density.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")
    return out


# =============================================================================
#  PLOT 2 — GLOBAL NETWORK EFFICIENCY vs DENSITY
# =============================================================================

def plot_efficiency_vs_density(comparison: list[dict],
                                density_geo: np.ndarray,
                                efficiency_geo: np.ndarray) -> str:
    """
    Plot global network efficiency (Latora–Marchiori) vs graph density
    for each comparison topology.  Geometric data uses the stored efficiency.
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    palette = {
        "ring":        "#2ca02c",
        "path":        "#8c564b",
        "small_world": "#9467bd",
        "erdos_renyi": "#e377c2",
        "scale_free":  "#7f7f7f",
        "geometric":   "#4C72B0",
        "complete":    "#17becf",
    }

    # ── geometric dataset (density ≈ efficiency for these graphs) ─────────────
    # bin the geometric data
    n_bins = 25
    bin_means, bin_edges, _ = binned_statistic(
        density_geo, efficiency_geo, statistic='mean', bins=n_bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    valid = ~np.isnan(bin_means)
    ax.plot(bin_centers[valid], bin_means[valid],
            color="#4C72B0", lw=2.5, zorder=4,
            label="Geometric (dataset, binned)")
    ax.fill_between(bin_centers[valid],
                    bin_means[valid] * 0.97,
                    bin_means[valid] * 1.03,
                    alpha=0.2, color="#4C72B0")

    # ── comparison topologies ─────────────────────────────────────────────────
    type_scatter: dict[str, tuple[list, list]] = {}
    for item in comparison:
        ttype = item["topology_type"]
        ge = item.get("global_efficiency", np.nan)
        if np.isnan(ge):
            continue
        type_scatter.setdefault(ttype, ([], []))
        type_scatter[ttype][0].append(item["density"])
        type_scatter[ttype][1].append(ge)

    for ttype, (xs, ys) in type_scatter.items():
        color = palette.get(ttype, "gray")
        # pick a display label
        label_map = {
            "ring":        "Ring",
            "path":        "Path",
            "small_world": "Small-world (WS)",
            "erdos_renyi": "Erdős-Rényi",
            "scale_free":  "Scale-free (BA)",
            "geometric":   "Geometric (comparison)",
            "complete":    "All-to-all (complete)",
        }
        ax.scatter(xs, ys, s=90, color=color, zorder=5,
                   label=label_map.get(ttype, ttype), marker="D")

    ax.set_xlabel("Network Density  (edges / max edges)", fontsize=13)
    ax.set_ylabel("Global Network Efficiency  (Latora–Marchiori)", fontsize=13)
    ax.set_title(
        "Global Network Efficiency vs Density\nMultiple Graph Topologies",
        fontsize=13)
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(0.0, 1.05)
    ax.legend(fontsize=9, loc="lower right", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "plot2_efficiency_vs_density.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")
    return out


# =============================================================================
#  DIAGRAM — SPARSE vs DENSE NETWORK COMPARISON
# =============================================================================

def plot_sparse_dense_diagram() -> str:
    """
    Side-by-side geometric random graph drawings:
    left = sparse (low density), right = dense (high density).
    Colour nodes by degree, annotate key metrics.
    """
    n_diag = 40
    seed_d = 7

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    fig.suptitle(
        "Geometric Random Graph: Sparse vs Dense Network",
        fontsize=14, fontweight="bold", y=1.01)

    configs = [
        dict(r=0.18, title="Sparse Network",
             subtitle="low density · fewer edges\npoor synchronisation",
             color="#4C72B0", ax=axes[0]),
        dict(r=0.38, title="Dense Network",
             subtitle="high density · more edges\nstrong synchronisation",
             color="#C44E52", ax=axes[1]),
    ]

    for cfg in configs:
        G = nx.random_geometric_graph(n_diag, cfg["r"], seed=seed_d)
        # ensure connected
        if not nx.is_connected(G):
            comps = list(nx.connected_components(G))
            for ci in range(len(comps) - 1):
                a = list(comps[ci])[0]
                b = list(comps[ci + 1])[0]
                G.add_edge(a, b)

        ax = cfg["ax"]
        pos = nx.get_node_attributes(G, "pos")

        degrees = dict(G.degree())
        max_deg = max(degrees.values()) if degrees else 1
        node_colors = [plt.cm.YlOrRd(degrees[v] / max_deg) for v in G.nodes()]
        node_sizes  = [60 + 8 * degrees[v] for v in G.nodes()]

        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.35,
                               width=0.9, edge_color="#888888")
        nx.draw_networkx_nodes(G, pos, ax=ax,
                               node_color=node_colors, node_size=node_sizes,
                               linewidths=0.5, edgecolors="white")

        dens   = G.number_of_edges() / (n_diag * (n_diag - 1) / 2)
        avg_cl = nx.average_clustering(G)
        eff    = global_efficiency(G)
        try:
            apl = nx.average_shortest_path_length(G)
        except Exception:
            apl = float("nan")

        info = (f"N={n_diag}  |  Edges={G.number_of_edges()}\n"
                f"Density={dens:.3f}  |  ⟨k⟩={np.mean(list(degrees.values())):.1f}\n"
                f"Clustering={avg_cl:.3f}  |  Efficiency={eff:.3f}\n"
                f"Avg path length={apl:.2f}")

        ax.set_title(f"{cfg['title']}\n{cfg['subtitle']}",
                     fontsize=12, color=cfg["color"], fontweight="bold")
        ax.text(0.01, -0.08, info, transform=ax.transAxes,
                fontsize=8.5, va="top", family="monospace",
                bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow",
                          ec="gray", alpha=0.9))
        ax.axis("off")

    # shared colorbar (degree)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.YlOrRd,
                               norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(),
                        orientation="horizontal", fraction=0.03, pad=0.12,
                        shrink=0.5)
    cbar.set_label("Relative node degree (0 = min, 1 = max)", fontsize=9)

    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "diagram_sparse_vs_dense.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")
    return out


# =============================================================================
#  TABLE — PERFORMANCE COMPARISON
# =============================================================================

def build_and_save_table(comparison: list[dict],
                          density_geo: np.ndarray,
                          r_final_geo: np.ndarray,
                          efficiency_geo: np.ndarray,
                          extras: dict) -> str:
    """
    Build a summary table as a matplotlib figure (and also save as CSV).
    Columns: Topology | Density | ⟨r_final⟩ | Glob. Efficiency |
             Avg Clustering | Avg Path Length | Notes
    """
    rows = []

    # ── Geometric (from dataset, low / mid / high density bins) ──────────────
    geo_bands = [
        ("Geometric (sparse)",  (0.35, 0.40)),
        ("Geometric (mid)",     (0.44, 0.46)),
        ("Geometric (dense)",   (0.50, 0.55)),
    ]
    for label, (lo, hi) in geo_bands:
        mask = (density_geo >= lo) & (density_geo <= hi)
        if mask.sum() == 0:
            continue
        rows.append({
            "Topology":          label,
            "Density":           f"{density_geo[mask].mean():.3f}",
            "⟨r_final⟩":        f"{r_final_geo[mask].mean():.4f} ± {r_final_geo[mask].std():.4f}",
            "Glob. Efficiency":  f"{efficiency_geo[mask].mean():.4f}",
            "Avg Clustering":    f"{extras['avg_clustering'][mask].mean():.4f}",
            "Avg Path Length":   f"{extras['avg_path_length'][mask].mean():.3f}",
            "Notes":             "empirical dataset",
        })

    # ── All-to-all reference row (theoretical) ───────────────────────────────
    rows.append({
        "Topology":          "All-to-all (N=150)",
        "Density":           "1.000",
        "⟨r_final⟩":        "≈ 1.000 (theoretical)",
        "Glob. Efficiency":  "1.000",
        "Avg Clustering":    "1.000",
        "Avg Path Length":   "1.000",
        "Notes":             "complete graph, K→∞ limit",
    })

    # ── Comparison topologies (simulated) ─────────────────────────────────────
    type_order = ["path", "ring", "small_world", "erdos_renyi",
                  "scale_free", "geometric", "complete"]
    seen = set()
    for ttype in type_order:
        for item in comparison:
            if item["topology_type"] != ttype:
                continue
            key = item["label"]
            if key in seen:
                continue
            seen.add(key)
            mr = item.get("mean_r", float("nan"))
            sr = item.get("std_r",  float("nan"))
            ge = item.get("global_efficiency", float("nan"))
            cl = item.get("avg_clustering",    float("nan"))
            pl = item.get("avg_path_length",   float("nan"))
            rows.append({
                "Topology":          f"{item['label']} (N={SIM_N})",
                "Density":           f"{item['density']:.3f}",
                "⟨r_final⟩":        (f"{mr:.4f} ± {sr:.4f}"
                                      if not math.isnan(mr) else "—"),
                "Glob. Efficiency":  (f"{ge:.4f}" if not math.isnan(ge) else "—"),
                "Avg Clustering":    (f"{cl:.4f}" if not math.isnan(cl) else "—"),
                "Avg Path Length":   (f"{pl:.3f}"  if not math.isnan(pl) else "—"),
                "Notes":             f"simulated (N={SIM_N})",
            })

    # ── Save as CSV ───────────────────────────────────────────────────────────
    csv_out = os.path.join(OUTPUT_DIR, "table_graph_comparison.csv")
    cols = ["Topology", "Density", "⟨r_final⟩",
            "Glob. Efficiency", "Avg Clustering", "Avg Path Length", "Notes"]
    with open(csv_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  CSV saved → {csv_out}")

    # ── Render as matplotlib table ────────────────────────────────────────────
    n_rows = len(rows)
    fig_h = max(4, 0.45 * n_rows + 2.0)
    fig, ax = plt.subplots(figsize=(16, fig_h))
    ax.axis("off")

    col_labels = cols
    cell_data = [[r[c] for c in cols] for r in rows]

    tbl = ax.table(
        cellText=cell_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.auto_set_column_width(list(range(len(cols))))

    # shade header
    for j in range(len(cols)):
        tbl[(0, j)].set_facecolor("#2C3E50")
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")

    # shade geometric rows
    for i, row in enumerate(rows, start=1):
        if "Geometric" in row["Topology"]:
            for j in range(len(cols)):
                tbl[(i, j)].set_facecolor("#EBF5FB")
        elif "All-to-all" in row["Topology"]:
            for j in range(len(cols)):
                tbl[(i, j)].set_facecolor("#FDFEFE")
                tbl[(i, j)].set_text_props(fontstyle="italic")
        elif i % 2 == 0:
            for j in range(len(cols)):
                tbl[(i, j)].set_facecolor("#F9F9F9")

    ax.set_title(
        "Graph Topology Performance Comparison\n"
        "Geometric Random Graph vs Other Topologies  |  Kuramoto Sync Model",
        fontsize=11, fontweight="bold", pad=16)

    fig.tight_layout()
    png_out = os.path.join(OUTPUT_DIR, "table_graph_comparison.png")
    fig.savefig(png_out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Table image saved → {png_out}")
    return png_out


# =============================================================================
#  MAIN
# =============================================================================

def main():
    print("=" * 65)
    print("  GRAPH PERFORMANCE ANALYSIS")
    print("=" * 65)

    # ── 1. Load dataset ───────────────────────────────────────────────────────
    density_geo, r_final_geo, efficiency_geo, extras = load_latest_data()

    # ── 2. Build comparison graphs & compute metrics ──────────────────────────
    print("\nBuilding comparison topologies …")
    comparison = build_comparison_graphs(n=SIM_N)

    print(f"  Running Kuramoto simulations ({SIM_TRIALS} trials × "
          f"{len(comparison)} topologies, T={SIM_T}) …")
    for idx, item in enumerate(comparison):
        print(f"    [{idx+1:2d}/{len(comparison)}] {item['label']} "
              f"(density={item['density']:.3f}) …", end=" ", flush=True)
        mr, sr = mean_r_final(item["graph"],
                              n_trials=SIM_TRIALS,
                              base_seed=idx * 100)
        ge  = global_efficiency(item["graph"])
        acl = nx.average_clustering(item["graph"])
        try:
            apl = nx.average_shortest_path_length(item["graph"])
        except Exception:
            apl = float("nan")
        item["mean_r"] = mr
        item["std_r"]  = sr
        item["global_efficiency"] = ge
        item["avg_clustering"]    = acl
        item["avg_path_length"]   = apl
        print(f"r={mr:.4f} ± {sr:.4f}")

    # ── 3. Generate plots ─────────────────────────────────────────────────────
    print("\nGenerating plots …")
    plot_r_final_vs_density(density_geo, r_final_geo, comparison)
    plot_efficiency_vs_density(comparison, density_geo, efficiency_geo)
    plot_sparse_dense_diagram()
    build_and_save_table(comparison, density_geo, r_final_geo,
                         efficiency_geo, extras)

    print("\n" + "=" * 65)
    print(f"  All outputs saved to: {OUTPUT_DIR}/")
    print("  Files:")
    for fname in sorted(os.listdir(OUTPUT_DIR)):
        print(f"    • {fname}")
    print("=" * 65)


if __name__ == "__main__":
    main()

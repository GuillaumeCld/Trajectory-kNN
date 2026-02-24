import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# MATPLOTLIB STYLE
# ============================================================

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman"],
})
plt.rcParams["font.size"] = 19

# ============================================================
# CONFIG
# ============================================================

RESULTS_DIR = Path("experiments/results")
FIGURES_DIR = Path("experiments/figures")

DIMENSIONS = [50, 100, 150, 200, 250]

FILES = {
    "algo_cpu": "algo_cpu.csv",
    "algo_gpu": "algo_cuda.csv",
    "faiss_cpu": "faiss_results_cpu.csv",
    "faiss_gpu": "faiss_results_gpu.csv",
}

# ============================================================
# VISUAL STYLE CONFIG
# ============================================================

ALGO_COLORS = {
    "TRAKNN": "#0072B2",   # deep blue
    "FAISS":  "#D55E00",   # vermillion
}

HARDWARE_LINESTYLE = {
    "CPU": "--",
    "GPU": "-",
}

MARKERS = {
    ("TRAKNN", "CPU"): "o",
    ("TRAKNN", "GPU"): "s",
    ("FAISS", "CPU"):  "^",
    ("FAISS", "GPU"):  "D",
}

SOURCE_PRETTY = {
    "algo_cpu": "TRAKNN (CPU)",
    "algo_gpu": "TRAKNN (GPU)",
    "faiss_cpu": "FAISS (CPU)",
    "faiss_gpu": "FAISS (GPU)",
}

SOURCE_MAP = {
    "TRAKNN (CPU)": ("TRAKNN", "CPU"),
    "TRAKNN (GPU)": ("TRAKNN", "GPU"),
    "FAISS (CPU)":  ("FAISS",  "CPU"),
    "FAISS (GPU)":  ("FAISS",  "GPU"),
}

# ============================================================
# DATA LOADING
# ============================================================

def load_csv(path):
    return pd.read_csv(path)


def load_dimension_times(dimensions):
    results = {
        "TRAKNN (CPU)": [],
        "TRAKNN (GPU)": [],
        "FAISS (CPU)": [],
        "FAISS (GPU)": [],
    }

    for H in dimensions:
        results["TRAKNN (CPU)"].append(
            load_csv(RESULTS_DIR / f"algo_cpu_{H}.csv")["faiss_time"].iloc[0]
        )
        results["TRAKNN (GPU)"].append(
            load_csv(RESULTS_DIR / f"algo_cuda_{H}.csv")["faiss_time"].iloc[0]
        )
        results["FAISS (CPU)"].append(
            load_csv(RESULTS_DIR / f"faiss_results_cpu_{H}.csv")["faiss_time"].iloc[0]
        )
        results["FAISS (GPU)"].append(
            load_csv(RESULTS_DIR / f"faiss_results_gpu_{H}.csv")["faiss_time"].iloc[0]
        )

    return results


def load_full_experiment_data():
    dfs = []
    for source, filename in FILES.items():
        df = load_csv(RESULTS_DIR / filename)
        df = df[df["status"] == "OK"].copy()
        df["source"] = source
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)

# ============================================================
# GENERIC STYLED PLOT
# ============================================================

def styled_plot(x, y, label):
    algo, hw = SOURCE_MAP[label]

    plt.plot(
        x,
        y,
        color=ALGO_COLORS[algo],
        linestyle=HARDWARE_LINESTYLE[hw],
        marker=MARKERS[(algo, hw)],
        linewidth=3,          # thicker line
        markersize=10,          # larger markers
        # markeredgewidth=1.8,
        # markeredgecolor="black",
        markerfacecolor=ALGO_COLORS[algo],
        label=label,
    )

# ============================================================
# FIGURE FUNCTIONS
# ============================================================

def plot_time_vs_dimension(dimensions, results):
    plt.figure(figsize=(8, 5))

    for label, times in results.items():
        styled_plot(dimensions, times, label)

    plt.xlabel("\(h\) (spatial dimension, with \(h=w\))")
    plt.ylabel("Time (seconds)")
    plt.grid(True, alpha=0.25)
    plt.ylim(-3, 300)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "time_vs_dimension.pdf", bbox_inches="tight")
    plt.show()


def plot_time_vs_T(data):
    plt.figure(figsize=(8, 5))
    subset = data[data["traj_length"] == 1]

    for source, group in subset.groupby("source"):
        label = SOURCE_PRETTY[source]
        group = group.sort_values("T")
        styled_plot(group["T"], group["faiss_time"], label)

    plt.xlabel("\(n\) (time dimension)")
    plt.ylabel("Time (seconds)")
    plt.grid(True, alpha=0.25)
    plt.legend(frameon=False)
    plt.ylim(-3, 300)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "time_vs_T_traj_1.pdf", bbox_inches="tight")
    plt.show()


def plot_time_vs_traj_length(data):
    plt.figure(figsize=(8, 5))
    subset = data[data["T"] == 27375]

    for source, group in subset.groupby("source"):
        label = SOURCE_PRETTY[source]
        group = group.sort_values("traj_length")
        styled_plot(group["traj_length"], group["faiss_time"], label)

    plt.xlabel("\(d\) (trajectory length)")
    plt.ylabel("Time (seconds)")
    plt.xscale("log", base=2)
    plt.xticks([1, 2, 4, 8, 16], [1, 2, 4, 8, 16])
    plt.grid(True, alpha=0.25)
    # plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "time_vs_traj_length_T_27375.pdf",
                bbox_inches="tight")
    plt.show()


def plot_time_vs_k():
    datasets = {
        "TRAKNN (CPU)": pd.read_csv(RESULTS_DIR / "algo_cpu_k_trajlen1.csv"),
        "TRAKNN (GPU)": pd.read_csv(RESULTS_DIR / "algo_cuda_k_trajlen1.csv"),
        "FAISS (CPU)":  pd.read_csv(RESULTS_DIR / "faiss_cpu_k_trajlen1.csv"),
        "FAISS (GPU)":  pd.read_csv(RESULTS_DIR / "faiss_gpu_k_trajlen1.csv"),
    }

    plt.figure(figsize=(8, 5))

    for label, df in datasets.items():
        styled_plot(df["k"], df["runtime"], label)

    plt.xlabel("\(k\) (number of nearest neighbors)")
    plt.ylabel("Time (seconds)")
    plt.grid(True, alpha=0.25)
    # plt.legend(frameon=False)
    plt.ylim(-3, 300)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "time_vs_k_traj_length_1.pdf",
                bbox_inches="tight")
    plt.show()


# ============================================================
# MAIN
# ============================================================

def main():
    dim_results = load_dimension_times(DIMENSIONS)
    plot_time_vs_dimension(DIMENSIONS, dim_results)

    data = load_full_experiment_data()
    plot_time_vs_T(data)
    plot_time_vs_traj_length(data)

    plot_time_vs_k()


if __name__ == "__main__":
    main()
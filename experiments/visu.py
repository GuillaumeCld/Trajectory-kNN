import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

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
# UTILS
# ============================================================

def load_csv(path):
    return pd.read_csv(path)


def load_dimension_times(dimensions):
    """
    Load timing results for different dimensions.
    """
    results = {
        "algo_cpu": [],
        "algo_gpu": [],
        "faiss_cpu": [],
        "faiss_gpu": [],
    }

    for H in dimensions:
        results["algo_cpu"].append(
            load_csv(RESULTS_DIR / f"algo_cpu_{H}.csv")["faiss_time"].iloc[0]
        )
        results["algo_gpu"].append(
            load_csv(RESULTS_DIR / f"algo_cuda_{H}.csv")["faiss_time"].iloc[0]
        )
        results["faiss_cpu"].append(
            load_csv(RESULTS_DIR / f"faiss_results_cpu_{H}.csv")["faiss_time"].iloc[0]
        )
        results["faiss_gpu"].append(
            load_csv(RESULTS_DIR / f"faiss_results_gpu_{H}.csv")["faiss_time"].iloc[0]
        )

    return results


def load_full_experiment_data():
    """
    Load and merge all experiment CSV files.
    """
    dfs = []
    for source, filename in FILES.items():
        df = load_csv(RESULTS_DIR / filename)
        df = df[df["status"] == "OK"].copy()
        df["source"] = source
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


# ============================================================
# PLOTTING FUNCTIONS
# ============================================================

def plot_time_vs_dimension(dimensions, results):
    plt.figure(figsize=(10, 6))

    for label, times in results.items():
        plt.plot(dimensions, times, marker="o", label=label)

    plt.xlabel("Dimension (H=W)")
    plt.ylabel("Time (seconds)")
    plt.title("Time vs Dimension")
    plt.legend()
    plt.grid()

    plt.savefig(RESULTS_DIR / "time_vs_dimension.png")
    plt.show()


def plot_by_group(data, group_key, x_key, title, xlabel, filename):
    """
    Generic grouped subplot plotting.
    """
    groups = sorted(data[group_key].unique())

    fig, axes = plt.subplots(
        1, len(groups),
        figsize=(4 * len(groups), 4),
        sharey=True
    )

    if len(groups) == 1:
        axes = [axes]

    for ax, value in zip(axes, groups):
        subset = data[data[group_key] == value]

        for source, group in subset.groupby("source"):
            group = group.sort_values(x_key)
            ax.plot(
                group[x_key],
                group["faiss_time"],
                marker="o",
                label=source
            )

        ax.set_title(f"{group_key} = {value}")
        ax.set_xlabel(xlabel)
        ax.grid(True)

    axes[0].set_ylabel("FAISS Time (seconds)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.suptitle(title)

    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig(filename)
    plt.close()


def plot_scaling_summary(data):
    fig, axes = plt.subplots(1, 2, figsize=(14, 3), sharey=True)

    markers = ["o", "o", "s", "s", "^", "^"]
    linestyles = ["-", "--", "-", "--", "-", "--"]

    # ---- Subplot 1: traj_length == 4 ----
    subset = data[data["traj_length"] == 4]
    for i, (source, group) in enumerate(subset.groupby("source")):
        group = group.sort_values("T")
        axes[0].plot(
            group["T"],
            group["faiss_time"],
            marker=markers[i],
            linestyle=linestyles[i],
            label=source
        )

    axes[0].set_xlabel("Number of timesteps")
    axes[0].set_ylabel("Time (seconds)")
    axes[0].set_title("Trajectory length = 4")

    # ---- Subplot 2: T == 27375 ----
    subset = data[data["T"] == 27375]
    for i, (source, group) in enumerate(subset.groupby("source")):
        group = group.sort_values("traj_length")
        axes[1].plot(
            group["traj_length"],
            group["faiss_time"],
            marker=markers[i],
            linestyle=linestyles[i],
            label=source
        )

    axes[1].set_xlabel("Length of trajectories")
    axes[1].set_title("T = 27375")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=3)

    plt.savefig(FIGURES_DIR / "time_scaling.pdf", bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "time_scaling.png", bbox_inches="tight", dpi=300)
    plt.show()


# ============================================================
# MAIN
# ============================================================

def main():

    # ---- Dimension scaling ----
    dim_results = load_dimension_times(DIMENSIONS)
    plot_time_vs_dimension(DIMENSIONS, dim_results)

    # ---- Full experiment plots ----
    data = load_full_experiment_data()

    plot_by_group(
        data=data,
        group_key="traj_length",
        x_key="T",
        title="FAISS Time vs T (one subplot per trajectory length)",
        xlabel="T",
        filename=FIGURES_DIR / "time_vs_T.png"
    )

    plot_by_group(
        data=data,
        group_key="T",
        x_key="traj_length",
        title="FAISS Time vs Trajectory Length (one subplot per T)",
        xlabel="Trajectory Length",
        filename=FIGURES_DIR / "time_vs_traj_length.png"
    )

    plot_scaling_summary(data)


if __name__ == "__main__":
    main()

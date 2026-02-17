import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

time_algo_cpu = []
time_algo_gpu = []
time_faiss_cpu = []
time_faiss_gpu = []

dim = [50, 100, 150, 200, 250]

for H in dim:
    df_algo_cpu = pd.read_csv(f"experiments/results/algo_cpu_{H}.csv")
    df_algo_gpu = pd.read_csv(f"experiments/results/algo_cuda_{H}.csv")
    df_faiss_cpu = pd.read_csv(f"experiments/results/faiss_results_cpu_{H}.csv")
    df_faiss_gpu = pd.read_csv(f"experiments/results/faiss_results_gpu_{H}.csv")

    time_algo_cpu.append(df_algo_cpu["faiss_time"].values[0])
    time_algo_gpu.append(df_algo_gpu["faiss_time"].values[0])
    time_faiss_cpu.append(df_faiss_cpu["faiss_time"].values[0])
    time_faiss_gpu.append(df_faiss_gpu["faiss_time"].values[0])


plt.figure(figsize=(10, 6))
plt.plot(dim, time_algo_cpu, label="Algo CPU", marker="o")
plt.plot(dim, time_algo_gpu, label="Algo GPU", marker="o")
plt.plot(dim, time_faiss_cpu, label="FAISS CPU", marker="o")
plt.plot(dim, time_faiss_gpu, label="FAISS GPU", marker="o")
plt.xlabel("Dimension (H=W)")
plt.ylabel("Time (seconds)")
plt.title("Time vs Dimension for Algo and FAISS on CPU and GPU")
plt.legend()
plt.grid()
plt.savefig("experiments/results/time_vs_dimension.png")
plt.show()

import pandas as pd
import matplotlib.pyplot as plt

all_files = [
    "experiments/results/algo_cpu.csv",
    "experiments/results/algo_cuda.csv",
    "experiments/results/faiss_results_cpu.csv",
    "experiments/results/faiss_results_gpu.csv",
]

# ---- LOAD ALL DATA ----
dfs = []
for file in all_files:
    df = pd.read_csv(file)
    df = df[df["status"] == "OK"].copy()
    df["source"] = file.replace(".csv", "")
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)

traj_lengths = sorted(data["traj_length"].unique())
Ts = sorted(data["T"].unique())

# ============================================================
# FIGURE 1: one subplot per traj_length (x = T)
# ============================================================

fig1, axes1 = plt.subplots(
    nrows=1,
    ncols=len(traj_lengths),
    figsize=(4 * len(traj_lengths), 4),
    sharey=True
)

if len(traj_lengths) == 1:
    axes1 = [axes1]

for ax, traj_length in zip(axes1, traj_lengths):
    subset = data[data["traj_length"] == traj_length]

    for source, group in subset.groupby("source"):
        group = group.sort_values("T")
        ax.plot(
            group["T"],
            group["faiss_time"],
            marker="o",
            label=source
        )

    ax.set_title(f"traj_length = {traj_length}")
    ax.set_xlabel("T")
    ax.grid(True)

axes1[0].set_ylabel("FAISS Time (seconds)")

handles, labels = axes1[0].get_legend_handles_labels()
fig1.legend(handles, labels, loc="upper center", ncol=3)
fig1.suptitle("FAISS Time vs T (one subplot per trajectory length)")

plt.tight_layout(rect=[0, 0, 1, 0.9])
# plt.show()
plt.close()
# ============================================================
# FIGURE 2: one subplot per T (x = traj_length)
# ============================================================

fig2, axes2 = plt.subplots(
    nrows=1,
    ncols=len(Ts),
    figsize=(4 * len(Ts), 4),
    sharey=True
)

if len(Ts) == 1:
    axes2 = [axes2]

for ax, T in zip(axes2, Ts):
    subset = data[data["T"] == T]

    for source, group in subset.groupby("source"):
        group = group.sort_values("traj_length")
        ax.plot(
            group["traj_length"],
            group["faiss_time"],
            marker="o",
            label=source
        )

    ax.set_title(f"T = {T}")
    ax.set_xlabel("Trajectory Length")
    ax.grid(True)

axes2[0].set_ylabel("FAISS Time (seconds)")

handles, labels = axes2[0].get_legend_handles_labels()
fig2.legend(handles, labels, loc="upper center", ncol=3)
fig2.suptitle("FAISS Time vs Trajectory Length (one subplot per T)")

plt.tight_layout(rect=[0, 0, 1, 0.9])

# plt.show()
plt.close()




import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 3), sharey=True)

markers = ["o", "o", "s", "s", "^", "^"]
ls = ["-", "--", "-", "--", "-", "--"]

# ---- Subplot 1: faiss_time vs T (traj_length == 4) ----
subset = data[data["traj_length"] == 4]
i = 0
for source, group in subset.groupby("source"):
    group = group.sort_values("T")
    axes[0].plot(
        group["T"],
        group["faiss_time"],
        marker=markers[i],
        linestyle=ls[i],
        label=source
    )
    i += 1

axes[0].set_xlabel("Number of timesteps")
axes[0].set_ylabel("Time (seconds)")
axes[0].set_title("Trajectory length = 4")

# ---- Subplot 2: faiss_time vs traj_length (T == 27375) ----
subset = data[data["T"] == 27375]
i = 0
for source, group in subset.groupby("source"):
    group = group.sort_values("traj_length")
    axes[1].plot(
        group["traj_length"],
        group["faiss_time"],
        marker=markers[i],
        linestyle=ls[i],
        label=source
    )
    i += 1

axes[1].set_xlabel("Length of trajectories")
axes[1].set_title("T = 27375")

# ---- Common legend ----
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=3)

plt.savefig("experiments/figures/time_scaling.pdf", bbox_inches='tight')
plt.savefig("experiments/figures/time_scaling.png", bbox_inches='tight', dpi=300)
plt.show()

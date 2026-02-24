import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman"],
})
plt.rcParams["font.size"] = 16

# ============================================================
# LOAD DATA
# ============================================================

csv_file = "experiments/results/memory.csv"
df = pd.read_csv(csv_file)

# Extract algorithm and device
df["algorithm"] = df["name"].str.extract(r"^(.*?)\s*\(")
df["device"] = df["name"].str.extract(r"\((.*?)\)")

# Aggregate in case of duplicates
grouped = (
    df.groupby(["algorithm", "duration", "device"])["memory"]
    .mean()
    .reset_index()
)

algorithms = sorted(grouped["algorithm"].unique())
durations = sorted(grouped["duration"].unique())

# ============================================================
# STYLE CONFIG (MATCHES OTHER FIGURES)
# ============================================================

ALGO_COLORS = {
    "TRAKNN": "#0072B2",   # deep blue
    "FAISS":  "#D55E00",   # vermillion
}

DEVICE_HATCH = {
    "CPU": "",
    "GPU": "//",
}

bar_width = 0.35
group_spacing = 0.8

fig, ax = plt.subplots(figsize=(10, 5))

positions = []
labels = []

index = 0

# ============================================================
# PLOTTING
# ============================================================

for duration in durations:
    for algo in algorithms:

        subset = grouped[
            (grouped["algorithm"] == algo) &
            (grouped["duration"] == duration)
        ]

        cpu_val = subset[subset["device"] == "CPU"]["memory"]
        gpu_val = subset[subset["device"] == "GPU"]["memory"]

        cpu_val = cpu_val.values[0] if len(cpu_val) else 0
        gpu_val = gpu_val.values[0] if len(gpu_val) else 0

        color = ALGO_COLORS.get(algo, "#333333")

        # CPU bar
        ax.bar(
            index - bar_width/2,
            cpu_val,
            bar_width,
            color=color,
            edgecolor="black",
            hatch=DEVICE_HATCH["CPU"],
            label=f"{algo} (CPU)" if (duration == durations[0]) else "",
        )

        # GPU bar
        ax.bar(
            index + bar_width/2,
            gpu_val,
            bar_width,
            color=color,
            edgecolor="black",
            hatch=DEVICE_HATCH["GPU"],
            label=f"{algo} (GPU)" if (duration == durations[0]) else "",
        )

        positions.append(index)
        labels.append(f"{algo}\n(d={duration})")

        index += 1

    index += group_spacing  # spacing between duration groups

# ============================================================
# AXIS / LEGEND
# ============================================================

ax.set_xticks(positions)
ax.set_xticklabels(labels)
ax.set_ylabel("Memory Usage (GB)")

ax.legend(frameon=False, ncol=2)

plt.tight_layout()
plt.savefig("experiments/figures/memory_comparison.pdf",
            bbox_inches="tight")
plt.show()
import torch
import xarray as xr
import numpy as np
import plotly.express as px
from tqdm import tqdm
import pandas as pd

torch.backends.cudnn.benchmark = True


def knn_scores(nc_path, var, traj_length, k=10, q_batch=128, r_chunk=4096, device=None):
    # Load dataset
    ds = xr.open_dataset(nc_path)
    da = ds[var]

    spatial_dims = [d for d in da.dims if d != "time"]
    # (T, H, W) !!! load all data into memory !!!
    data = da.transpose("time", *spatial_dims).values.astype(np.float32)
    time = ds["time"].values
    ds.close()

    T, H, W = data.shape
    D = H * W  # vectorize spatial dimensions

    # Choose device for compute if not specified
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    # Move full dataset to device once so all computation stays on device
    X = torch.from_numpy(data).to(torch.float32).reshape(
        T, D).to(dev)  # (T, D) on device

    del data  # free CPU memory

    # Precompute norms on device
    norms = torch.empty(T, dtype=torch.float32, device=dev)
    with torch.no_grad():
        for rs in range(0, T, r_chunk):
            re = min(rs + r_chunk, T)
            block = X[rs:re]
            norms[rs:re] = (block * block).sum(dim=1)

    N = T - traj_length + 1
    scores = torch.empty(N, dtype=torch.float32, device=dev)
    # effective k cannot include self (N-1 neighbors max)
    k_eff = min(k, max(0, N - 1))

    # If k_eff == 0 just fill with NaN
    if k_eff == 0:
        scores[:] = float("nan")
    else:
        # Process query-starts in batches and streaming over reference chunks.
        # For each (qs:qe) we maintain a running top-k (smallest) over all reference blocks.
        with torch.no_grad():
            for qs in range(0, N, q_batch):
                qe = min(qs + q_batch, N)
                qlen = qe - qs

                # initialize top-k buffer with +inf
                knn_vals = torch.full((qlen, k_eff), float("inf"), device=dev)

                for rs in range(0, N, r_chunk):
                    re = min(rs + r_chunk, N)
                    rlen = re - rs

                    # accumulate Gram (Q @ R^T) and norm sums across trajectory offsets
                    G_sum = torch.zeros(
                        (qlen, rlen), dtype=torch.float32, device=dev)
                    Qn_sum = torch.zeros(qlen, dtype=torch.float32, device=dev)
                    Rn_sum = torch.zeros(rlen, dtype=torch.float32, device=dev)

                    # Sum contributions for t=0..traj_length-1
                    # Q indices: qs+t .. qe+t-1  (qlen rows)
                    # R indices: rs+t .. re+t-1  (rlen rows)
                    for t in range(traj_length):
                        Qt = X[qs + t: qe + t]    # (qlen, D)
                        Rt = X[rs + t: re + t]    # (rlen, D)
                        G_sum += Qt @ Rt.T
                        Qn_sum += norms[qs + t: qe + t]
                        Rn_sum += norms[rs + t: re + t]

                    # S_block = sum_t distances = Qn_sum[:,None] + Rn_sum[None,:] - 2 * G_sum
                    S_block = Qn_sum.unsqueeze(
                        1) + Rn_sum.unsqueeze(0) - 2.0 * G_sum

                    # Exclude self-matches where (qs + i) == (rs + j)
                    if (rs <= qs) and (qs < re):
                        d = qs - rs  # diagonal offset in this block
                        idx = torch.arange(0, qlen, device=dev)
                        j = idx + d
                        mask = j < rlen
                        if mask.any():
                            S_block[idx[mask], j[mask]] = float("inf")

                    # Merge current block with running knn_vals and keep smallest k_eff
                    # Concatenate along columns: (qlen, k_eff + rlen)
                    merged = torch.cat((knn_vals, S_block), dim=1)
                    knn_vals, _ = torch.topk(
                        merged, k_eff, dim=1, largest=False)

                # After scanning all reference chunks, compute mean of k nearest distances
                scores[qs:qe] = torch.sqrt(knn_vals).mean(dim=1)

    return scores.to("cpu"), time[:N]


if __name__ == "__main__":

    traj_length = 3
    k = 30

    scores, time = knn_scores(
        "Data/era5_msl_daily_eu.nc", "msl", traj_length, k, q_batch=256, r_chunk=4096*2, device="cuda")

    # Plot
    fig = px.line(x=time, y=scores.numpy(), labels={
        "x": "Time", "y": "Anomaly Score"})
    fig.show()

    top100_idx = np.argsort(-scores.numpy())[:100]
    for i, idx in enumerate(top100_idx):
        print(
           f"{i+1} {pd.to_datetime(time[idx]).date()}: {scores[idx].item():.3e}")

    # save the top dates to  a csv file
    top100_dates = [pd.to_datetime(time[idx]).date() for idx in top100_idx]
    df = pd.DataFrame(
        {"date": top100_dates, "anomaly_score": scores.numpy()[top100_idx]})
    df.to_csv(f"result/traj/top100_anomalous_dates_{traj_length}_{k}.csv", index=False)

    # print abnormal threshold score with IQR method, percentile 95
    q75, q25 = np.percentile(scores.numpy(), [75, 25])
    iqr = q75 - q25
    abnormal_threshold = q75 + 1.5 * iqr
    print(f"Abnormal threshold score (IQR method): {abnormal_threshold:.3e}")
    # print number of scores above abnormal thresholds
    num_abnormal = (scores.numpy() > abnormal_threshold).sum()
    print(f"Number of abnormal windows (score > threshold): {num_abnormal}")
    percentile_95 = np.percentile(scores.numpy(), 95)
    percentile_99 = np.percentile(scores.numpy(), 99)
    print(f"99th percentile score: {percentile_99:.3e}")
    print(f"95th percentile score: {percentile_95:.3e}")


    # Bar chart of the number of score above 99th percentile per year
    years = pd.to_datetime(time).year
    df_scores = pd.DataFrame({
        "year": years,
        "score": scores.numpy()
    })
    threshold_99 = percentile_99
    df_above_99 = df_scores[df_scores["score"] > threshold_99]
    counts = df_above_99["year"].value_counts().sort_index()    
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(counts.index, counts.values, color='C0')
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of scores above 99th percentile")
    ax.set_title("Number of anomaly scores above 99th percentile per year")
    ax.set_xticks(counts.index)
    ax.set_xticklabels(counts.index, rotation=45)
    plt.tight_layout()
    out_path = f"result/traj/anomaly_scores_above_99th_per_year_{traj_length}_{k}.png"
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
import torch
import xarray as xr
import numpy as np
import plotly.express as px
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

@torch.compile
def knn_scores(nc_path, var, k=10, q_batch=128, r_chunk=4096, device=None):
    # Load dataset
    ds = xr.open_dataset(nc_path) #  
    da = ds[var]

    spatial_dims = [d for d in da.dims if d != "time"]
    
    data = da.transpose("time", *spatial_dims).values # (T, H, W) !!! load all data into memory !!!
    time = ds["time"].values


    ds.close()

    T, H, W = data.shape
    D = H * W  # vectorize spatial dimensions

    # Choose device for compute if not specified
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    print(f"Using device: {dev}")

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

    # Output scores on device, 1 value per time step
    scores = torch.empty(T, dtype=torch.float32, device=dev)

    # Compute k-NN entirely on device
    for qs in tqdm(range(0, T, q_batch)):
        qe = min(qs + q_batch, T)
        Q = X[qs:qe]                    # (Qb, D) on device
        Qn = norms[qs:qe].unsqueeze(1)  # (Qb, 1) on device

        # best-so-far k per row in this query block (on device)
        best_d = torch.full((qe - qs, k), float("inf"), device=dev)
        best_i = torch.full((qe - qs, k), -1, dtype=torch.int, device=dev)

        for rs in range(0, T, r_chunk):
            re = min(rs + r_chunk, T)
            R = X[rs:re]                    # (Rb, D) on device
            Rn = norms[rs:re].unsqueeze(0)  # (1, Rb) on device

            # distances^2 = ||Q||^2 + ||R||^2 - 2 Q R^T
            G = Q @ R.T                    # (Qb,Rb) on device
            d2 = Qn + Rn - 2.0 * G

            # Exclude exact self matches when ref chunk overlaps query indices
            if rs <= qe and re > qs:
                r_idx = torch.arange(
                    rs, re, device=dev).unsqueeze(0)    # (1,Rb)
                q_idx = torch.arange(
                    qs, qe, device=dev).unsqueeze(1)    # (Qb,1)
                d2 = d2.masked_fill(r_idx == q_idx, float("inf"))

            # Merge with best-so-far and keep k smallest
            # (Qb, k+Rb)
            cand_d = torch.cat([best_d, d2], dim=1)
            r_inds = torch.arange(rs, re, device=dev).expand(qe - qs, -1)
            cand_i = torch.cat([best_i, r_inds], dim=1)

            new_d, idx = torch.topk(cand_d, k, dim=1, largest=False)
            new_i = torch.gather(cand_i, 1, idx)
            best_d, best_i = new_d, new_i

        # Mean of k-NN distances is the anomaly score
        scores[qs:qe] = torch.sqrt(torch.clamp_min(best_d, 0)).mean(dim=1)

    return scores.to("cpu"), time


if __name__ == "__main__":

    k = 30

    scores, time = knn_scores(
        "Data/era5_msl_daily_eu.nc", "msl", k=k, q_batch=256, r_chunk=4096*2, device="cuda")

    # Plot
    fig = px.line(x=time, y=scores.numpy(), labels={
                "x": "Time", "y": "Anomaly Score"})
    fig.show()

    top100_idx = np.argsort(-scores.numpy())[:100]

    # save the top dates to  a csv file
    top100_dates = [pd.to_datetime(time[idx]).date() for idx in top100_idx]
    df = pd.DataFrame(
        {"date": top100_dates, "anomaly_score": scores.numpy()[top100_idx]})
    df.to_csv(f"result/single/top100_anomalous_dates_{k}.csv", index=False)

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

    # Bar chart of the number of score above 99th percentile per year
    years = pd.to_datetime(time).year
    df_scores = pd.DataFrame({
        "year": years,
        "score": scores.numpy()
    })
    threshold_99 = percentile_99
    df_above_99 = df_scores[df_scores["score"] > threshold_99]
    counts = df_above_99["year"].value_counts().sort_index()    

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(counts.index, counts.values, color='C0')
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of scores above 99th percentile")
    ax.set_title("Number of anomaly scores above 99th percentile per year")
    ax.set_xticks(counts.index)
    ax.set_xticklabels(counts.index, rotation=45)
    plt.tight_layout()
    out_path = f"result/single/anomaly_scores_above_99th_per_year_{k}.png"
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
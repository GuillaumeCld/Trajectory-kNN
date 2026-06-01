import torch
import xarray as xr
import numpy as np
import os
import tempfile

torch.backends.cudnn.benchmark = False
torch.set_float32_matmul_precision('highest')


# ---------------------------------------------------------------------------
# Helper: compute norms in blocks, never loading more than block_size rows
# ---------------------------------------------------------------------------
@torch.no_grad()
def _blocked_norm_compute(X_memmap, block_size, dev, dtype, T, D):
    """
    Compute squared L2 norms row-by-row from a memmap array.
    X_memmap: numpy memmap of shape (T, D)
    Returns a CPU float32 tensor of shape (T,)
    """
    norms = torch.empty(T, dtype=dtype, device="cpu")
    for start in range(0, T, block_size):
        end = min(start + block_size, T)
        block = torch.from_numpy(X_memmap[start:end]).to(dtype).to(dev)
        norms[start:end] = (block * block).sum(dim=1).to("cpu")
    return norms


# ---------------------------------------------------------------------------
# Phase 0: stream NetCDF -> flat memmap (T, D) on disk
# Applies latitude weighting in-place during streaming.
# Never loads the full dataset into RAM.
# ---------------------------------------------------------------------------
def _stream_dataset_to_memmap(nc_path, var, tmp_dir, dtype=np.float32, chunk_time=256):
    """
    Stream a NetCDF variable to a flat (T, D) float32 memmap on disk.
    Latitude weighting is applied during streaming.

    Returns
    -------
    mmap      : np.memmap of shape (T, D)
    mmap_path : str, path to the memmap file (caller must delete)
    T, H, W   : int dimensions
    """
    ds = xr.open_dataset(nc_path)

    lat = ds["lat"].values if "lat" in ds else ds["latitude"].values
    lon_key = "lon" if "lon" in ds else "longitude"
    nlat = len(lat)
    nlon = len(ds[lon_key])
    D = nlat * nlon

    wlat = np.cos(np.deg2rad(lat))
    Ws = np.sqrt(np.tile(wlat, (nlon, 1)).T).astype(np.float32)   # (H, W)

    time_len = len(ds["time"])
    spatial_dims = ["lat", lon_key]

    mmap_path = os.path.join(tmp_dir, "X_flat.dat")
    mmap = np.memmap(mmap_path, dtype=dtype, mode="w+", shape=(time_len, D))

    for t_start in range(0, time_len, chunk_time):
        t_end = min(t_start + chunk_time, time_len)
        chunk = (
            ds[var]
            .isel(time=slice(t_start, t_end))
            .transpose("time", *spatial_dims)
            .values.astype(np.float32)
        )                                        # (chunk, H, W)
        chunk *= Ws                              # latitude weighting in-place
        mmap[t_start:t_end] = chunk.reshape(t_end - t_start, D)

    mmap.flush()
    ds.close()
    return mmap, mmap_path, time_len, nlat, nlon


# ---------------------------------------------------------------------------
# Phase 1: chunked GeMM -> write S tiles directly to a (T, T) memmap on disk
# RAM usage per step: O(q_batch * D + r_chunk * D + q_batch * r_chunk)
# ---------------------------------------------------------------------------
@torch.no_grad()
def _compute_spatial_distance_memmap(X_mmap, norms, T, D, tmp_dir,
                                     q_batch, r_chunk, dev, dtype):
    """
    Compute full pairwise spatial squared distances S[i,j] and write to disk.
    Exploits symmetry: only upper triangle is computed, lower is mirrored.

    Returns
    -------
    S_mmap      : np.memmap of shape (T, T), float32
    S_mmap_path : str
    """
    S_path = os.path.join(tmp_dir, "S_spatial.dat")
    S_mmap = np.memmap(S_path, dtype=np.float32, mode="w+", shape=(T, T))

    for row_start in range(0, T, q_batch):
        row_end = min(row_start + q_batch, T)
        rows = torch.from_numpy(X_mmap[row_start:row_end]).to(dtype).to(dev)
        row_norms = norms[row_start:row_end].to(dev)

        # Only compute upper triangle blocks (column_start >= row_start)
        for col_start in range(row_start, T, r_chunk):
            col_end = min(col_start + r_chunk, T)
            cols = torch.from_numpy(X_mmap[col_start:col_end]).to(dtype).to(dev)
            col_norms = norms[col_start:col_end].to(dev)

            block = (
                row_norms[:, None] + col_norms[None, :]
                - 2.0 * (rows @ cols.T)
            ).clamp(min=0.0).to("cpu").numpy()

            S_mmap[row_start:row_end, col_start:col_end] = block
            S_mmap[col_start:col_end, row_start:row_end] = block.T   # symmetry

        # Flush after each row block to avoid large dirty-page accumulation
        S_mmap.flush()

    return S_mmap, S_path


# ---------------------------------------------------------------------------
# Phase 2: recurrence over S memmap — only two diagonal bands read per step
# RAM usage: O(N) for distance vector + O(N * k) for kNN heap
# ---------------------------------------------------------------------------
def _recurrence_scores(S_mmap, T, traj_length, k, exclusion_zone, dtype):
    """
    Compute rarity scores using the sliding-window recurrence.
    Reads only two length-N diagonal bands of S per recurrence step.

    D_0(j)   = sum_{m=0}^{L-1} S[m, j+m]          (initialization)
    D_i(j)   = D_{i-1}(j-1) - S[i-1, j-1]
                             + S[i+L-1, j+L-1]     (recurrence, j >= 1)
    D_i(0)   = D_0(i)                              (symmetry)
    """
    L = traj_length
    N = T - L + 1
    scores = torch.empty(N, dtype=dtype)

    # --- Initialization: D_0 ---
    distances_traj = torch.zeros(N, dtype=dtype)
    for t_offset in range(L):
        # Read one diagonal band: S[t_offset, t_offset : t_offset + N]
        band = torch.from_numpy(
            np.array(S_mmap[t_offset, t_offset: t_offset + N])
        ).to(dtype)
        distances_traj += band

    distances_traj.clamp_(min=0.0)
    distances_traj0 = distances_traj.clone()   # keep D_0 for the symmetry fix

    # Score for i=0: mask exclusion zone then take topk
    masked = distances_traj.clone()
    masked[:exclusion_zone] = float("inf")
    scores[0] = torch.topk(masked, k=k, largest=False).values.mean()

    # --- Recurrence: i = 1 .. N-1 ---
    for i in range(1, N):
        # Band to subtract: S[i-1, 0 : N-1]  (length N-1)
        old_band = torch.from_numpy(
            np.array(S_mmap[i - 1, 0: N - 1])
        ).to(dtype)

        # Band to add: S[i+L-1, L : T]  (length N-1)
        new_band = torch.from_numpy(
            np.array(S_mmap[i + L - 1, L: T])
        ).to(dtype)

        distances_traj[1:] = (
            distances_traj[:-1] - old_band + new_band
        )
        distances_traj[0] = distances_traj0[i]   # symmetry: D_i(0) = D_0(i)
        distances_traj.clamp_(min=0.0)

        # Apply exclusion zone
        ez_lo = max(0, i - exclusion_zone + 1)
        ez_hi = min(N, i + exclusion_zone)
        distances_traj[ez_lo:ez_hi] = float("inf")

        scores[i] = torch.topk(distances_traj, k=k, largest=False).values.mean()

    return scores


# ---------------------------------------------------------------------------
# Public API — drop-in replacement for the original knn_scores
# ---------------------------------------------------------------------------
def knn_scores(
    nc_path,
    var,
    traj_length,
    k=10,
    q_batch=128,
    r_chunk=4096,
    device=None,
    dtype=torch.float32,
    exclusion_zone=-1,
    tmp_dir=None,
    keep_tmp=False,
    stream_chunk_time=256,
):
    """
    Out-of-core TRAKNN rarity scoring.

    Memory guarantee
    ----------------
    Peak RAM ≈ O(q_batch * D  +  r_chunk * D  +  q_batch * r_chunk  +  T)
    where D = H * W is the (weighted) spatial dimension.
    The full dataset X and the full distance matrix S are stored on disk.

    Parameters
    ----------
    nc_path           : path to NetCDF file
    var               : variable name inside the NetCDF
    traj_length       : trajectory duration d
    k                 : number of nearest neighbours
    q_batch           : row block size for GeMM (controls RAM during Phase 1)
    r_chunk           : column block size for GeMM (controls RAM during Phase 1)
    device            : torch device for GeMM ('cuda' or 'cpu'); auto-detected
    dtype             : floating point precision (default float32)
    exclusion_zone    : number of timesteps to exclude around self-match
                        (default: traj_length)
    tmp_dir           : directory for temporary memmap files; uses system tmp
                        if None
    keep_tmp          : if True, do not delete temporary files after completion
                        (useful for debugging or reusing S across runs)
    stream_chunk_time : number of time steps loaded at once when streaming
                        the NetCDF to the flat memmap (Phase 0)

    Returns
    -------
    scores : torch.Tensor of shape (T - traj_length + 1,) on CPU
    """
    if exclusion_zone == -1:
        exclusion_zone = traj_length

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    own_tmp = tmp_dir is None
    if own_tmp:
        tmp_dir = tempfile.mkdtemp(prefix="traknn_")

    mmap_path = None
    S_path = None

    try:
        # ------------------------------------------------------------------
        # Phase 0: stream NetCDF -> flat memmap on disk
        # ------------------------------------------------------------------
        print("[TRAKNN] Phase 0: streaming dataset to disk memmap...")
        X_mmap, mmap_path, T, H, W = _stream_dataset_to_memmap(
            nc_path, var, tmp_dir, chunk_time=stream_chunk_time
        )
        D = H * W
        print(f"[TRAKNN] Dataset shape: T={T}, H={H}, W={W}, D={D}")

        # ------------------------------------------------------------------
        # Phase 0b: compute norms from memmap (never full X in RAM)
        # ------------------------------------------------------------------
        print("[TRAKNN] Phase 0b: computing norms...")
        norms = _blocked_norm_compute(X_mmap, r_chunk, dev, dtype, T, D)

        # ------------------------------------------------------------------
        # Phase 1: chunked GeMM -> S memmap on disk
        # ------------------------------------------------------------------
        print("[TRAKNN] Phase 1: computing spatial distance matrix (out-of-core)...")
        S_mmap, S_path = _compute_spatial_distance_memmap(
            X_mmap, norms, T, D, tmp_dir, q_batch, r_chunk, dev, dtype
        )
        print(f"[TRAKNN] S matrix written to {S_path} ({os.path.getsize(S_path) / 1e9:.2f} GB)")

        # X memmap no longer needed after S is computed
        del X_mmap

        # ------------------------------------------------------------------
        # Phase 2: recurrence over S memmap -> rarity scores
        # ------------------------------------------------------------------
        print("[TRAKNN] Phase 2: computing trajectory distances and scores...")
        scores = _recurrence_scores(S_mmap, T, traj_length, k, exclusion_zone, dtype)

    finally:
        if not keep_tmp:
            for p in [mmap_path, S_path]:
                if p is not None and os.path.exists(p):
                    os.remove(p)
            if own_tmp and os.path.isdir(tmp_dir):
                try:
                    os.rmdir(tmp_dir)
                except OSError:
                    pass   # non-empty if keep_tmp=False but files remain

    print("[TRAKNN] Done.")
    return scores


# ---------------------------------------------------------------------------
# Convenience: run directly on a pre-loaded numpy array (no NetCDF needed)
# useful for synthetic scaling experiments
# ---------------------------------------------------------------------------
def compute_distances_and_scores(
    data,
    traj_length,
    k,
    q_batch,
    r_chunk,
    device,
    dtype,
    exclusion_zone,
    tmp_dir=None,
    keep_tmp=False,
):
    """
    Out-of-core TRAKNN on a numpy array or torch tensor.
    data: array-like of shape (T, H, W)
    """
    if exclusion_zone == -1:
        exclusion_zone = traj_length

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    own_tmp = tmp_dir is None
    if own_tmp:
        tmp_dir = tempfile.mkdtemp(prefix="traknn_")

    S_path = None
    mmap_path = None

    try:
        T, H, W = data.shape
        D = H * W

        # Write data to a memmap so Phase 1 can stream it
        mmap_path = os.path.join(tmp_dir, "X_flat.dat")
        X_mmap = np.memmap(mmap_path, dtype=np.float32, mode="w+", shape=(T, D))
        if isinstance(data, torch.Tensor):
            X_mmap[:] = data.reshape(T, D).numpy()
        else:
            X_mmap[:] = data.reshape(T, D)
        X_mmap.flush()

        norms = _blocked_norm_compute(X_mmap, r_chunk, dev, dtype, T, D)

        S_mmap, S_path = _compute_spatial_distance_memmap(
            X_mmap, norms, T, D, tmp_dir, q_batch, r_chunk, dev, dtype
        )
        del X_mmap

        scores = _recurrence_scores(S_mmap, T, traj_length, k, exclusion_zone, dtype)

    finally:
        if not keep_tmp:
            for p in [mmap_path, S_path]:
                if p is not None and os.path.exists(p):
                    os.remove(p)
            if own_tmp and os.path.isdir(tmp_dir):
                try:
                    os.rmdir(tmp_dir)
                except OSError:
                    pass

    return scores


if __name__ == "__main__":
    datapath = "Data/era5_msl_daily_eu.nc"
    parameter = "msl"
    traj_length = 7
    k = 10

    device = "cuda" if torch.cuda.is_available() else "cpu"


    scores = knn_scores(
        datapath,
        parameter,
        traj_length,
        k=k,
        q_batch=128,
        r_chunk=4096,
        device=device,
        dtype=torch.float32,
        exclusion_zone=traj_length,
        tmp_dir=None,
        keep_tmp=False,
        stream_chunk_time=256,
    )
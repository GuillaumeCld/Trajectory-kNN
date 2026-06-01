import torch
import xarray as xr
import numpy as np

torch.backends.cudnn.benchmark = False
torch.set_float32_matmul_precision('highest')


# ---------------------------------------------------------------------------
# Helper: compute a single row of the spatial distance matrix S
# S[i, :] = ||X[i]||^2 + ||X[:]||^2 - 2 * X[i] @ X[:].T
# computed in column chunks to avoid loading all of X at once.
# ---------------------------------------------------------------------------
@torch.no_grad()
def _compute_S_row(i, X, norms, T, r_chunk, dev, dtype):
    """
    Compute the full row S[i, :] of the spatial distance matrix.

    Parameters
    ----------
    i      : int, row index
    X      : torch.Tensor of shape (T, D) on dev — full dataset on device
             OR numpy array / memmap if dataset does not fit in RAM
    norms  : torch.Tensor of shape (T,) on CPU, precomputed squared norms
    T      : int, number of timesteps
    r_chunk: int, column chunk size (controls RAM if X is a memmap)
    dev    : torch.device
    dtype  : torch.dtype

    Returns
    -------
    row : torch.Tensor of shape (T,) on CPU
    """
    row = torch.empty(T, dtype=dtype, device="cpu")
    xi = X[i] if isinstance(X, torch.Tensor) else torch.from_numpy(
        np.array(X[i])).to(dtype).to(dev)
    norm_i = norms[i].to(dev)

    for col_start in range(0, T, r_chunk):
        col_end = min(col_start + r_chunk, T)
        if isinstance(X, torch.Tensor):
            xj = X[col_start:col_end]
        else:
            xj = torch.from_numpy(
                np.array(X[col_start:col_end])).to(dtype).to(dev)
        col_norms = norms[col_start:col_end].to(dev)
        block = (norm_i + col_norms - 2.0 * (xi @ xj.T)).clamp(min=0.0)
        row[col_start:col_end] = block.to("cpu")

    return row


# ---------------------------------------------------------------------------
# Helper: precompute norms in blocks
# ---------------------------------------------------------------------------
@torch.no_grad()
def _compute_norms(X, T, r_chunk, dev, dtype):
    norms = torch.empty(T, dtype=dtype, device="cpu")
    for start in range(0, T, r_chunk):
        end = min(start + r_chunk, T)
        block = X[start:end] if isinstance(X, torch.Tensor) else \
            torch.from_numpy(np.array(X[start:end])).to(dtype).to(dev)
        norms[start:end] = (block * block).sum(dim=1).to("cpu")
    return norms


# ---------------------------------------------------------------------------
# Core algorithm: circular buffer TRAKNN
# ---------------------------------------------------------------------------
def _traknn_buffer(X, T, D, traj_length, k, exclusion_zone,
                   r_chunk, dev, dtype):
    """
    TRAKNN with a circular row buffer of size L = traj_length.
    No disk I/O. Peak RAM = O(L * T + r_chunk * D).

    The buffer holds L consecutive rows of S.
    At recurrence step i we need:
      - S[i-1, :]   : the row to subtract  -> buffer slot (i-1) % L
      - S[i+L-1, :] : the row to add       -> same slot, overwritten just-in-time

    Since we are done with row i-1 before we need slot (i-1) % L for i+L-1,
    we can safely overwrite it.

    Buffer layout at step i (0-indexed):
      slot s = (i + L - 1) % L  holds row  i + L - 1   (freshly computed)
      slot s = (i - 1)    % L  holds row  i - 1        (about to be consumed)
      These are the same slot, confirming L rows suffice.
    """
    L = traj_length
    N = T - L + 1

    # --- Precompute norms ---
    norms = _compute_norms(X, T, r_chunk, dev, dtype)

    # --- Allocate circular buffer: L rows of length T ---
    # buf[s] = S[row_index] where row_index % L == s
    buf = torch.empty((L, T), dtype=dtype, device="cpu")

    # --- Fill the initial buffer: rows 0 .. L-1 ---
    # These are needed to initialize D_0
    for row_idx in range(L):
        buf[row_idx % L] = _compute_S_row(
            row_idx, X, norms, T, r_chunk, dev, dtype)

    # --- Initialization: D_0(j) = sum_{m=0}^{L-1} S[m, m+j] ---
    distances_traj = torch.zeros(N, dtype=dtype)
    for m in range(L):
        # S[m, m : m+N] is the diagonal band starting at (m, m)
        distances_traj += buf[m % L][m: m + N]
    distances_traj.clamp_(min=0.0)
    distances_traj0 = distances_traj.clone()   # save D_0 for symmetry fix

    scores = torch.empty(N, dtype=dtype)

    # Score for i=0
    masked = distances_traj.clone()
    masked[:exclusion_zone] = float("inf")
    scores[0] = torch.topk(masked, k=k, largest=False).values.mean()

    # --- Recurrence: i = 1 .. N-1 ---
    for i in range(1, N):
        # The new row we need is i+L-1.
        # Its buffer slot is (i+L-1) % L == (i-1) % L
        # which is exactly the slot holding row i-1 (no longer needed).
        new_row_idx = i + L - 1
        slot = new_row_idx % L   # == (i-1) % L

        # Overwrite the slot with row i+L-1
        buf[slot] = _compute_S_row(
            new_row_idx, X, norms, T, r_chunk, dev, dtype)

        # Recurrence update (vectorized, length N-1):
        # D_i(j) = D_{i-1}(j-1) - S[i-1, j-1] + S[i+L-1, j+L-1]  for j>=1
        old_slot = (i - 1) % L   # holds row i-1; same as slot, shown for clarity
        # S[i-1, 0:N-1]   -> buf[old_slot][0:N-1]
        # S[i+L-1, L:T]   -> buf[slot][L:T]
        distances_traj[1:] = (
            distances_traj[:-1]
            - buf[old_slot][0: N - 1]
            + buf[slot][L: T]
        )
        distances_traj[0] = distances_traj0[i]   # symmetry: D_i(0) = D_0(i)
        distances_traj.clamp_(min=0.0)

        # Exclusion zone
        ez_lo = max(0, i - exclusion_zone + 1)
        ez_hi = min(N, i + exclusion_zone)
        distances_traj[ez_lo:ez_hi] = float("inf")

        scores[i] = torch.topk(distances_traj, k=k, largest=False).values.mean()

    return scores


# ---------------------------------------------------------------------------
# Public API: load from NetCDF
# ---------------------------------------------------------------------------
def knn_scores(
    nc_path,
    var,
    traj_length,
    k=10,
    r_chunk=256,
    device=None,
    dtype=torch.float32,
    exclusion_zone=-1,
):
    """
    In-memory TRAKNN with circular row buffer. No disk writes.

    Peak RAM = O(traj_length * T  +  r_chunk * D)
    where T = number of timesteps, D = H * W (spatial dimension).

    Parameters
    ----------
    nc_path      : path to NetCDF file
    var          : variable name
    traj_length  : trajectory duration d
    k            : number of nearest neighbours
    r_chunk      : column chunk size for row GeMM computation.
                   Tune to fit GPU/CPU memory: larger = faster, more RAM.
    device       : 'cuda' or 'cpu' (auto-detected if None)
    dtype        : floating point precision
    exclusion_zone: timesteps excluded around self-match (default: traj_length)

    Returns
    -------
    scores : torch.Tensor of shape (T - traj_length + 1,) on CPU
    """
    if exclusion_zone == -1:
        exclusion_zone = traj_length
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    # --- Load dataset ---
    ds = xr.open_dataset(nc_path)
    lat = ds["lat"].values if "lat" in ds else ds["latitude"].values
    lon_key = "lon" if "lon" in ds else "longitude"
    nlat = len(lat)
    nlon = len(ds[lon_key])
    wlat = np.cos(np.deg2rad(lat))
    Ws = np.sqrt(np.tile(wlat, (nlon, 1)).T).astype(np.float32)

    data = (
        ds[var]
        .transpose("time", "lat", lon_key)
        .values.astype(np.float32)
    )
    data *= Ws
    ds.close()

    T, H, W = data.shape
    D = H * W
    print(f"[TRAKNN] T={T}, H={H}, W={W}, D={D}, L={traj_length}")
    print(f"[TRAKNN] Buffer RAM ≈ {traj_length * T * 4 / 1e6:.1f} MB  "
          f"(buffer) + {r_chunk * D * 4 / 1e6:.1f} MB (GeMM chunk)")

    # Move full dataset to device if it fits; otherwise keep as numpy
    # and _compute_S_row will stream chunks to device.
    X = torch.from_numpy(data.reshape(T, D)).to(dtype).to(dev)

    return _traknn_buffer(X, T, D, traj_length, k, exclusion_zone,
                          r_chunk, dev, dtype)


# ---------------------------------------------------------------------------
# Public API: run on a pre-loaded array (for synthetic scaling experiments)
# ---------------------------------------------------------------------------
def compute_distances_and_scores(
    data,
    traj_length,
    k,
    r_chunk,
    device,
    dtype,
    exclusion_zone,
):
    """
    In-memory TRAKNN on a numpy array or torch tensor of shape (T, H, W).
    """
    if exclusion_zone == -1:
        exclusion_zone = traj_length
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    T, H, W = data.shape
    D = H * W

    X = torch.from_numpy(data.reshape(T, D)).to(dtype).to(dev) \
        if not isinstance(data, torch.Tensor) \
        else data.reshape(T, D).to(dtype).to(dev)

    return _traknn_buffer(X, T, D, traj_length, k, exclusion_zone,
                          r_chunk, dev, dtype)

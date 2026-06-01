"""
TRAKNN — Batched Circular Buffer + Out-of-Core Dataset
=======================================================
The dataset X is never fully loaded into RAM or device memory.
It is streamed from a NetCDF file via dask in chunks, only during
norm computation and GeMM.

All other hot tensors (buffer, distance vector, scores) remain on
device exactly as in the in-memory version.

Peak device memory
------------------
  norms           : T * 4 bytes                      (on device)
  query rows Xi   : B * D * 4 bytes                  (on device, transient)
  column chunk Xj : r_chunk * D * 4 bytes            (on device, transient)
  buffer          : (L + B - 1) * T * 4 bytes        (on device)
  GeMM output     : B * T * 4 bytes                  (on device, transient)
  distance vector : T * 4 bytes                      (on device)
  scores          : N * 4 bytes                      (on device)

Peak host RAM
-------------
  one dask chunk  : r_chunk * D * 4 bytes            (transient)

batch_size B
------------
  B=1   -> row-by-row (correct but slow)
  B=128 -> near-original speed
  B=T   -> full S materialised in buffer (maximum speed, most device memory)
"""

import torch
import xarray as xr
import dask.array as da
import numpy as np

torch.backends.cudnn.benchmark = False
torch.set_float32_matmul_precision('highest')


# ---------------------------------------------------------------------------
# OOC dataset wrapper
# ---------------------------------------------------------------------------
class _OOCDataset:
    """
    Wraps a dask array of shape (T, D) and provides device-resident
    chunk access without ever loading the full array into RAM.
    """
    def __init__(
        self,
        data_da: da.Array,
        T: int,
        D: int,
        dev: torch.device,
        dtype: torch.dtype,
    ):
        self._da    = data_da
        self.T      = T
        self.D      = D
        self._dev   = dev
        self._dtype = dtype

    def get_rows(self, row_start: int, row_end: int) -> torch.Tensor:
        """Fetch rows [row_start, row_end) as (B, D) tensor on device."""
        chunk = np.asarray(self._da[row_start:row_end].compute())
        return torch.from_numpy(chunk).to(self._dtype).to(self._dev)


# ---------------------------------------------------------------------------
# Norms — streamed, result on device
# ---------------------------------------------------------------------------
@torch.no_grad()
def _compute_norms_ooc(ds: _OOCDataset, r_chunk: int) -> torch.Tensor:
    """
    Compute squared L2 norms for all T rows, streaming r_chunk rows at a time.

    Returns
    -------
    norms : (T,) float32 tensor on device
    """
    T = ds.T
    norms = torch.empty(T, dtype=ds._dtype, device=ds._dev)
    for start in range(0, T, r_chunk):
        end   = min(start + r_chunk, T)
        block = ds.get_rows(start, end)
        norms[start:end] = (block * block).sum(dim=1)
        del block
    return norms


# ---------------------------------------------------------------------------
# Batch of S rows — streamed query and column chunks, result on device
# ---------------------------------------------------------------------------
@torch.no_grad()
def _compute_S_rows_ooc(
    row_start: int,
    row_end: int,
    ds: _OOCDataset,
    norms: torch.Tensor,
    r_chunk: int,
) -> torch.Tensor:
    """
    Compute S[row_start:row_end, :] without X fully on device.

    Query rows Xi are fetched once (B * D) and kept on device for the
    inner column loop. Each column chunk Xj is fetched, used, discarded.

    Parameters
    ----------
    row_start, row_end : row slice to compute
    ds                 : OOC dataset wrapper
    norms              : (T,) on device
    r_chunk            : column chunk size

    Returns
    -------
    rows : (B, T) float32 tensor on device
    """
    T   = ds.T
    B   = row_end - row_start
    dev = ds._dev

    Xi     = ds.get_rows(row_start, row_end)    # (B, D) on device, fetched once
    norm_i = norms[row_start:row_end]

    rows = torch.empty((B, T), dtype=ds._dtype, device=dev)

    for col_start in range(0, T, r_chunk):
        col_end = min(col_start + r_chunk, T)
        Xj      = ds.get_rows(col_start, col_end)   # (chunk, D) on device
        norm_j  = norms[col_start:col_end]

        rows[:, col_start:col_end] = (
            norm_i[:, None] + norm_j[None, :]
            - 2.0 * (Xi @ Xj.T)
        ).clamp(min=0.0)

        del Xj

    del Xi
    return rows                                 # (B, T) on device


# ---------------------------------------------------------------------------
# Core: batched circular buffer recurrence — OOC dataset, fully on device
# Identical logic to the in-memory version; only S-row computation differs.
# ---------------------------------------------------------------------------
def _traknn_batched_buffer_ooc(
    ds: _OOCDataset,
    traj_length: int,
    k: int,
    exclusion_zone: int,
    r_chunk: int,
    batch_size: int,
) -> torch.Tensor:
    """
    TRAKNN batched buffer with OOC dataset.

    Parameters
    ----------
    ds             : _OOCDataset wrapping the lazy dask array
    traj_length    : trajectory duration L
    k              : number of nearest neighbours
    exclusion_zone : self-match exclusion radius
    r_chunk        : column chunk size for S row computation
    batch_size     : B, rows of S computed per GeMM batch

    Returns
    -------
    scores : (N,) float32 tensor on CPU
    """
    T      = ds.T
    L      = traj_length
    B      = batch_size
    N      = T - L + 1
    dtype  = ds._dtype
    dev    = ds._dev

    buf_size = L + B - 1

    # All hot tensors on device
    buf            = torch.empty((buf_size, T), dtype=dtype, device=dev)
    distances_traj = torch.zeros(N,             dtype=dtype, device=dev)
    scores         = torch.empty(N,             dtype=dtype, device=dev)

    print("[TRAKNN-OOC] Computing norms ...")
    norms = _compute_norms_ooc(ds, r_chunk)

    # Warm-up: fill buffer rows 0 .. L+B-2
    print("[TRAKNN-OOC] Warm-up GeMM ...")
    initial_rows = min(buf_size, T)
    buf[:initial_rows] = _compute_S_rows_ooc(
        0, initial_rows, ds, norms, r_chunk
    )

    # D_0(j) = sum_{m=0}^{L-1} S[m, m+j]
    for m in range(L):
        distances_traj += buf[m % buf_size, m: m + N]
    distances_traj.clamp_(min=0.0)
    distances_traj0 = distances_traj.clone()

    # Score for i=0
    masked = distances_traj.clone()
    masked[:exclusion_zone] = float("inf")
    scores[0] = torch.topk(masked, k=k, largest=False).values.mean()

    # Main loop
    for i_start in range(1, N, B):
        i_end    = min(i_start + B, N)
        actual_B = i_end - i_start

        # Recurrence — purely on device, no I/O
        for i in range(i_start, i_end):
            slot_sub = (i - 1)     % buf_size
            slot_add = (i + L - 1) % buf_size

            distances_traj[1:] = (
                distances_traj[:-1]
                - buf[slot_sub, 0: N - 1]
                + buf[slot_add, L: T]
            )
            distances_traj[0] = distances_traj0[i]
            distances_traj.clamp_(min=0.0)

            ez_lo = max(0, i - exclusion_zone + 1)
            ez_hi = min(N, i + exclusion_zone)
            distances_traj[ez_lo:ez_hi] = float("inf")

            scores[i] = torch.topk(
                distances_traj, k=k, largest=False
            ).values.mean()

        # Fetch next B rows from disk in one GeMM batch
        next_row_start = i_end + L - 1
        next_row_end   = min(next_row_start + actual_B, T)

        if next_row_start < T:
            new_rows = _compute_S_rows_ooc(
                next_row_start, next_row_end, ds, norms, r_chunk
            )
            for offset, row_idx in enumerate(
                range(next_row_start, next_row_end)
            ):
                buf[row_idx % buf_size] = new_rows[offset]
            del new_rows

    return scores.cpu()


# ---------------------------------------------------------------------------
# Dataset loader — dask-backed, latitude-weighted, never loaded into RAM
# ---------------------------------------------------------------------------
def _open_ooc(
    nc_path: str,
    var: str,
    dev: torch.device,
    dtype: torch.dtype,
    time_chunk: int = 256,
) -> tuple:
    """
    Open a NetCDF file lazily via dask. Nothing is loaded until
    .compute() is called on a slice inside _compute_S_rows_ooc.

    Returns
    -------
    ds      : _OOCDataset
    T, H, W : int dimensions
    """
    xr_ds = xr.open_dataset(nc_path, chunks={"time": time_chunk}, engine="netcdf4")

    lat_key = "lat" if "lat" in xr_ds else "latitude"
    lon_key = "lon" if "lon" in xr_ds else "longitude"

    lat  = xr_ds[lat_key].values
    nlat = len(lat)
    nlon = len(xr_ds[lon_key])
    T    = len(xr_ds["time"])

    wlat = np.cos(np.deg2rad(lat)).astype(np.float32)
    Ws   = np.sqrt(np.tile(wlat, (nlon, 1)).T)             # (H, W)

    data_da = (
        xr_ds[var]
        .transpose("time", lat_key, lon_key)
        .data
        .astype(np.float32)
    )
    H, W = int(data_da.shape[1]), int(data_da.shape[2])
    D    = H * W

    Ws_da   = da.from_array(Ws, chunks=Ws.shape)
    data_da = (data_da * Ws_da).reshape(T, D)               # lazy (T, D)

    xr_ds.close()
    return _OOCDataset(data_da, T, D, dev, dtype), T, H, W


# ---------------------------------------------------------------------------
# Public API: NetCDF path
# ---------------------------------------------------------------------------
def knn_scores(
    nc_path: str,
    var: str,
    traj_length: int,
    k: int = 10,
    r_chunk: int = 256,
    batch_size: int = 128,
    device: str = None,
    dtype: torch.dtype = torch.float32,
    exclusion_zone: int = -1,
    time_chunk: int = 256,
) -> torch.Tensor:
    """
    TRAKNN rarity scoring — out-of-core dataset, batched buffer.

    The dataset is never fully loaded into RAM or device memory.
    S is never materialised. All recurrence tensors stay on device.

    Peak device memory = O((L+B)*T + B*D + r_chunk*D)
    Peak host RAM      = O(r_chunk*D)

    Parameters
    ----------
    nc_path        : path to NetCDF file
    var            : variable name
    traj_length    : trajectory duration d
    k              : number of nearest neighbours
    r_chunk        : column chunk size for GeMM and norm streaming
    batch_size     : B, query rows per GeMM batch (default 128)
    device         : 'cuda' / 'cpu' (auto-detected if None)
    dtype          : torch.dtype (default float32)
    exclusion_zone : timesteps excluded around self-match (default: traj_length)
    time_chunk     : dask time chunk size when reading NetCDF

    Returns
    -------
    scores : torch.Tensor of shape (T - traj_length + 1,) on CPU
    """
    if exclusion_zone == -1:
        exclusion_zone = traj_length
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    print(f"[TRAKNN-OOC] Opening {nc_path} lazily ...")
    ds, T, H, W = _open_ooc(nc_path, var, dev, dtype, time_chunk)
    D = ds.D

    buf_mb   = (traj_length + batch_size - 1) * T * 4 / 1e6
    query_mb = batch_size * D * 4 / 1e6
    col_mb   = r_chunk * D * 4 / 1e6
    print(
        f"[TRAKNN-OOC] T={T}, H={H}, W={W}, D={D}, "
        f"L={traj_length}, B={batch_size}\n"
        f"[TRAKNN-OOC] Peak device memory: "
        f"buffer={buf_mb:.0f} MB  "
        f"query_rows={query_mb:.0f} MB  "
        f"col_chunk={col_mb:.0f} MB"
    )

    return _traknn_batched_buffer_ooc(
        ds, traj_length, k, exclusion_zone, r_chunk, batch_size
    )


# ---------------------------------------------------------------------------
# Public API: pre-loaded or dask array of shape (T, H, W)
# ---------------------------------------------------------------------------
def compute_distances_and_scores(
    data,
    traj_length: int,
    k: int,
    r_chunk: int = 256,
    batch_size: int = 128,
    device: str = None,
    dtype: torch.dtype = torch.float32,
    exclusion_zone: int = -1,
) -> torch.Tensor:
    """
    TRAKNN on a dask array, numpy array, or torch tensor of shape (T, H, W).
    Pass a dask array for the full OOC guarantee.
    """
    if exclusion_zone == -1:
        exclusion_zone = traj_length
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    if isinstance(data, da.Array):
        T, H, W = data.shape
        D       = H * W
        data_da = data.reshape(T, D)
    elif isinstance(data, torch.Tensor):
        T, H, W = data.shape
        D       = H * W
        data_da = da.from_array(data.numpy().reshape(T, D), chunks=(256, D))
    else:
        T, H, W = data.shape
        D       = H * W
        data_da = da.from_array(data.reshape(T, D), chunks=(256, D))

    ds = _OOCDataset(data_da, T, D, dev, dtype)

    return _traknn_batched_buffer_ooc(
        ds, traj_length, k, exclusion_zone, r_chunk, batch_size
    )
# ---------------------------------------------------------------------------
# Public API: pre-loaded array (numpy, torch, or dask)
# ---------------------------------------------------------------------------
def compute_distances_and_scores(
    data,
    traj_length,
    k,
    r_chunk=256,
    device=None,
    dtype=torch.float32,
    exclusion_zone=-1,
):
    """
    TRAKNN on any array-like of shape (T, H, W).
    Pass a dask array to keep the out-of-core guarantee.
    """
    if exclusion_zone == -1:
        exclusion_zone = traj_length
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    if isinstance(data, da.Array):
        T, H, W = data.shape
        lazy_ds = _LazyDataset(data.reshape(T, H * W), T, H * W, dev, dtype)
    elif isinstance(data, torch.Tensor):
        T, H, W = data.shape
        X = data.reshape(T, H * W).to(dtype).to(dev)
        lazy_ds = _LazyDataset(X, T, H * W, dev, dtype)
    else:
        T, H, W = data.shape
        X = torch.from_numpy(data.reshape(T, H * W)).to(dtype).to(dev)
        lazy_ds = _LazyDataset(X, T, H * W, dev, dtype)

    return _traknn_buffer(lazy_ds, traj_length, k, exclusion_zone, r_chunk)



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
        time_chunk=1024,
        batch_size=13000,
        r_chunk=4096,
        device=device,
        dtype=torch.float32,
        exclusion_zone=traj_length,
    )

    
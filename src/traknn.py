"""
TRAKNN — Three algorithmic variants
=====================================

Version 1 — Base (rarity_scoring_base)
  Dataset X and full spatial distance matrix S both in RAM.
  Batched GeMM on GPU to compute S, then recurrence on CPU.

Version 2 — Buffer (rarity_scoring_buffer)
  Dataset X in RAM. S never materialised.
  Circular buffer of L+B-1 rows on device.
  Batched GeMM on device, recurrence on device.
  One CPU<->device transfer only: final scores.

Version 3 — OOC with prefetch (rarity_scoring_ooc)
  Dataset X on disk (dask/NetCDF). S never materialised.
  Double-buffering pipeline: while GPU runs GeMM on batch i,
  CPU thread prefetches batch i+1 via dask into pinned host memory,
  then transfers asynchronously via a CUDA stream.
  Recurrence on device. One CPU<->device transfer: final scores.

Shared interface
----------------
All three expose:
  knn_scores(nc_path, var, traj_length, k, r_chunk, batch_size,
             device, dtype, exclusion_zone) -> Tensor (N,)
  compute_distances_and_scores(data, traj_length, k, r_chunk,
             batch_size, device, dtype, exclusion_zone) -> Tensor (N,)

Complexity
----------
              Time          Device RAM              Host RAM
  Base        O(D T^2)      O(D T + T^2)            O(D T + T^2)
  Buffer      O(D T^2)      O(D T + (L+B) T)        O(D T)
  OOC         O(D T^2 / B)  O((L+B) T + B D)        O(B D)  [pinned]
"""

import torch
import xarray as xr
import dask.array as da
import numpy as np
import threading

torch.backends.cudnn.benchmark = False
torch.set_float32_matmul_precision('highest')


# =============================================================================
# Shared helpers
# =============================================================================

@torch.no_grad()
def _compute_norms_inram(X: torch.Tensor, block_size: int) -> torch.Tensor:
    """
    Squared L2 norms of all rows of X (X already on device).
    Returns (T,) tensor on same device as X.
    """
    T = X.shape[0]
    norms = torch.empty(T, dtype=X.dtype, device=X.device)
    for start in range(0, T, block_size):
        end = min(start + block_size, T)
        norms[start:end] = (X[start:end] * X[start:end]).sum(dim=1)
    return norms


@torch.no_grad()
def _compute_norms_from_ram(
    X_cpu: torch.Tensor,
    block_size: int,
    dev: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Squared L2 norms of all rows of X_cpu (X stays in host RAM).
    Streams block_size rows to device at a time.
    X_cpu must be pinned (page-locked) for non_blocking transfers to work.
    Returns (T,) tensor on device.
    """
    T = X_cpu.shape[0]
    norms = torch.empty(T, dtype=dtype, device=dev)
    nb = X_cpu.is_pinned() and dev.type == "cuda"
    for start in range(0, T, block_size):
        end   = min(start + block_size, T)
        block = X_cpu[start:end].to(dtype).to(dev, non_blocking=nb)
        norms[start:end] = (block * block).sum(dim=1)
        del block
    return norms


@torch.no_grad()
def _compute_S_rows_inram(
    row_start: int,
    row_end: int,
    X: torch.Tensor,
    norms: torch.Tensor,
    r_chunk: int,
) -> torch.Tensor:
    """
    Compute S[row_start:row_end, :] using X already on device.
    GeMM runs on device; result is returned on CPU so the recurrence
    loop runs without GPU sync overhead.
    Used by versions 1 and 2.
    """
    T = X.shape[0]
    B = row_end - row_start
    rows = torch.empty((B, T), dtype=X.dtype, device="cpu")
    Xi     = X[row_start:row_end]
    norm_i = norms[row_start:row_end]
    for col_start in range(0, T, r_chunk):
        col_end = min(col_start + r_chunk, T)
        rows[:, col_start:col_end] = (
            norm_i[:, None] + norms[col_start:col_end][None, :]
            - 2.0 * (Xi @ X[col_start:col_end].T)
        ).clamp(min=0.0).cpu()
    return rows


@torch.no_grad()
def _compute_S_rows_from_ram(
    row_start: int,
    row_end: int,
    X_cpu: torch.Tensor,
    norms: torch.Tensor,
    rows_buf: torch.Tensor,
    xi_bufs: list,
    xj_buf: torch.Tensor,
    transfer_stream,
) -> torch.Tensor:
    """
    Compute S[row_start:row_end, :] with X in host RAM.
    Writes result into the pre-allocated rows_buf[:B] view.

    Double-buffered H2D pipeline (when X_cpu is pinned and transfer_stream
    is not None):
      - transfer_stream sends query sub-tile i+1 while the default stream
        runs GeMM for sub-tile i.
      - Column chunks are written into the pre-allocated xj_buf to avoid
        allocation per tile.

    All workspace tensors (rows_buf, xi_bufs, xj_buf) are pre-allocated
    once in the caller and reused across every call.

    Parameters
    ----------
    row_start, row_end : row slice to compute (B = row_end - row_start)
    X_cpu              : (T, D) pinned float32 tensor in host RAM
    norms              : (T,) tensor on device
    rows_buf           : (B_max, T) pre-allocated device tensor
    xi_bufs            : [buf0, buf1] two (q_batch, D) device tensors
    xj_buf             : (r_chunk, D) pre-allocated device tensor
    transfer_stream    : torch.cuda.Stream or None (sync fallback)

    Returns
    -------
    view of rows_buf[:B] on device — valid until next call
    """
    T       = X_cpu.shape[0]
    D       = X_cpu.shape[1]
    B       = row_end - row_start
    q_batch = xi_bufs[0].shape[0]
    r_chunk = xj_buf.shape[0]
    dev     = rows_buf.device
    dtype   = rows_buf.dtype

    use_async = (transfer_stream is not None)
    sub_starts = list(range(0, B, q_batch))

    if use_async:
        # Prefetch first query sub-tile
        actual0 = min(q_batch, B)
        with torch.cuda.stream(transfer_stream):
            xi_bufs[0][:actual0].copy_(
                X_cpu[row_start: row_start + actual0], non_blocking=True
            )

        for idx, sub_start in enumerate(sub_starts):
            cur_buf  = idx % 2
            next_buf = (idx + 1) % 2
            sub_end  = min(sub_start + q_batch, B)
            actual   = sub_end - sub_start

            # Prefetch next query sub-tile while GeMM runs
            if idx + 1 < len(sub_starts):
                ns       = sub_starts[idx + 1]
                n_actual = min(q_batch, B - ns)
                with torch.cuda.stream(transfer_stream):
                    xi_bufs[next_buf][:n_actual].copy_(
                        X_cpu[row_start + ns: row_start + ns + n_actual],
                        non_blocking=True
                    )

            # Ensure query tile is on device before GeMM
            torch.cuda.current_stream(dev).wait_stream(transfer_stream)

            Xi     = xi_bufs[cur_buf][:actual]
            norm_i = norms[row_start + sub_start: row_start + sub_end]

            for col_start in range(0, T, r_chunk):
                col_end  = min(col_start + r_chunk, T)
                col_size = col_end - col_start
                # Write into pre-allocated column buffer — no new allocation
                xj_buf[:col_size].copy_(
                    X_cpu[col_start:col_end], non_blocking=True
                )
                Xj     = xj_buf[:col_size]
                norm_j = norms[col_start:col_end]
                rows_buf[sub_start:sub_end, col_start:col_end] = (
                    norm_i[:, None] + norm_j[None, :]
                    - 2.0 * (Xi @ Xj.T)
                ).clamp(min=0.0)

        torch.cuda.synchronize(dev)

    else:
        # Synchronous fallback
        for sub_start in range(0, B, q_batch):
            sub_end  = min(sub_start + q_batch, B)
            actual   = sub_end - sub_start
            abs_start = row_start + sub_start

            xi_bufs[0][:actual].copy_(X_cpu[abs_start: abs_start + actual])
            Xi     = xi_bufs[0][:actual]
            norm_i = norms[abs_start: abs_start + actual]

            for col_start in range(0, T, r_chunk):
                col_end  = min(col_start + r_chunk, T)
                col_size = col_end - col_start
                xj_buf[:col_size].copy_(X_cpu[col_start:col_end])
                Xj     = xj_buf[:col_size]
                norm_j = norms[col_start:col_end]
                rows_buf[sub_start:sub_end, col_start:col_end] = (
                    norm_i[:, None] + norm_j[None, :]
                    - 2.0 * (Xi @ Xj.T)
                ).clamp(min=0.0)

    return rows_buf[:B]   # view, no copy


def _recurrence(
    buf: torch.Tensor,
    distances_traj: torch.Tensor,
    distances_traj0: torch.Tensor,
    scores: torch.Tensor,
    staging: torch.Tensor,
    cols: torch.Tensor,
    i_start: int,
    i_end: int,
    L: int,
    N: int,
    T: int,
    k: int,
    exclusion_zone: int,
    buf_size: int,
):
    """
    Apply recurrence for steps i_start .. i_end-1 with a batched topk.

    Pre-allocated workspaces (no allocations inside hot loop):
      staging : (B, N) device tensor — accumulates distance vectors
      cols    : (1, N) device tensor — column indices for exclusion mask

    The recurrence update is sequential (step i depends on i-1), so
    the distance vector is updated one step at a time. However, instead
    of calling topk at every step (causing one GPU sync per step), we:

      1. Accumulate all B distance vectors into staging.
      2. Apply the exclusion zone mask to staging at once.
      3. Call torch.topk once on the entire staging matrix (dim=1).

    This reduces B GPU syncs per batch to one.
    """
    actual_B = i_end - i_start
    dev      = buf.device

    # View into pre-allocated staging for this (possibly smaller) batch
    stg = staging[:actual_B]

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
        stg[i - i_start] = distances_traj

    # Exclusion zone mask — reuses pre-allocated cols tensor
    ez      = exclusion_zone
    abs_idx = torch.arange(i_start, i_end, device=dev)      # (actual_B,)
    lo = (abs_idx - ez + 1).clamp(min=0).unsqueeze(1)       # (actual_B, 1)
    hi = (abs_idx + ez    ).clamp(max=N).unsqueeze(1)       # (actual_B, 1)
    stg.masked_fill_((cols >= lo) & (cols < hi), float("inf"))

    # One batched topk — one GPU sync for the whole batch
    scores[i_start:i_end] = torch.topk(
        stg, k=k, dim=1, largest=False
    ).values.mean(dim=1)


def _load_netcdf(nc_path, var):
    """Load full dataset from NetCDF. Returns (T, H, W) float32 numpy array
    with latitude weighting applied."""
    ds = xr.open_dataset(nc_path)
    lat_key = "lat" if "lat" in ds else "latitude"
    lon_key = "lon" if "lon" in ds else "longitude"
    lat  = ds[lat_key].values
    nlon = len(ds[lon_key])
    wlat = np.cos(np.deg2rad(lat)).astype(np.float32)
    Ws   = np.sqrt(np.tile(wlat, (nlon, 1)).T)
    data = (
        ds[var].transpose("time", lat_key, lon_key).values.astype(np.float32)
    )
    ds.close()
    data *= Ws
    return data


def _open_dask(nc_path, var, time_chunk=256):
    """
    Open NetCDF lazily via dask. Returns (data_da, T, H, W) where
    data_da is a lazy (T, D) dask array with latitude weighting applied.
    """
    xr_ds = xr.open_dataset(nc_path, chunks={"time": time_chunk}, engine="netcdf4")
    lat_key = "lat" if "lat" in xr_ds else "latitude"
    lon_key = "lon" if "lon" in xr_ds else "longitude"
    lat  = xr_ds[lat_key].values
    nlat = len(lat)
    nlon = len(xr_ds[lon_key])
    T    = len(xr_ds["time"])
    wlat = np.cos(np.deg2rad(lat)).astype(np.float32)
    Ws   = np.sqrt(np.tile(wlat, (nlon, 1)).T)
    data_da = (
        xr_ds[var].transpose("time", lat_key, lon_key).data.astype(np.float32)
    )
    H, W = int(data_da.shape[1]), int(data_da.shape[2])
    D    = H * W
    Ws_da   = da.from_array(Ws, chunks=Ws.shape)
    data_da = (data_da * Ws_da).reshape(T, D)
    xr_ds.close()
    return data_da, T, H, W


# =============================================================================
# Version 1 — Base: X and S both in RAM, batched GeMM on GPU
# =============================================================================

@torch.no_grad()
def _base_compute_S(X_dev, norms_dev, T, r_chunk, q_batch, dtype):
    """
    Compute the full (T, T) spatial distance matrix S.
    Batched GeMM on device, result stored in CPU RAM.
    Exploits symmetry: only upper triangle computed.
    """
    S = torch.empty((T, T), dtype=dtype, device="cpu")
    for row_start in range(0, T, q_batch):
        row_end  = min(row_start + q_batch, T)
        Xi       = X_dev[row_start:row_end]
        norm_i   = norms_dev[row_start:row_end]
        for col_start in range(row_start, T, r_chunk):
            col_end  = min(col_start + r_chunk, T)
            Xj       = X_dev[col_start:col_end]
            norm_j   = norms_dev[col_start:col_end]
            block    = (
                norm_i[:, None] + norm_j[None, :]
                - 2.0 * (Xi @ Xj.T)
            ).clamp(min=0.0).cpu()
            S[row_start:row_end, col_start:col_end] = block
            S[col_start:col_end, row_start:row_end] = block.T
    return S                                    # (T, T) on CPU


def _base_recurrence(S, T, L, N, k, exclusion_zone, dtype):
    """
    Recurrence over the full S matrix (on CPU).
    Identical logic to original algorithm.
    """
    scores = torch.empty(N, dtype=dtype)

    distances_traj0 = torch.zeros(N, dtype=dtype)
    for t_offset in range(L):
        distances_traj0 += S[t_offset, t_offset: t_offset + N]
    distances_traj0.clamp_(min=0.0)

    distances_traj = distances_traj0.clone()

    masked = distances_traj.clone()
    masked[:exclusion_zone] = float("inf")
    scores[0] = torch.topk(masked, k=k, largest=False).values.mean()

    for i in range(1, N):
        distances_traj[1:] = (
            distances_traj[:-1]
            - S[i - 1, 0: N - 1]
            + S[i + L - 1, L: T]
        )
        distances_traj[0] = distances_traj0[i]
        distances_traj.clamp_(min=0.0)
        ez_lo = max(0, i - exclusion_zone + 1)
        ez_hi = min(N, i + exclusion_zone)
        distances_traj[ez_lo:ez_hi] = float("inf")
        scores[i] = torch.topk(distances_traj, k=k, largest=False).values.mean()

    return scores


def rarity_scoring_base(
    data: np.ndarray,
    traj_length: int,
    k: int,
    r_chunk: int = 4096,
    q_batch: int = 128,
    device: str = None,
    dtype: torch.dtype = torch.float32,
    exclusion_zone: int = -1,
) -> torch.Tensor:
    """
    Version 1 — Base algorithm.

    Dataset X and full S matrix both in RAM.
    GeMM computed on device in (q_batch x r_chunk) tiles.
    Recurrence on CPU.

    Peak RAM  : O(T*D + T^2)
    Peak VRAM : O(q_batch*D + r_chunk*D)   [tiles only, not full S]

    Parameters
    ----------
    data         : (T, H, W) float32 numpy array, latitude-weighted
    traj_length  : trajectory duration L
    k            : number of nearest neighbours
    r_chunk      : column tile size for GeMM
    q_batch      : row tile size for GeMM
    device       : compute device for GeMM (default: auto)
    dtype        : floating point precision
    exclusion_zone: self-match exclusion (default: traj_length)

    Returns
    -------
    scores : (N,) float32 tensor on CPU
    """
    if exclusion_zone == -1:
        exclusion_zone = traj_length
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    T, H, W = data.shape
    D = H * W
    N = T - traj_length + 1

    print(
        f"[TRAKNN-Base] T={T}, D={D}, L={traj_length}\n"
        f"[TRAKNN-Base] Peak RAM: X={T*D*4/1e6:.0f} MB  "
        f"S={T*T*4/1e6:.0f} MB"
    )

    X_dev  = torch.from_numpy(data.reshape(T, D)).to(dtype).to(dev)
    norms  = _compute_norms_inram(X_dev, q_batch)

    S = _base_compute_S(X_dev, norms, T, r_chunk, q_batch, dtype)
    del X_dev

    return _base_recurrence(S, T, traj_length, N, k, exclusion_zone, dtype)


# =============================================================================
# Version 2 — Buffer: X in RAM, S never materialised, all hot tensors on device
# =============================================================================

def rarity_scoring_buffer(
    data: np.ndarray,
    traj_length: int,
    k: int,
    q_batch: int = 128,
    r_chunk: int = 256,
    batch_size: int = 128,
    device: str = None,
    dtype: torch.dtype = torch.float32,
    exclusion_zone: int = -1,
) -> torch.Tensor:
    """
    Version 2 — Batched circular buffer, X in host RAM.

    Dataset X stays in host RAM throughout. GeMM tiles are transferred
    to device on demand (q_batch query rows + r_chunk column rows at a time).
    S is never materialised. Circular buffer of L+B-1 rows lives on device.
    Recurrence and scoring run on device. Only final scores transferred to CPU.

    Peak RAM  : O(T*D)                         [X in host RAM]
    Peak VRAM : O((L+B)*T + q_batch*D + r_chunk*D)

    Parameters
    ----------
    data         : (T, H, W) float32 numpy array, latitude-weighted
    traj_length  : trajectory duration L
    k            : number of nearest neighbours
    q_batch      : row tile size for GeMM (query rows sent to device at once)
    r_chunk      : column tile size for GeMM
    batch_size   : B, rows of S per GeMM batch (controls buffer refresh rate)
    device       : compute device (default: auto)
    dtype        : floating point precision
    exclusion_zone: self-match exclusion (default: traj_length)

    Returns
    -------
    scores : (N,) float32 tensor on CPU
    """
    if exclusion_zone == -1:
        exclusion_zone = traj_length
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    T, H, W = data.shape
    D  = H * W
    L  = traj_length
    B  = batch_size
    N  = T - L + 1
    buf_size = L + B      # must hold L+B consecutive rows

    print(
        f"[TRAKNN-Buffer] T={T}, D={D}, L={L}, B={B}\n"
        f"[TRAKNN-Buffer] Peak RAM:  X={T*D*4/1e6:.0f} MB  "
        f"buffer={(buf_size*T*4)/1e6:.0f} MB\n"
        f"[TRAKNN-Buffer] Peak VRAM: "
        f"tiles={(q_batch+r_chunk)*D*4/1e6:.0f} MB"
    )

    # X stays in host RAM. Pinning enables non_blocking H2D transfers.
    X_cpu = torch.from_numpy(data.reshape(T, D)).to(dtype)
    if dev.type == "cuda":
        X_cpu = X_cpu.pin_memory()
    # row wise storage for coalesced access in GeMM
    X_cpu = X_cpu.contiguous()
    # Norms streamed to device once, kept there for GeMM reuse
    norms = _compute_norms_from_ram(X_cpu, r_chunk, dev, dtype)

    # ------------------------------------------------------------------
    # Pre-allocate all workspaces once — zero allocations in the hot loop
    # ------------------------------------------------------------------
    # Circular buffer and recurrence tensors on device
    buf            = torch.empty((buf_size, T), dtype=dtype, device=dev)
    distances_traj = torch.zeros(N,             dtype=dtype, device=dev)
    scores         = torch.empty(N,             dtype=dtype, device=dev)

    # Staging matrix for batched topk: (B, N) — reused every batch
    staging = torch.empty((B, N), dtype=dtype, device=dev)

    # Column index tensor for exclusion zone mask — constant, allocated once
    cols = torch.arange(N, device=dev).unsqueeze(0)          # (1, N)

    # GeMM workspace: output rows buffer (B, T), two query sub-tile
    # buffers for double-buffering, one column tile buffer
    rows_buf = torch.empty((buf_size, T), dtype=dtype, device=dev)  # buf_size >= B (warm-up needs L+B-1 rows)
    xi_bufs  = [
        torch.empty((q_batch, D), dtype=dtype, device=dev),
        torch.empty((q_batch, D), dtype=dtype, device=dev),
    ]
    xj_buf   = torch.empty((r_chunk, D), dtype=dtype, device=dev)

    # CUDA transfer stream — created once, reused every GeMM call
    transfer_stream = torch.cuda.Stream(dev) if dev.type == "cuda" else None

    # ------------------------------------------------------------------
    # Warm-up: fill buffer rows 0 .. buf_size-1
    # ------------------------------------------------------------------
    initial = min(buf_size, T)
    buf[:initial] = _compute_S_rows_from_ram(
        0, initial, X_cpu, norms,
        rows_buf, xi_bufs, xj_buf, transfer_stream
    )

    # D_0(j) = sum_{m=0}^{L-1} S[m, m+j]
    for m in range(L):
        distances_traj += buf[m % buf_size, m: m + N]
    distances_traj.clamp_(min=0.0)
    distances_traj0 = distances_traj.clone()

    # Score i=0: use first row of staging as a temporary
    staging[0] = distances_traj
    staging[0, :exclusion_zone] = float("inf")
    scores[0] = torch.topk(staging[0], k=k, largest=False).values.mean()

    # ------------------------------------------------------------------
    # Main loop — all workspaces reused, zero allocations per iteration
    # ------------------------------------------------------------------
    for i_start in range(1, N, B):
        i_end    = min(i_start + B, N)
        actual_B = i_end - i_start

        _recurrence(
            buf, distances_traj, distances_traj0, scores,
            staging, cols,
            i_start, i_end, L, N, T, k, exclusion_zone, buf_size
        )

        next_row_start = i_end + L - 1
        next_row_end   = min(next_row_start + actual_B, T)
        if next_row_start < T:
            new_rows = _compute_S_rows_from_ram(
                next_row_start, next_row_end,
                X_cpu, norms,
                rows_buf, xi_bufs, xj_buf, transfer_stream
            )
            for offset, row_idx in enumerate(
                range(next_row_start, next_row_end)
            ):
                buf[row_idx % buf_size] = new_rows[offset]

    return scores.cpu()


# =============================================================================
# Version 3 — OOC with CPU prefetch + async GPU transfer
# =============================================================================

class _Prefetcher:
    """
    Double-buffer prefetcher.

    A background thread calls dask.compute() on the next row batch
    while the GPU is executing the current GeMM. The result is staged
    in a pinned CPU buffer and transferred to device asynchronously
    via a dedicated CUDA stream.

    Usage:
        pf = _Prefetcher(data_da, T, D, B, dev, dtype)
        pf.start(next_row_start, next_row_end)   # kick off first prefetch
        ...
        tensor_on_dev = pf.result()              # blocks until ready
        pf.start(next_next_row_start, ...)       # kick off next
    """

    def __init__(
        self,
        data_da: da.Array,
        T: int,
        D: int,
        B: int,
        dev: torch.device,
        dtype: torch.dtype,
    ):
        self._da    = data_da
        self.T      = T
        self.D      = D
        self.B      = B
        self._dev   = dev
        self._dtype = dtype

        # Pinned host buffer — reused across batches to avoid allocation overhead
        self._host_buf = torch.empty((B, D), dtype=dtype).pin_memory()

        # Dedicated CUDA stream for async H2D transfers
        self._stream = torch.cuda.Stream(dev) if dev.type == "cuda" else None

        # Device staging tensor — result handed to the caller
        self._dev_buf = torch.empty((B, D), dtype=dtype, device=dev)

        self._thread: threading.Thread = None
        self._actual_B: int = 0
        self._error = None

    def start(self, row_start: int, row_end: int):
        """Launch background thread to prefetch rows [row_start, row_end)."""
        actual_B = row_end - row_start
        self._actual_B = actual_B
        self._error    = None

        def _fetch():
            try:
                chunk = np.asarray(
                    self._da[row_start:row_end].compute()
                ).astype(np.float32)
                # Write into pinned buffer (host-side, no GPU involved)
                self._host_buf[:actual_B].copy_(
                    torch.from_numpy(chunk)
                )
            except Exception as e:
                self._error = e

        self._thread = threading.Thread(target=_fetch, daemon=True)
        self._thread.start()

    def result(self) -> torch.Tensor:
        """
        Wait for the background fetch to complete, then transfer to device.
        Returns (actual_B, D) tensor on device.
        """
        if self._thread is not None:
            self._thread.join()
        if self._error is not None:
            raise self._error

        actual_B = self._actual_B

        if self._stream is not None:
            # Async H2D transfer on dedicated stream
            with torch.cuda.stream(self._stream):
                self._dev_buf[:actual_B].copy_(
                    self._host_buf[:actual_B], non_blocking=True
                )
            # Synchronise before the GeMM uses this data
            self._stream.synchronize()
        else:
            self._dev_buf[:actual_B].copy_(self._host_buf[:actual_B])

        return self._dev_buf[:actual_B]          # (actual_B, D) on device


@torch.no_grad()
def _compute_norms_ooc(
    data_da: da.Array,
    T: int,
    D: int,
    r_chunk: int,
    dev: torch.device,
    dtype: torch.dtype,
    host_buf: torch.Tensor,
    dev_buf: torch.Tensor,
    stream,
) -> torch.Tensor:
    """
    Stream rows in r_chunk blocks, compute norms on device.
    Reuses pre-allocated pinned host_buf (r_chunk, D) and dev_buf (r_chunk, D).
    Returns (T,) tensor on device.
    """
    norms  = torch.empty(T, dtype=dtype, device=dev)
    use_nb = (stream is not None)
    for start in range(0, T, r_chunk):
        end      = min(start + r_chunk, T)
        actual   = end - start
        # Write dask chunk into pinned host buffer — no new allocation
        host_buf[:actual].copy_(
            torch.from_numpy(np.asarray(data_da[start:end].compute()).astype(np.float32))
        )
        if use_nb:
            with torch.cuda.stream(stream):
                dev_buf[:actual].copy_(host_buf[:actual], non_blocking=True)
            stream.synchronize()
        else:
            dev_buf[:actual].copy_(host_buf[:actual])
        norms[start:end] = (dev_buf[:actual] * dev_buf[:actual]).sum(dim=1)
    return norms


@torch.no_grad()
def _compute_S_rows_from_Xi(
    Xi_dev: torch.Tensor,
    norm_i: torch.Tensor,
    norms: torch.Tensor,
    T: int,
    r_chunk: int,
    data_da: da.Array,
    rows_buf: torch.Tensor,
    col_host_buf: torch.Tensor,
    col_dev_buf: torch.Tensor,
    stream,
) -> torch.Tensor:
    """
    Compute S[row_start:row_end, :] given Xi already on device.
    All workspace tensors are pre-allocated — no allocations inside.

    Parameters
    ----------
    Xi_dev       : (B, D) tensor on device — query rows
    norm_i       : (B,) tensor on device
    norms        : (T,) tensor on device
    T            : number of timesteps
    r_chunk      : column chunk size
    data_da      : lazy (T, D) dask array
    rows_buf     : (B, T) pre-allocated device tensor — written in-place
    col_host_buf : (r_chunk, D) pinned CPU tensor — column staging
    col_dev_buf  : (r_chunk, D) device tensor — column staging
    stream       : CUDA transfer stream or None

    Returns
    -------
    view of rows_buf[:B] on device
    """
    B      = Xi_dev.shape[0]
    use_nb = (stream is not None)

    for col_start in range(0, T, r_chunk):
        col_end  = min(col_start + r_chunk, T)
        col_size = col_end - col_start

        # Load column chunk into pinned host buffer — no new allocation
        col_host_buf[:col_size].copy_(
            torch.from_numpy(
                np.asarray(data_da[col_start:col_end].compute()).astype(np.float32)
            )
        )
        # Transfer to device — async if pinned
        if use_nb:
            with torch.cuda.stream(stream):
                col_dev_buf[:col_size].copy_(
                    col_host_buf[:col_size], non_blocking=True
                )
            torch.cuda.current_stream(Xi_dev.device).wait_stream(stream)
        else:
            col_dev_buf[:col_size].copy_(col_host_buf[:col_size])

        Xj     = col_dev_buf[:col_size]
        norm_j = norms[col_start:col_end]
        rows_buf[:B, col_start:col_end] = (
            norm_i[:, None] + norm_j[None, :]
            - 2.0 * (Xi_dev @ Xj.T)
        ).clamp(min=0.0)

    return rows_buf[:B]   # view, on device


def rarity_scoring_ooc(
    data_da: da.Array,
    T: int,
    traj_length: int,
    k: int,
    r_chunk: int = 256,
    batch_size: int = 128,
    device: str = None,
    dtype: torch.dtype = torch.float32,
    exclusion_zone: int = -1,
) -> torch.Tensor:
    """
    Version 3 — OOC with CPU prefetch and async GPU transfer.

    Dataset X never fully in RAM or device memory.
    While the GPU runs GeMM on batch i, the CPU prefetches batch i+1
    from dask into pinned memory and transfers it asynchronously.

    Peak VRAM : O((L+B)*T + B*D + r_chunk*D)
    Peak RAM  : O(B*D)   [one pinned prefetch buffer]

    Parameters
    ----------
    data_da      : lazy (T, D) dask array, float32, latitude-weighted
    T            : number of timesteps
    traj_length  : trajectory duration L
    k            : number of nearest neighbours
    r_chunk      : column chunk size for inner GeMM loop
    batch_size   : B, query rows per GeMM call
    device       : compute device (default: auto)
    dtype        : floating point precision
    exclusion_zone: self-match exclusion (default: traj_length)

    Returns
    -------
    scores : (N,) float32 tensor on CPU
    """
    if exclusion_zone == -1:
        exclusion_zone = traj_length
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    D  = data_da.shape[1]
    L  = traj_length
    B  = batch_size
    N  = T - L + 1
    buf_size = L + B      # must hold L+B consecutive rows

    print(
        f"[TRAKNN-OOC] T={T}, D={D}, L={L}, B={B}\n"
        f"[TRAKNN-OOC] Peak VRAM: "
        f"buffer={(buf_size*T*4)/1e6:.0f} MB  "
        f"query_rows={B*D*4/1e6:.0f} MB  "
        f"col_chunk={r_chunk*D*4/1e6:.0f} MB\n"
        f"[TRAKNN-OOC] Peak RAM (pinned): {B*D*4/1e6:.0f} MB"
    )

    # ------------------------------------------------------------------
    # Pre-allocate all workspaces — zero allocations in the hot loop
    # ------------------------------------------------------------------
    # Recurrence tensors on device
    buf            = torch.empty((buf_size, T), dtype=dtype, device=dev)
    distances_traj = torch.zeros(N,             dtype=dtype, device=dev)
    scores         = torch.empty(N,             dtype=dtype, device=dev)

    # Staging + exclusion mask tensor (shared with _recurrence)
    staging = torch.empty((B, N), dtype=dtype, device=dev)
    cols    = torch.arange(N, device=dev).unsqueeze(0)       # (1, N)

    # GeMM output buffer (B, T) on device — returned as a view
    rows_buf = torch.empty((buf_size, T), dtype=dtype, device=dev)  # buf_size >= B

    # Column staging: pinned host + device pair for async H2D
    col_host_buf = torch.empty((r_chunk, D), dtype=dtype).pin_memory()         if dev.type == "cuda" else torch.empty((r_chunk, D), dtype=dtype)
    col_dev_buf  = torch.empty((r_chunk, D), dtype=dtype, device=dev)

    # Dedicated CUDA stream for column H2D transfers
    col_stream = torch.cuda.Stream(dev) if dev.type == "cuda" else None

    # Prefetcher: background thread + pinned buffer for query row batches
    pf = _Prefetcher(data_da, T, D, B, dev, dtype)

    # ------------------------------------------------------------------
    # Norms: streamed from dask using pre-allocated buffers
    # ------------------------------------------------------------------
    print("[TRAKNN-OOC] Computing norms ...")
    norms = _compute_norms_ooc(
        data_da, T, D, r_chunk, dev, dtype,
        col_host_buf, col_dev_buf, col_stream
    )

    # ------------------------------------------------------------------
    # Warm-up: fill buffer rows 0 .. buf_size-1
    # ------------------------------------------------------------------
    print("[TRAKNN-OOC] Warm-up ...")
    initial = min(buf_size, T)
    filled  = 0
    while filled < initial:
        sub_end = min(filled + B, initial)
        sub_B   = sub_end - filled
        pf.start(filled, sub_end)
        Xi_dev = pf.result()                    # (sub_B, D) on device
        norm_i = norms[filled:sub_end]
        rows   = _compute_S_rows_from_Xi(
            Xi_dev, norm_i, norms, T, r_chunk, data_da,
            rows_buf, col_host_buf, col_dev_buf, col_stream
        )
        for offset in range(sub_B):
            buf[(filled + offset) % buf_size] = rows[offset]
        filled = sub_end

    # --- D_0 ---
    for m in range(L):
        distances_traj += buf[m % buf_size, m: m + N]
    distances_traj.clamp_(min=0.0)
    distances_traj0 = distances_traj.clone()

    # Score i=0: reuse staging[0] as temporary — no clone
    staging[0] = distances_traj
    staging[0, :exclusion_zone] = float("inf")
    scores[0] = torch.topk(staging[0], k=k, largest=False).values.mean()

    # ------------------------------------------------------------------
    # Main loop — double-buffering pipeline
    #
    #   CPU thread : prefetch query rows for batch i+1 via dask
    #   GPU        : recurrence + batched topk for batch i
    #              : GeMM for batch i once prefetch completes
    # ------------------------------------------------------------------
    next_rs = buf_size                           # first row not yet in buffer
    next_re = min(next_rs + B, T)
    if next_rs < T:
        pf.start(next_rs, next_re)

    for i_start in range(1, N, B):
        i_end    = min(i_start + B, N)
        actual_B = i_end - i_start

        # Recurrence + batched topk — overlaps with prefetch on CPU thread
        _recurrence(
            buf, distances_traj, distances_traj0, scores,
            staging, cols,
            i_start, i_end, L, N, T, k, exclusion_zone, buf_size
        )

        if next_rs < T:
            Xi_dev = pf.result()                # blocks until prefetch done
            norm_i = norms[next_rs:next_re]

            # Kick off next prefetch before running GeMM (hides I/O latency)
            nn_rs = next_re + L - 1
            nn_re = min(nn_rs + actual_B, T)
            if nn_rs < T:
                pf.start(nn_rs, nn_re)

            # GeMM — writes into pre-allocated rows_buf, no allocation
            rows = _compute_S_rows_from_Xi(
                Xi_dev, norm_i, norms, T, r_chunk, data_da,
                rows_buf, col_host_buf, col_dev_buf, col_stream
            )
            for offset, row_idx in enumerate(range(next_rs, next_re)):
                buf[row_idx % buf_size] = rows[offset]

            next_rs, next_re = nn_rs, nn_re

    return scores.cpu()


# =============================================================================
# Public API — unified entry points
# =============================================================================

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
    version: str = "buffer",
    q_batch: int = 128,
    time_chunk: int = 256,
) -> torch.Tensor:
    """
    TRAKNN rarity scoring from a NetCDF file.

    Parameters
    ----------
    nc_path      : path to NetCDF file
    var          : variable name
    traj_length  : trajectory duration d
    k            : number of nearest neighbours
    r_chunk      : column block size for GeMM
    batch_size   : B, rows of S per GeMM call (versions 2 and 3)
    device       : 'cuda' / 'cpu' (auto-detected if None)
    dtype        : torch.dtype (default float32)
    exclusion_zone: self-match exclusion (default: traj_length)
    version      : 'base' | 'buffer' | 'ooc'
    q_batch      : row tile size for version 1 base GeMM
    time_chunk   : dask time chunk for version 3

    Returns
    -------
    scores : (T - traj_length + 1,) float32 tensor on CPU
    """
    if version == "base":
        data = _load_netcdf(nc_path, var)
        return rarity_scoring_base(
            data, traj_length, k, r_chunk, q_batch, device, dtype, exclusion_zone
        )
    elif version == "buffer":
        data = _load_netcdf(nc_path, var)
        return rarity_scoring_buffer(
            data, traj_length, k, q_batch, r_chunk, batch_size, device, dtype, exclusion_zone
        )
    elif version == "ooc":
        data_da, T, H, W = _open_dask(nc_path, var, time_chunk)
        return rarity_scoring_ooc(
            data_da, T, traj_length, k, r_chunk, batch_size,
            device, dtype, exclusion_zone
        )
    else:
        raise ValueError(f"version must be 'base', 'buffer', or 'ooc', got '{version}'")


def compute_distances_and_scores(
    data,
    traj_length: int,
    k: int,
    r_chunk: int = 256,
    batch_size: int = 128,
    device: str = None,
    dtype: torch.dtype = torch.float32,
    exclusion_zone: int = -1,
    version: str = "buffer",
    q_batch: int = 128,
) -> torch.Tensor:
    """
    TRAKNN on a numpy array, torch tensor, or dask array of shape (T, H, W).

    Parameters
    ----------
    data         : array-like (T, H, W) — numpy/torch -> versions 1 & 2;
                   dask array -> version 3
    version      : 'base' | 'buffer' | 'ooc'
    (other params same as knn_scores)

    Returns
    -------
    scores : (T - traj_length + 1,) float32 tensor on CPU
    """
    if isinstance(data, da.Array):
        T, H, W = data.shape
        D = H * W
        return rarity_scoring_ooc(
            data.reshape(T, D), T, traj_length, k, r_chunk,
            batch_size, device, dtype, exclusion_zone
        )

    # numpy or torch -> materialise as numpy for versions 1 and 2
    if isinstance(data, torch.Tensor):
        arr = data.numpy()
    else:
        arr = np.asarray(data)

    if version == "base":
        return rarity_scoring_base(
            arr, traj_length, k, r_chunk, q_batch, device, dtype, exclusion_zone
        )
    elif version == "buffer":
        return rarity_scoring_buffer(
            arr, traj_length, k, q_batch, r_chunk, batch_size, device, dtype, exclusion_zone
        )
    elif version == "ooc":
        T, H, W = arr.shape
        D = H * W
        data_da = da.from_array(arr.reshape(T, D), chunks=(batch_size, D))
        return rarity_scoring_ooc(
            data_da, T, traj_length, k, r_chunk,
            batch_size, device, dtype, exclusion_zone
        )
    else:
        raise ValueError(f"version must be 'base', 'buffer', or 'ooc', got '{version}'")

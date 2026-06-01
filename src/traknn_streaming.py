"""
TRAKNN — Versions 2 and 3, streaming design (X does not fit in VRAM)
====================================================================

Both versions compute the rarity score with an exact recurrence-based
kNN. The spatial distance matrix S is never materialised: B rows of S
are produced per batch by a single large GeMM and cycled through a
circular buffer of L+B rows on device.

The two versions share ONE compute kernel (_compute_S_batch) and differ
only in where a tile of X comes from:

  Version 2 (buffer) : X pinned in host RAM. Tiles sliced and copied to
                       device synchronously (non_blocking on pinned mem).
                       Transfer is <1% of GeMM time for B>=64, so no
                       prefetch machinery is used.

  Version 3 (ooc)    : X on disk via dask. Query rows and column tiles
                       are prefetched by background threads so disk I/O
                       overlaps GPU compute. Robust on slow disks,
                       negligible overhead on fast disks.

Data re-read cost (inherent to not materialising S)
---------------------------------------------------
Each GeMM batch reads ALL T rows of X (in r_chunk tiles) to form the
column operand. With T/B batches, X is read T/B times in total.
  - V2: re-reads come from host RAM (cheap).
  - V3: re-reads come from disk. On a slow disk this dominates runtime.
The only lever is B: larger B -> fewer passes. The functions print the
expected number of full-dataset passes so users can tune B.

Complexity
----------
  Version 2:  time O(D*T^2)     VRAM O((L+B)*T + B*D + r_chunk*D)   RAM O(D*T)
  Version 3:  time O(D*T^2)     VRAM O((L+B)*T + B*D + r_chunk*D)   RAM O((B+r_chunk)*D)
  (V3 wall-clock also bounded below by (T/B) full-dataset disk reads.)
"""

import threading
import torch
import xarray as xr
import numpy as np

torch.backends.cudnn.benchmark = False
torch.set_float32_matmul_precision("highest")


# =============================================================================
# Shared: recurrence with batched topk (device-resident)
# =============================================================================

def _recurrence(
    buf:             torch.Tensor,   # (buf_size, T) device
    distances_traj:  torch.Tensor,   # (N,) device, in-place
    distances_traj0: torch.Tensor,   # (N,) device, D_0 reference
    scores:          torch.Tensor,   # (N,) device, in-place
    staging:         torch.Tensor,   # (B, N) device workspace
    cols:            torch.Tensor,   # (1, N) device constant
    i_start: int, i_end: int,
    L: int, N: int, T: int,
    k: int, exclusion_zone: int, buf_size: int,
):
    """
    Recurrence for steps i_start..i_end-1 with a single batched topk.

    The distance-vector update is sequential (step i depends on i-1), so
    the B vectors are produced one at a time into `staging` with no GPU
    sync. The exclusion zone is applied to the whole (B,N) staging matrix
    at once, and one topk(dim=1) call yields all B scores — one sync per
    batch instead of one per step.
    """
    actual_B = i_end - i_start
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

    abs_idx = torch.arange(i_start, i_end, device=buf.device)
    lo = (abs_idx - exclusion_zone + 1).clamp(min=0).unsqueeze(1)
    hi = (abs_idx + exclusion_zone    ).clamp(max=N).unsqueeze(1)
    stg.masked_fill_((cols >= lo) & (cols < hi), float("inf"))

    scores[i_start:i_end] = torch.topk(
        stg, k=k, dim=1, largest=False
    ).values.mean(dim=1)


# =============================================================================
# Shared: GeMM kernel — produces B rows of S given a column-tile provider
# =============================================================================

@torch.no_grad()
def _compute_S_batch(
    Xi_dev:   torch.Tensor,    # (B, D) query rows on device
    norm_i:   torch.Tensor,    # (B,)   query-row norms on device
    norms:    torch.Tensor,    # (T,)   all norms on device
    rows_buf: torch.Tensor,    # (buf_size, T) device, written in-place
    T:        int,
    r_chunk:  int,
    get_col_tile,              # callable(cs, ce) -> (ce-cs, D) tensor on device
) -> torch.Tensor:
    """
    Compute S[rows, :] = ||Xi||^2 + ||Xj||^2 - 2 Xi·Xj^T for all columns,
    one large GeMM per column tile.

    `get_col_tile(cs, ce)` returns the column block X[cs:ce] already on
    device. V2 slices pinned RAM; V3 returns a prefetched dask tile.
    Identical math for both — the only difference is tile provenance.

    Returns a view rows_buf[:B] on device.
    """
    B = Xi_dev.shape[0]
    for cs in range(0, T, r_chunk):
        ce     = min(cs + r_chunk, T)
        Xj     = get_col_tile(cs, ce)          # (ce-cs, D) on device
        norm_j = norms[cs:ce]
        rows_buf[:B, cs:ce] = (
            norm_i[:, None] + norm_j[None, :]
            - 2.0 * (Xi_dev @ Xj.T)
        ).clamp(min=0.0)
    return rows_buf[:B]


@torch.no_grad()
def _compute_S_batch_pipe(
    Xi_dev:   torch.Tensor,    # (B, D) query rows on device
    norm_i:   torch.Tensor,    # (B,) query-row norms on device
    norms:    torch.Tensor,    # (T,) all norms on device
    rows_buf: torch.Tensor,    # (buf_size, T) device, written in-place
    T:        int,
    r_chunk:  int,
    cp:       "_ColumnPipeline",   # started pipeline over all column tiles
) -> torch.Tensor:
    """
    Same math as _compute_S_batch, but the column operand is supplied by a
    three-stage _ColumnPipeline. After issuing each GeMM, record_compute(j)
    lets the pipeline gate device-slot reuse on this GeMM's completion.
    """
    B = Xi_dev.shape[0]
    col_starts = cp._starts
    for j, cs in enumerate(col_starts):
        ce     = min(cs + r_chunk, T)
        Xj     = cp.next_tile(j)               # (ce-cs, D) on device, gated
        norm_j = norms[cs:ce]
        rows_buf[:B, cs:ce] = (
            norm_i[:, None] + norm_j[None, :]
            - 2.0 * (Xi_dev @ Xj.T)
        ).clamp(min=0.0)
        cp.record_compute(j)                   # mark GeMM enqueued on this slot
    return rows_buf[:B]


# =============================================================================
# Dataset loaders
# =============================================================================

def _load_netcdf(nc_path, var):
    """
    Load the full dataset into a (T, H, W) float32 numpy array, mean-centered
    per grid point (computed in float64).

    Centering is mathematically exact for distances — ||(Xi-mu)-(Xj-mu)||^2 =
    ||Xi-Xj||^2 — but it shrinks the values so that the float32 norm-identity
    ||Xi||^2 + ||Xj||^2 - 2<Xi,Xj> no longer suffers catastrophic cancellation.
    Without it, raw fields (e.g. sea-level pressure ~1e5 Pa) give norms ~1e15,
    whose float32 difference cannot resolve true distances ~1e6 and collapses
    to zero.
    """
    ds = xr.open_dataset(nc_path)
    lat_key = "lat" if "lat" in ds else "latitude"
    lon_key = "lon" if "lon" in ds else "longitude"
    data = ds[var].transpose("time", lat_key, lon_key).values.astype(np.float64)
    ds.close()
    data -= data.mean(axis=0, keepdims=True)          # center per grid point (f64)
    return np.ascontiguousarray(data, dtype=np.float32)


class _TileReader:
    """
    Fast out-of-core reader backed by a (T, D) float32 binary file laid out
    in row-major (time-major) order — exactly the access pattern the GeMM
    column operand needs.

    On first construction it converts the NetCDF variable into a flat binary
    file `<nc_path>.<var>.traknn.f32` (skipped if already present and the
    right size). Subsequent reads are pure np.memmap slices: a contiguous
    block memmap[rs:re] is read at full disk bandwidth, the netCDF4 C layer
    and xarray CF-decoding overhead are gone, and the OS page cache + readahead
    accelerate repeated passes automatically. Pages are evictable, so the file
    may be far larger than host RAM (the Case-B regime).

    Reads are thread-safe: np.memmap slicing is a plain mmap read with no
    shared mutable handle state, so no lock is needed and the GIL is released
    during the underlying memcpy — letting disk reads overlap GPU compute.
    """

    def __init__(self, nc_path, var, time_chunk=512):
        import os
        ds = xr.open_dataset(nc_path)
        lat = "lat" if "lat" in ds else "latitude"
        lon = "lon" if "lon" in ds else "longitude"
        v   = ds[var].transpose("time", lat, lon)
        self.T = int(v.shape[0])
        self.H = int(v.shape[1])
        self.W = int(v.shape[2])
        self.D = self.H * self.W

        self._path = f"{nc_path}.{var}.traknn.f32"
        expected   = self.T * self.D * 4

        # A completed build is marked by a `.done` sentinel written AFTER the
        # data file is fully flushed. Without the sentinel we always rebuild,
        # so a partial/zero-filled file from a crashed run (np.memmap w+ mode
        # zero-fills immediately) can never be mistaken for valid data.
        done_path  = self._path + ".done"
        valid = (os.path.exists(self._path)
                 and os.path.getsize(self._path) == expected
                 and os.path.exists(done_path))

        if not valid:
            # Build into a temp file, flush, then atomically rename. A crash
            # mid-build leaves only the temp file, never a half-written target.
            tmp_path = self._path + ".tmp"
            if os.path.exists(done_path):
                os.remove(done_path)

            # Pass 1: per-grid-point mean over all time, accumulated in float64.
            # Centering is exact for distances but removes the large offset that
            # makes the float32 norm-identity catastrophically cancel (raw SLP
            # ~1e5 -> norms ~1e15 -> distances ~1e6 lost). Streamed so peak RAM
            # is O(time_chunk * D) plus the (D,) accumulator.
            mean = np.zeros(self.D, dtype=np.float64)
            for cs in range(0, self.T, time_chunk):
                ce = min(cs + time_chunk, self.T)
                block = v.isel(time=slice(cs, ce)).values.reshape(ce - cs, self.D)
                mean += block.astype(np.float64).sum(axis=0)
            mean /= self.T
            mean32 = mean.astype(np.float32)

            # Pass 2: write centered data to the memmap.
            mm = np.memmap(tmp_path, dtype=np.float32, mode="w+",
                           shape=(self.T, self.D))
            for cs in range(0, self.T, time_chunk):
                ce = min(cs + time_chunk, self.T)
                block = v.isel(time=slice(cs, ce)).values.reshape(ce - cs, self.D)
                mm[cs:ce] = (block.astype(np.float64) - mean).astype(np.float32)
            mm.flush()
            del mm
            os.replace(tmp_path, self._path)      # atomic on POSIX
            with open(done_path, "w") as fhd:     # sentinel marks completion
                fhd.write("ok")
        ds.close()

        # Read-only memmap reused for all tile reads
        self._mm = np.memmap(self._path, dtype=np.float32, mode="r",
                             shape=(self.T, self.D))

    def read(self, rs, re):
        """Return (re-rs, D) float32 contiguous array for rows [rs, re)."""
        # np.array(...) forces a contiguous copy out of the mmap so the
        # caller (pinned host buffer copy_) gets a stable, owned buffer.
        return np.array(self._mm[rs:re], dtype=np.float32, order="C")

    def read_into(self, rs, re, dst_tensor):
        """
        Copy mmap rows [rs, re) directly into dst_tensor[:re-rs] (a pinned
        CPU torch tensor), avoiding the intermediate numpy allocation that
        `read` makes. dst_tensor must be (>=re-rs, D) float32 on CPU.
        The numpy view of the pinned tensor shares its memory, so this is a
        single mmap->pinned memcpy (GIL released during the copy).
        """
        n = re - rs
        dst_np = dst_tensor.numpy()                 # view into pinned memory
        np.copyto(dst_np[:n], self._mm[rs:re])      # one contiguous memcpy

    def close(self):
        del self._mm


def _open_reader(nc_path, var):
    """Open a fast (T,D) memmap reader; return (reader, T, H, W)."""
    reader = _TileReader(nc_path, var)
    return reader, reader.T, reader.H, reader.W


# =============================================================================
# Version 2 — X pinned in host RAM, synchronous tile streaming
# =============================================================================

def rarity_scoring_buffer(
    data:           np.ndarray,
    traj_length:    int,
    k:              int,
    r_chunk:        int          = 256,
    batch_size:     int          = 128,
    device:         str          = None,
    dtype:          torch.dtype  = torch.float32,
    exclusion_zone: int          = -1,
) -> torch.Tensor:
    """
    Version 2 — circular buffer, X pinned in host RAM.

    Each batch transfers all B query rows to device once, then streams
    column tiles synchronously (non_blocking on pinned memory). With
    B>=64 the column transfer is <1% of GeMM time, so no prefetch is used.

    Peak RAM  : O(T*D)                          [X pinned in host RAM]
    Peak VRAM : O((L+B)*T + B*D + r_chunk*D)
    """
    if exclusion_zone == -1:
        exclusion_zone = traj_length
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    T, H, W  = data.shape
    D        = H * W
    L        = traj_length
    B        = batch_size
    N        = T - L + 1
    buf_size = L + B
    n_pass   = -(-T // B)   # ceil(T/B) full-dataset passes over host RAM

    print(
        f"[TRAKNN-Buffer] T={T} D={D} L={L} B={B}  ({n_pass} host-RAM passes)\n"
        f"[TRAKNN-Buffer] RAM  X={T*D*4/1e6:.0f} MB | "
        f"VRAM buf={buf_size*T*4/1e6:.0f} MB "
        f"Xi={B*D*4/1e6:.0f} MB Xj={r_chunk*D*4/1e6:.0f} MB"
    )

    # Pin X so non_blocking H2D works
    X_cpu = torch.from_numpy(data.reshape(T, D)).to(dtype)
    if dev.type == "cuda":
        X_cpu = X_cpu.pin_memory()
    nb = (dev.type == "cuda")

    # Device workspaces (all pre-allocated, reused every batch)
    buf            = torch.empty((buf_size, T), dtype=dtype, device=dev)
    distances_traj = torch.zeros(N,             dtype=dtype, device=dev)
    scores         = torch.empty(N,             dtype=dtype, device=dev)
    staging        = torch.empty((B, N),        dtype=dtype, device=dev)
    cols           = torch.arange(N, device=dev).unsqueeze(0)
    rows_buf       = torch.empty((buf_size, T), dtype=dtype, device=dev)
    Xi_dev         = torch.empty((B, D),        dtype=dtype, device=dev)
    Xj_dev         = torch.empty((r_chunk, D),  dtype=dtype, device=dev)

    # Column-tile provider: slice pinned RAM, copy to device buffer
    def get_col_tile(cs, ce):
        sz = ce - cs
        Xj_dev[:sz].copy_(X_cpu[cs:ce], non_blocking=nb)
        return Xj_dev[:sz]

    # Norms: one streaming pass
    norms = torch.empty(T, dtype=dtype, device=dev)
    for cs in range(0, T, r_chunk):
        ce = min(cs + r_chunk, T)
        Xj = get_col_tile(cs, ce)
        norms[cs:ce] = (Xj * Xj).sum(dim=1)

    # Fill rows [rs, re) of S into the circular buffer
    def fill(rs, re):
        re = min(re, T)
        if rs >= T:
            return
        B_local = re - rs
        Xi_dev[:B_local].copy_(X_cpu[rs:re], non_blocking=nb)
        rows = _compute_S_batch(
            Xi_dev[:B_local], norms[rs:re], norms,
            rows_buf, T, r_chunk, get_col_tile
        )
        idx = torch.arange(rs, re, device=dev) % buf_size
        buf.index_copy_(0, idx, rows[:B_local])

    # Warm-up: fill exactly rows 0 .. buf_size-1 (cap so row buf_size does
    # not wrap and overwrite row 0, which is still needed at step i=1)
    for rs in range(0, buf_size, B):
        fill(rs, min(rs + B, buf_size))

    # D_0
    for m in range(L):
        distances_traj += buf[m % buf_size, m: m + N]
    distances_traj.clamp_(min=0.0)
    distances_traj0 = distances_traj.clone()

    # Score i=0
    staging[0] = distances_traj
    staging[0, :exclusion_zone] = float("inf")
    scores[0] = torch.topk(staging[0], k=k, largest=False).values.mean()

    # Main loop
    for i_start in range(1, N, B):
        i_end = min(i_start + B, N)
        _recurrence(
            buf, distances_traj, distances_traj0, scores,
            staging, cols,
            i_start, i_end, L, N, T, k, exclusion_zone, buf_size
        )
        fill(i_end + L - 1, i_end + B + L - 1)

    return scores.cpu()


# =============================================================================
# Version 3 — X on disk, prefetched tile streaming
# =============================================================================

class _Prefetcher:
    """
    FIFO double-buffered prefetcher over a bounded tile reader.

    Contract (strict FIFO, at most 2 requests in flight):
        push(rs, re)    enqueue a fetch for rows [rs, re)
        pop() -> tensor  block until the OLDEST pending fetch is on device,
                         return it, and free its slot

    Two host+device buffer pairs (ping-pong). A background thread fills the
    pinned host buffer; pop() does the async H2D on a dedicated stream and
    syncs the default stream before returning. Typical use:

        pf.push(batch0)
        for each batch:
            if more: pf.push(batch_next)   # prefetch next
            x = pf.pop()                   # consume current (overlapped)
            ... use x ...
    """

    def __init__(self, reader, rows_max, D, dev, dtype):
        self._reader = reader
        self._dev    = dev
        self._dtype = dtype
        self._stream = torch.cuda.Stream(dev) if dev.type == "cuda" else None
        mk = lambda: (torch.empty((rows_max, D), dtype=dtype).pin_memory()
                      if dev.type == "cuda"
                      else torch.empty((rows_max, D), dtype=dtype))
        self._host  = [mk(), mk()]
        self._dev_b = [
            torch.empty((rows_max, D), dtype=dtype, device=dev),
            torch.empty((rows_max, D), dtype=dtype, device=dev),
        ]
        # FIFO of in-flight requests: each entry = (slot, actual, thread, err_box)
        self._queue   = []
        self._next_slot = 0     # slot to use for the next push

    def push(self, rs, re):
        """Enqueue a background fetch for rows [rs, re)."""
        actual = re - rs
        slot   = self._next_slot
        self._next_slot = 1 - slot          # ping-pong
        err_box = [None]
        host = self._host[slot]

        def _fetch():
            try:
                self._reader.read_into(rs, re, host)          # mmap -> pinned
            except Exception as e:
                err_box[0] = e

        th = threading.Thread(target=_fetch, daemon=True)
        th.start()
        self._queue.append((slot, actual, th, err_box))

    def pop(self):
        """Block until the oldest pending fetch is on device; return it."""
        slot, actual, th, err_box = self._queue.pop(0)
        th.join()
        if err_box[0] is not None:
            raise err_box[0]
        if self._stream is not None:
            with torch.cuda.stream(self._stream):
                self._dev_b[slot][:actual].copy_(
                    self._host[slot][:actual], non_blocking=True
                )
            torch.cuda.current_stream(self._dev).wait_stream(self._stream)
        else:
            self._dev_b[slot][:actual].copy_(self._host[slot][:actual])
        return self._dev_b[slot][:actual]


class _ColumnPipeline:
    """
    Three-stage pipeline that streams the column operand of one GeMM batch.

    Concurrent stages, one column tile per step:
      READ  (worker thread)  : disk -> pinned host[slot]
      COPY  (copy stream)    : pinned host[slot] -> device dev[slot]   (DMA)
      GEMM  (default stream) : caller consumes dev[slot]

    A ring of `depth` host+device slots (depth=3) lets the three stages
    occupy distinct slots at once. Correctness is enforced by:

      * read_done[slot]      (threading.Event): COPY waits for READ
      * h2d_done[slot]        (CUDA event): GEMM waits for COPY; the reader
                              also blocks on it (.synchronize) before refilling
                              a reused host slot, so the DMA out of that host
                              buffer is guaranteed complete first
      * compute_done[slot]   (CUDA event): COPY waits for the prior GEMM that
                              used dev[slot] (tile j-depth) to finish, so an
                              in-use device tile is never overwritten

    The last gate (compute_done) is the critical one: it makes device-slot
    reuse safe without a full device sync.

    Usage:
        cp = _ColumnPipeline(reader, col_starts, r_chunk, D, dev, dtype)
        cp.start()
        for j in range(len(col_starts)):
            Xj = cp.next_tile(j)        # device tensor for tile j
            ... GeMM with Xj ...        # caller records compute_done inside
        cp.join()
    """

    def __init__(self, reader, col_starts, r_chunk, D, dev, dtype, depth=3):
        self._reader = reader
        self._starts = col_starts
        self._r      = r_chunk
        self._dev    = dev
        self._dtype  = dtype
        self._n      = len(col_starts)
        self._depth  = min(depth, max(1, self._n))
        d = self._depth
        self._copy_stream = torch.cuda.Stream(dev) if dev.type == "cuda" else None

        mk_host = lambda: (torch.empty((r_chunk, D), dtype=dtype).pin_memory()
                           if dev.type == "cuda"
                           else torch.empty((r_chunk, D), dtype=dtype))
        self._host = [mk_host() for _ in range(d)]
        self._devb = [torch.empty((r_chunk, D), dtype=dtype, device=dev)
                      for _ in range(d)]

        self._read_done    = [threading.Event() for _ in range(d)]
        self._h2d_done     = [torch.cuda.Event() for _ in range(d)] if dev.type == "cuda" else [None]*d
        self._compute_done = [torch.cuda.Event() for _ in range(d)] if dev.type == "cuda" else [None]*d

        self._worker = None
        self._err    = [None]

    # ---- READ stage: disk -> pinned host ring, in tile order ----
    def _run_worker(self):
        try:
            for j in range(self._n):
                slot = j % self._depth
                # Before refilling a reused host slot, block until the DMA
                # that copied its previous contents (tile j-depth) has
                # physically completed. h2d_done is recorded on the copy
                # stream; .synchronize() blocks this CPU thread until done.
                if j >= self._depth and self._h2d_done[slot] is not None:
                    self._h2d_done[slot].synchronize()
                cs = self._starts[j]
                ce = min(cs + self._r, self._reader.T)
                # Single mmap -> pinned-host memcpy (no intermediate numpy)
                self._reader.read_into(cs, ce, self._host[slot])
                self._read_done[slot].set()
        except Exception as e:
            self._err[0] = e
            for ev in self._read_done:
                ev.set()                           # unblock any waiter

    def start(self):
        """Start a fresh pass over col_starts, reusing the pre-allocated
        buffers. Safe to call repeatedly (after join() of the prior pass)."""
        self._err[0] = None
        for ev in self._read_done:
            ev.clear()
        # Device buffers from the previous pass are guaranteed free: the prior
        # fill() called join() (disk worker done) and the caller synchronised
        # compute before the next fill. Events are one-shot-per-record, so a
        # fresh record() in this pass overrides any stale state.
        self._worker = threading.Thread(target=self._run_worker, daemon=True)
        self._worker.start()

    def next_tile(self, j):
        """
        Return the device tensor for column tile j. The caller must, after
        issuing its GeMM on the default stream, call record_compute(j) so the
        pipeline can safely reuse this device slot.
        """
        if self._err[0] is not None:
            raise self._err[0]
        slot = j % self._depth
        cs = self._starts[j]
        ce = min(cs + self._r, self._reader.T)
        size = ce - cs

        # COPY stage
        if self._copy_stream is not None:
            # Gate: do not overwrite dev[slot] until the GEMM that used it
            # `depth` steps ago has completed.
            if j >= self._depth:
                self._copy_stream.wait_event(self._compute_done[slot])
            # Wait for READ of this tile to finish (host side)
            self._read_done[slot].wait()
            self._read_done[slot].clear()
            if self._err[0] is not None:
                raise self._err[0]
            with torch.cuda.stream(self._copy_stream):
                self._devb[slot][:size].copy_(
                    self._host[slot][:size], non_blocking=True
                )
                self._h2d_done[slot].record(self._copy_stream)
            # The DMA reads host[slot]; once the H2D event is recorded the
            # host buffer is consumed by the copy engine. The next READ into
            # this host slot must wait for the DMA to actually finish — which
            # is guaranteed once the COMPUTE stream has waited on h2d_done and
            # run. We free the host slot after the GEMM waits (below).
            # GEMM stage gating: default stream waits on this tile's H2D.
            torch.cuda.current_stream(self._dev).wait_event(self._h2d_done[slot])
        else:
            self._read_done[slot].wait()
            self._read_done[slot].clear()
            if self._err[0] is not None:
                raise self._err[0]
            self._devb[slot][:size].copy_(self._host[slot][:size])

        return self._devb[slot][:size]

    def record_compute(self, j):
        """Record that the GEMM consuming tile j has been enqueued on the
        default stream, so the copy stream can gate dev-slot reuse on it."""
        if self._copy_stream is not None:
            slot = j % self._depth
            self._compute_done[slot].record(
                torch.cuda.current_stream(self._dev)
            )

    def join(self):
        if self._worker is not None:
            self._worker.join()


def rarity_scoring_ooc(
    reader,
    T:              int,
    traj_length:    int,
    k:              int,
    r_chunk:        int          = 256,
    batch_size:     int          = 128,
    device:         str          = None,
    dtype:          torch.dtype  = torch.float32,
    exclusion_zone: int          = -1,
) -> torch.Tensor:
    """
    Version 3 — out-of-core, prefetched tile streaming.

    Query rows and column tiles are each served by a prefetcher so disk
    reads overlap GPU compute. X is read T/B times in total (see module
    docstring); raise B to reduce passes on slow disks.

    Peak VRAM : O((L+B)*T + B*D + r_chunk*D)
    Peak RAM  : O((B + r_chunk)*D)   [two double-buffers, pinned]
    """
    if exclusion_zone == -1:
        exclusion_zone = traj_length
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    D        = reader.D
    L        = traj_length
    B        = batch_size
    N        = T - L + 1
    buf_size = L + B
    n_pass   = -(-T // B)   # ceil(T/B) full-dataset DISK passes

    print(
        f"[TRAKNN-OOC] T={T} D={D} L={L} B={B}  ({n_pass} disk passes — raise B to reduce)\n"
        f"[TRAKNN-OOC] VRAM buf+rows={2*buf_size*T*4/1e6:.0f} MB "
        f"staging={B*N*4/1e6:.0f} MB Xi(2x)={2*B*D*4/1e6:.0f} MB "
        f"Xj(3x)={3*r_chunk*D*4/1e6:.0f} MB | "
        f"RAM pinned={(2*B + 3*r_chunk)*D*4/1e6:.0f} MB"
    )

    # Device workspaces
    buf            = torch.empty((buf_size, T), dtype=dtype, device=dev)
    distances_traj = torch.zeros(N,             dtype=dtype, device=dev)
    scores         = torch.empty(N,             dtype=dtype, device=dev)
    staging        = torch.empty((B, N),        dtype=dtype, device=dev)
    cols           = torch.arange(N, device=dev).unsqueeze(0)
    rows_buf       = torch.empty((buf_size, T), dtype=dtype, device=dev)

    # Query-row prefetcher (overlaps row disk-read with the recurrence)
    row_pf = _Prefetcher(reader, B, D, dev, dtype)

    # Column tiles use the three-stage pipeline (disk || H2D || GeMM).
    # Allocate the pipeline ONCE and reuse its buffers for every pass —
    # rebuilding it per batch would re-allocate (depth*r_chunk*D) each call
    # and exhaust VRAM.
    col_starts = list(range(0, T, r_chunk))
    cp = _ColumnPipeline(reader, col_starts, r_chunk, D, dev, dtype)

    # Norms: one streaming pass via the column pipeline
    print("[TRAKNN-OOC] norms ...")
    norms = torch.empty(T, dtype=dtype, device=dev)
    cp.start()
    for j, cs in enumerate(col_starts):
        ce = min(cs + r_chunk, T)
        Xj = cp.next_tile(j)
        norms[cs:ce] = (Xj * Xj).sum(dim=1)
        cp.record_compute(j)
    cp.join()

    # Fill rows [rs, re) of S into the circular buffer.
    # Reuses the single pipeline `cp` (restart over the same buffers).
    def fill(rs, re, Xi_dev):
        re = min(re, T)
        if rs >= T:
            return
        B_local = re - rs
        cp.start()
        rows = _compute_S_batch_pipe(
            Xi_dev[:B_local], norms[rs:re], norms,
            rows_buf, T, r_chunk, cp
        )
        cp.join()
        idx = torch.arange(rs, re, device=dev) % buf_size
        buf.index_copy_(0, idx, rows[:B_local])

    # Warm-up: fill rows 0 .. buf_size-1 (capped so row buf_size does not wrap)
    print("[TRAKNN-OOC] warm-up ...")
    for rs in range(0, buf_size, B):
        re = min(rs + B, buf_size)
        row_pf.push(rs, re)
        Xi_dev = row_pf.pop()
        fill(rs, re, Xi_dev)

    # D_0
    for m in range(L):
        distances_traj += buf[m % buf_size, m: m + N]
    distances_traj.clamp_(min=0.0)
    distances_traj0 = distances_traj.clone()

    # Score i=0
    staging[0] = distances_traj
    staging[0, :exclusion_zone] = float("inf")
    scores[0] = torch.topk(staging[0], k=k, largest=False).values.mean()

    # Prime row prefetch for the first new batch
    next_rs = buf_size
    next_re = min(next_rs + B, T)
    if next_rs < T:
        row_pf.push(next_rs, next_re)

    # Main loop: recurrence overlaps the row prefetch (disk read of next batch)
    for i_start in range(1, N, B):
        i_end    = min(i_start + B, N)
        actual_B = i_end - i_start

        _recurrence(
            buf, distances_traj, distances_traj0, scores,
            staging, cols,
            i_start, i_end, L, N, T, k, exclusion_zone, buf_size
        )

        if next_rs < T:
            # Pop the query rows prefetched during the previous iteration
            Xi_dev = row_pf.pop()
            # Push next row batch before GeMM so its disk read overlaps GeMM
            nn_rs = next_re + L - 1
            nn_re = min(nn_rs + actual_B, T)
            if nn_rs < T:
                row_pf.push(nn_rs, nn_re)
            fill(next_rs, next_re, Xi_dev)
            next_rs, next_re = nn_rs, nn_re

    return scores.cpu()

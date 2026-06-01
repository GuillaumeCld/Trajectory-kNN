"""
Diagnostic for V3 all-zero scores. Run on your machine:
    python diag_v3.py
Pinpoints which stage produces zeros: reader, norms, GeMM, buffer, or recurrence.
"""
import numpy as np
import torch
from traknn_streaming import _open_reader, _Prefetcher, _ColumnPipeline, \
    _compute_S_batch_pipe, rarity_scoring_ooc, rarity_scoring_buffer, _load_netcdf

DATA = "Data/era5_msl_daily_eu.nc"
VAR  = "msl"
L, k, B, rc = 7, 10, 256, 512
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("="*60)
print("STAGE 1: reader returns non-zero data?")
reader, T, H, W = _open_reader(DATA, VAR)
D = reader.D
tile = reader.read(0, 5)                 # (5, D) numpy
print(f"  reader.read(0,5): shape={tile.shape} dtype={tile.dtype}")
print(f"  min={tile.min():.3e} max={tile.max():.3e} mean={tile.mean():.3e}")
print(f"  all zero? {np.all(tile==0)}")
if np.all(tile == 0):
    print("  >>> READER returns zeros. The .traknn.f32 file is bad.")
    print("  >>> Check: does the file exist and have real data?")
    import os
    p = f"{DATA}.{VAR}.traknn.f32"
    print(f"  file: {p}  exists={os.path.exists(p)} size={os.path.getsize(p) if os.path.exists(p) else 0}")
    print(f"  done sentinel: {os.path.exists(p+'.done')}")
    raise SystemExit

print("="*60)
print("STAGE 2: norms non-zero?")
norms = torch.empty(T, dtype=torch.float32, device=dev)
cp = _ColumnPipeline(reader, list(range(0, T, rc)), rc, D, dev, dtype=torch.float32)
cp.start()
for j, cs in enumerate(range(0, T, rc)):
    ce = min(cs + rc, T)
    Xj = cp.next_tile(j)
    if j == 0:
        print(f"  tile0 on device: min={Xj.min().item():.3e} max={Xj.max().item():.3e} allzero={bool((Xj==0).all())}")
    norms[cs:ce] = (Xj * Xj).sum(dim=1)
    cp.record_compute(j)
cp.join()
print(f"  norms: min={norms.min().item():.3e} max={norms.max().item():.3e} allzero={bool((norms==0).all())}")

print("="*60)
print("STAGE 3: one GeMM batch of S rows non-zero?")
rows_buf = torch.empty((L + B, T), dtype=torch.float32, device=dev)
row_pf = _Prefetcher(reader, B, D, dev, torch.float32)
row_pf.push(0, B)
Xi = row_pf.pop()
print(f"  Xi query rows: min={Xi.min().item():.3e} max={Xi.max().item():.3e} allzero={bool((Xi==0).all())}")
cp.start()
rows = _compute_S_batch_pipe(Xi, norms[0:B], norms, rows_buf, T, rc, cp)
cp.join()
print(f"  S rows: min={rows.min().item():.3e} max={rows.max().item():.3e} allzero={bool((rows==0).all())}")
print(f"  diagonal S[0,0] (should be 0): {rows[0,0].item():.3e}")
print(f"  off-diag S[0,1] (should be >0): {rows[0,1].item():.3e}")

print("="*60)
print("STAGE 4: full V3 vs V2 on first 5 scores")
s3 = rarity_scoring_ooc(reader, T=T, traj_length=L, k=k, r_chunk=rc, batch_size=B,
                        device=str(dev), exclusion_zone=L)
data = _load_netcdf(DATA, VAR)
s2 = rarity_scoring_buffer(data, traj_length=L, k=k, r_chunk=rc, batch_size=B,
                           device=str(dev), exclusion_zone=L)
print(f"  V3 scores[:5]: {s3[:5].tolist()}")
print(f"  V2 scores[:5]: {s2[:5].tolist()}")
print(f"  max|V2-V3| = {(s2-s3).abs().max().item():.3e}")

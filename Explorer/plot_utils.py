"""
Cartopy-based map plotting helpers.

matplotlib.use("Agg") is set here so it is guaranteed to run before any
pyplot import, regardless of module import order.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np


def make_cartopy_fig(field, lat, lon, title, cbar_label, vmax=None, figsize=(7, 4)):
    """Return a matplotlib Figure with a cartopy PlateCarree map.

    The colorbar is placed in a dedicated row below the map so it does not
    overlap or compress the geographic axes.
    """
    if vmax is None:
        vmax = float(np.nanpercentile(np.abs(field), 98))
    vmax = max(vmax, 1e-6)

    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 1, height_ratios=[18, 1], hspace=0.45)
    ax = fig.add_subplot(gs[0], projection=proj)

    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    mesh = ax.pcolormesh(
        lon, lat, field,
        cmap="RdBu_r", norm=norm,
        transform=proj, shading="auto",
    )
    ax.add_feature(cfeature.COASTLINE, linewidth=0.7, edgecolor="black")
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor="grey")
    ax.set_extent(
        [float(lon.min()), float(lon.max()),
         float(lat.min()), float(lat.max())],
        crs=proj,
    )
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="grey",
                      alpha=0.5, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    ax.set_title(title, fontsize=10, pad=6)

    cax = fig.add_subplot(gs[1])
    cbar = fig.colorbar(mesh, cax=cax, orientation="horizontal")
    cbar.set_label(cbar_label, fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    return fig

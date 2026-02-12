import h5py
import numpy as np
import xarray as xr
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
# Read the HDF5 file
with h5py.File('case_studies/results/z500/z500_trajlen5_k30_distances.mds.h5', 'r') as f:
    points_cloud = f['points'][:]
    eigen_values = f['eigenvalues'][:]

ds = xr.open_dataset("Data/z500_anom_daily_eu.nc")
times = ds["time"].values 
month_vector = np.array([t.astype('datetime64[M]').astype(int) % 12 + 1 for t in times])[: points_cloud.shape[0]]

# Plot the points cloud (assuming 2D or 3D)
levels = np.arange(0, 13, 1) # 12 months
if points_cloud.shape[1] >= 3:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(points_cloud[:, 0], points_cloud[:, 1], points_cloud[:, 2], 
                         alpha=0.6, c=month_vector, cmap='viridis')
    plt.colorbar(scatter, label='Month')
    ax.set_xlabel('MDS Dimension 1')
    ax.set_ylabel('MDS Dimension 2')
    ax.set_zlabel('MDS Dimension 3')
    ax.set_title('Points Cloud')
    plt.show()

# save the points cloud to a CSV file x, y, z, month
df = pd.DataFrame(points_cloud[:, :3], columns=['MDS1', 'MDS2', 'MDS3'])
df['Month'] = month_vector
df.to_csv('case_studies/results/z500/z500.csv', index=False)

eigen_values_normalized = eigen_values / np.sum(eigen_values)
cumsum = np.cumsum(eigen_values_normalized)

eigen_values_normalized = eigen_values_normalized[:50]
cumsum = cumsum[:50]

fig, ax1 = plt.subplots(figsize=(8, 5))

# Explained variance (log scale)
ax1.plot(eigen_values_normalized, color='tab:blue')
ax1.set_xlabel('Eigenvalue Index')
ax1.set_ylabel('Explained Variance')
ax1.set_yscale('log')
ax1.tick_params(axis='y')

# Cumulative variance
ax2 = ax1.twinx()
ax2.plot(cumsum, color='tab:orange')
ax2.set_ylabel('Cumulative Variance')
ax2.set_ylim(0, 1.05)
ax2.tick_params(axis='y')

plt.tight_layout()
plt.show()

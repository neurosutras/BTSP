__author__ = 'Aaron D. Milstein'
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic
from numpy import genfromtxt


# replace this file path with the path to your csv file
# the file should have 3 columns: min_t, initial_ramp, delta_ramp
# min_t is the timebase in seconds (delay to plateau onset) for each spatial bin
# append data from all cells and all bins to these three columns
# the number of rows should be # of cells * # of bins

data_file_path = 'data/20220808_BTSP_CA1_data.csv'
imported_data = genfromtxt(data_file_path, delimiter=',')[1:]
flat_min_t, flat_initial_ramp, flat_delta_ramp = imported_data.T


tmax = 5.  # s
res = 100  # number of temporal bins

points = np.array([flat_min_t, flat_initial_ramp]).transpose()
data = np.array(flat_delta_ramp)

kernel = RationalQuadratic(1., length_scale_bounds=(1e-10, 10.))
gp = GaussianProcessRegressor(kernel=kernel, alpha=3., n_restarts_optimizer=20, normalize_y=True)
start_time = time.time()
print('Starting Gaussian Process Regression with %i samples' % len(data))
gp.fit(points, data)
print('Gaussian Process Regression took %.1f s' % (time.time() - start_time))

current_time = time.time()
t_range = np.linspace(-tmax, tmax, res)
initial_ramp_range = np.linspace(np.min(flat_initial_ramp), np.max(flat_initial_ramp), res)
t_grid, initial_ramp_grid = np.meshgrid(t_range, initial_ramp_range)
interp_points = np.vstack((t_grid.flatten(), initial_ramp_grid.flatten())).T

interp_data = gp.predict(interp_points).reshape(-1, res)
print('Gaussian Process Interpolation took %.1f s' % (time.time() - current_time))

fig, this_axis = plt.subplots(1)
interp_cmap = 'bwr'
ymax = np.max(flat_initial_ramp)
ymin = min(0., np.min(flat_initial_ramp))
vmax = max(abs(np.max(flat_delta_ramp)), abs(np.min(flat_delta_ramp)))

cax = this_axis.pcolormesh(t_grid, initial_ramp_grid, interp_data, cmap=interp_cmap, vmin=-vmax,
                           vmax=vmax, zorder=0, edgecolors='face', rasterized=True, shading='auto')
this_axis.set_ylabel('Initial Vm ramp\namplitude (mV)')
this_axis.set_xlabel('Time from plateau (s)')
this_axis.set_ylim(0., ymax)
this_axis.set_xlim(-tmax, tmax)
this_axis.set_xticks(np.arange(-4., 5., 2.))
cbar = plt.colorbar(cax, ax=this_axis)
cbar.set_label(r'$\Delta$Vm (mV)', rotation=270., labelpad=15.)

fig.tight_layout()
fig.show()

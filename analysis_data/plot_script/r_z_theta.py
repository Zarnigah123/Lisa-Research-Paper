"""Created on Feb 20 15:21:55 2023."""

import numpy as np
from scipy import stats
from myavi import mlab

from utilities import H5Files

h5_files = H5Files('./../')

bhbh_df = h5_files.get_bhbh_file
nsns_df = h5_files.get_bhns_file
bhns_df = h5_files.get_bhns_file
nsbh_df = h5_files.get_nsbh_file

r = bhbh_df['R']
z = bhbh_df['z']
theta = bhbh_df['theta']

rzt = np.vstack([r, z, theta])
kde = stats.gaussian_kde(rzt)
density = kde(rzt)

figure = mlab.figure('DensityPlot')
# pts = mlab.points3d(x, y, z, density, scale_mode='none', scale_factor=0.07)
# mlab.axes()
# mlab.show()

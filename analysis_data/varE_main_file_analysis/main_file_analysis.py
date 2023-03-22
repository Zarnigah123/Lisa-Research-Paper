"""Created on Mar 04 00:43:43 2023."""

import h5py as h5
import numpy as np

h5out = h5.File('/media/astrophysicsandpython/A6EA5CEFEA5CBD6D/ms_project/h5out.h5')

bse_comm_env = h5out['BSE_Common_Envelopes.csv']
bse_dcos = h5out['BSE_Double_Compact_Objects.csv']
bse_pulsars = h5out['BSE_Pulsar_Evolution.csv']
bse_rlof = h5out['BSE_RLOF.csv']
bse_sn = h5out['BSE_Supernovae.csv']
bse_slog = h5out['BSE_Switch_Log.csv']
bse_sys_par = h5out['BSE_System_Parameters.csv']

# get the value of whether a DCO will merge within hubble time
merged = bse_dcos['Merges_Hubble_Time'][()]

# single out the ones that do merge
merge_index = np.where(merged == 1)[0]

# get the parameters
seeds_dco = bse_dcos['SEED'][()][merge_index]

# get the system seeds,
seeds_sys = bse_sys_par['SEED']

# get the indices of matching seeds
seeds_sys_ind = np.in1d(seeds_sys, seeds_dco)

# get the parameters
m1_zams = bse_sys_par['Mass@ZAMS(1)'][()][seeds_sys_ind]
m2_zams = bse_sys_par['Mass@ZAMS(2)'][()][seeds_sys_ind]
a_zams = bse_sys_par['SemiMajorAxis@ZAMS'][()][seeds_sys_ind]
e_zams = bse_sys_par['Eccentricity@ZAMS'][()][seeds_sys_ind]
z_zams = bse_sys_par['Metallicity@ZAMS(1)'][()][seeds_sys_ind]

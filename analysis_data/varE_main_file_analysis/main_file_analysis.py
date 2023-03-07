"""Created on Mar 04 00:43:43 2023."""

import h5py as h5

file_path = '/media/astrophysicsandpython/A6EA5CEFEA5CBD6D/ms_project/h5out.h5'

h5out = h5.File(file_path)

bse_comm_env = h5out['BSE_Common_Envelopes.csv']
bse_dcos = h5out['BSE_Double_Compact_Objects.csv']
bse_pulsars = h5out['BSE_Pulsar_Evolution.csv']
bse_rlof = h5out['BSE_RLOF.csv']
bse_sn = h5out['BSE_Supernovae.csv']
bse_slog = h5out['BSE_Switch_Log.csv']
bse_sys_par = h5out['BSE_System_Parameters.csv']

# print(bse_dcos['Coalescence_Time'][...].squeeze())

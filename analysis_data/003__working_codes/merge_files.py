"""Created on Apr 15 18:24:52 2023."""

from os import listdir as lst_dir

import h5py as h5
import pandas as pd

from backend_codes import functions as fnc

variable_eccentricity_files = [f for f in lst_dir(fnc.ANALYSIS_DATA_PATH) if
                               f.endswith('.h5') and not f.endswith('0e.h5')]

zero_eccentricity_files = [f for f in lst_dir(fnc.ANALYSIS_DATA_PATH) if f.endswith('0e.h5')]

variable_eccentricity_dfs = [pd.DataFrame(h5.File(i)['simulation'][()]) for i in variable_eccentricity_files]
zero_eccentricity_dfs = [pd.DataFrame(h5.File(i)['simulations'][()]) for i in zero_eccentricity_files]

for i, v in enumerate(variable_eccentricity_files):
    variable_eccentricity_dfs[i]['dco_type'] = v.split('.h5')[0]

for i, v in enumerate(zero_eccentricity_files):
    zero_eccentricity_dfs[i]['dco_type'] = v.split('.h5')[0]

variable_eccentricity_join = pd.concat(variable_eccentricity_dfs)
zero_eccentricity_joint = pd.concat(zero_eccentricity_dfs)

variable_eccentricity_join.to_csv(f'{fnc.ANALYSIS_DATA_PATH}/combined_dcos.csv', sep=',', index=False)
zero_eccentricity_joint.to_csv(f'{fnc.ANALYSIS_DATA_PATH}/combined_dcos0e.csv', sep=',', index=False)

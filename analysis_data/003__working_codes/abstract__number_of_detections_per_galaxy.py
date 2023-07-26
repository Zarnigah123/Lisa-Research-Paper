"""Created on Tue Jul 11 19:43:39 2023."""

import pandas as pd

from backend_codes import functions as fnc

df = pd.read_csv(f'{fnc.ANALYSIS_DATA_PATH}/combined_dcos.csv')
df0e = pd.read_csv(f'{fnc.ANALYSIS_DATA_PATH}/combined_dcos0e.csv')

df = df.sort_values(by='seed')
df0e = df0e.sort_values(by='seed')

df.reset_index(drop=True, inplace=True)
df0e.reset_index(drop=True, inplace=True)

n_galaxy_ind = fnc.get_detection_array(data_frame=df)

n_galaxy_bhbh = fnc.get_detection_array(data_frame=df, dco_type='BHBH')
n_galaxy_nsns = fnc.get_detection_array(data_frame=df, dco_type='NSNS')
n_galaxy_bhns = fnc.get_detection_array(data_frame=df, dco_type='BHNS')
n_galaxy_nsbh = fnc.get_detection_array(data_frame=df, dco_type='NSBH')

[print(fnc.get_minimum_maximum_values(i, j))
 for i, j in zip([n_galaxy_ind, n_galaxy_bhbh, n_galaxy_nsns, n_galaxy_bhns, n_galaxy_nsbh],
                 ['all', 'bhbh', 'nsns', 'bhns', 'nsbh'])]

print('')

n_galaxy_ind0e = fnc.get_detection_array(data_frame=df0e)

n_galaxy_bhbh0e = fnc.get_detection_array(data_frame=df0e, dco_type='BHBH0e')
n_galaxy_nsns0e = fnc.get_detection_array(data_frame=df0e, dco_type='NSNS0e')
n_galaxy_bhns0e = fnc.get_detection_array(data_frame=df0e, dco_type='BHNS0e')
n_galaxy_nsbh0e = fnc.get_detection_array(data_frame=df0e, dco_type='NSBH0e')

[print(fnc.get_minimum_maximum_values(i, j))
 for i, j in zip([n_galaxy_ind0e, n_galaxy_bhbh0e, n_galaxy_nsns0e, n_galaxy_bhns0e, n_galaxy_nsbh0e],
                 ['all', 'bhbh', 'nsns', 'bhns', 'nsbh'])]

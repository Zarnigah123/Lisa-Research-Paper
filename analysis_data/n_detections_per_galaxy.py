"""Created on Tue Jul 11 19:43:39 2023."""

import numpy as np
import pandas as pd

df = pd.read_csv('./combined_dcos.csv')
df0e = pd.read_csv('./combined_dcos0e.csv')

df = df.sort_values(by='seed')
df.reset_index(drop=True, inplace=True)

df0e = df0e.sort_values(by='seed')
df0e.reset_index(drop=True, inplace=True)


def get_n_detection_array(data_frame, dco_type='all'):
    if dco_type in ['BHBH', 'NSNS', 'NSBH', 'BHNS', 'BHBH0e', 'NSNS0e', 'NSBH0e', 'BHNS0e']:
        df_ = data_frame[data_frame.dco_type == dco_type]
        df_.reset_index(inplace=True, drop=True)
    else:
        df_ = data_frame

    return np.array([len(np.unique(df_[df_.galaxy_number == i].seed))
                     for i in range(1, 101)])


def get_min_max(data_frame, df_type):
    return f'{df_type} :: min: {data_frame.min()}, max: {data_frame.max()}'


n_galaxy_ind = get_n_detection_array(data_frame=df)

n_galaxy_bhbh = get_n_detection_array(data_frame=df, dco_type='BHBH')
n_galaxy_nsns = get_n_detection_array(data_frame=df, dco_type='NSNS')
n_galaxy_bhns = get_n_detection_array(data_frame=df, dco_type='BHNS')
n_galaxy_nsbh = get_n_detection_array(data_frame=df, dco_type='NSBH')

[print(get_min_max(i, j))
 for i, j in zip([n_galaxy_ind, n_galaxy_bhbh, n_galaxy_nsns, n_galaxy_bhns, n_galaxy_nsbh],
                 ['all', 'bhbh', 'nsns', 'bhns', 'nsbh'])]

print('')

n_galaxy_ind = get_n_detection_array(data_frame=df0e)

n_galaxy_bhbh = get_n_detection_array(data_frame=df0e, dco_type='BHBH0e')
n_galaxy_nsns = get_n_detection_array(data_frame=df0e, dco_type='NSNS0e')
n_galaxy_bhns = get_n_detection_array(data_frame=df0e, dco_type='BHNS0e')
n_galaxy_nsbh = get_n_detection_array(data_frame=df0e, dco_type='NSBH0e')

[print(get_min_max(i, j))
 for i, j in zip([n_galaxy_ind, n_galaxy_bhbh, n_galaxy_nsns, n_galaxy_bhns, n_galaxy_nsbh],
                 ['all', 'bhbh', 'nsns', 'bhns', 'nsbh'])]

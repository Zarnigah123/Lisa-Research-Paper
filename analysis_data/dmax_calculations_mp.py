"""
Created on Fri Dec 31 11:23:31 2021

@author: syedalimohsinbukhari
"""

import numpy as np
import pandas as pd

from dmax_calculations__no_plot import GetDMax

data_ = pd.read_csv('./combined_dcos.csv')

bhbh_df = data_[data_.dco_type == 'BHBH']
nsns_df = data_[data_.dco_type == 'NSNS']
nsbh_df = data_[data_.dco_type == 'NSBH']
bhns_df = data_[data_.dco_type == 'BHNS']

def df_drop_dup(data_frame):
    temp_ = data_frame.drop_duplicates(subset='seed', ignore_index = True)
    return temp_

bhbh_df, nsns_df, nsbh_df, bhns_df = list(map(df_drop_dup,
                                              [bhbh_df, nsns_df, nsbh_df, bhns_df]))

print('Starting BHBH\n')
d_bhbh = GetDMax(bhbh_df, n_proc=3).get_d_max()
np.save('BHBH_maxdist', d_bhbh)

print('Starting NSNS\n')
d_nsns = GetDMax(nsns_df, n_proc=3).get_d_max()
np.save('NSNS_maxdist', d_nsns)

print('Starting NSBH\n')
d_nsbh = GetDMax(nsbh_df, n_proc=3).get_d_max()
np.save('NSBH_maxdist', d_nsbh)

print('Starting BHNS\n')
d_bhns = GetDMax(bhns_df, n_proc=3).get_d_max()
np.save('BHNS_maxdist', d_bhns)

"""
Created on Fri Dec 31 11:23:31 2021

@author: syedalimohsinbukhari
"""

import time
import numpy as np
import pandas as pd

from dmax_calculations__no_plot import GetDMax

gal = np.reshape(np.linspace(1, 100, 100), (10, 10))

data_ = pd.read_csv('./combined_dcos.csv')

bhbh_df = data_[data_.dco_type == 'BHBH']
nsns_df = data_[data_.dco_type == 'NSNS']
nsbh_df = data_[data_.dco_type == 'NSBH']
bhns_df = data_[data_.dco_type == 'BHNS']

bhbh_df.reset_index(drop=True, inplace=True)
nsns_df.reset_index(drop=True, inplace=True)
nsbh_df.reset_index(drop=True, inplace=True)
bhns_df.reset_index(drop=True, inplace=True)

d_bhbh = GetDMax(bhbh_df, n_proc=3).run(gal)
np.save('BHBH_maxdist', d_bhbh)

d_nsns = GetDMax(nsns_df, n_proc=3).run(gal)
np.save('NSNS_maxdist', d_nsns)

d_nsbh = GetDMax(nsbh_df, n_proc=3).run(gal)
np.save('NSBH_maxdist', d_nsbh)

d_bhns = GetDMax(bhns_df, n_proc=3).run(gal)
np.save('BHNS_maxdist', d_bhns)

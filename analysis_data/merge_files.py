"""Created on Apr 15 18:24:52 2023."""

import os

import h5py as h5
import pandas as pd

h5files = [f for f in os.listdir(os.curdir) if f.endswith('.h5')]

dfs_ = [pd.DataFrame(h5.File(i)['simulation'][()]) for i in h5files]

for i, v in enumerate(h5files):
    dfs_[i]['dco_type'] = v.split('.h5')[0]

joint = pd.concat(dfs_)

joint.to_csv('./combined_dcos.csv',
             sep=',',
             index=False)

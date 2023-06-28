"""Created on Thu Jun 22 14:05:47 2023."""

import numpy as np
import pandas as pd

df = pd.read_csv('./combined_dcos.csv')
df0e = pd.read_csv('./combined_dcos0e.csv')

# get common seeds
common_seeds1 = np.in1d(df['seed'], df0e['seed'])
common_seeds2 = np.in1d(df0e['seed'], df['seed'])

df_ = df[common_seeds1]
df0e_ = df0e[common_seeds2]

df_.reset_index(drop=True, inplace=True)
df0e_.reset_index(drop=True, inplace=True)

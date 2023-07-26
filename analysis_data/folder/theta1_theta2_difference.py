"""Created on Fri Jul 21 18:45:32 2023."""

import numpy as np
import pandas as pd

df = pd.read_csv('./combined_dcos.csv')
df0e = pd.read_csv('./combined_dcos0e.csv')

df = df.sort_values(by='seed')
df0e = df0e.sort_values(by='seed')

df.reset_index(drop=True, inplace=True)
df0e.reset_index(drop=True, inplace=True)

# get common seeds
common_seeds1 = np.in1d(df['seed'], df0e['seed'])
common_seeds2 = np.in1d(df0e['seed'], df['seed'])

df_ = df[common_seeds1]
df0e_ = df0e[common_seeds2]

df_ = df_.sort_values(by='seed')
df0e_ = df0e_.sort_values(by='seed')

df_ = df_.drop_duplicates('seed')
df0e_ = df0e_.drop_duplicates('seed')

df_.reset_index(drop=True, inplace=True)
df0e_.reset_index(drop=True, inplace=True)

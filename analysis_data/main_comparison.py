"""Created on Thu Jun 22 14:05:47 2023."""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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

dco_type_df = pd.DataFrame([df_['seed'], df_['dco_type'], df0e_['dco_type']]).T
dco_type_df.columns = ['SEED', 'df_dco', 'df0e_dco']

dco_type_df['df0e_dco'] = [i.split('0e')[0] for i in dco_type_df.df0e_dco]
dco_type_df['df_T'] = [0 if i == 'BHBH' else 1 if i == 'NSNS' else 2 if i == 'BHNS' else 3
                       for i in dco_type_df.df_dco]
dco_type_df['df0e_T'] = [0 if i == 'BHBH' else 1 if i == 'NSNS' else 2 if i == 'BHNS' else 3
                         for i in dco_type_df.df0e_dco]


def draw_arrows(a, b, axes):
    axes.arrow(a[0], a[1], b[0] - a[0], b[1] - a[1], head_width=1, head_length=0.1,
               length_includes_head=True, color='k', zorder=2)


f, ax = plt.subplots(1, 1, figsize=(20, 4))
ax.grid('on', zorder=-10, ls='--')
[draw_arrows([1 + k, i], [1 + k, j], ax) for i, j, k in
 zip(dco_type_df.df_T, dco_type_df.df0e_T, range(len(dco_type_df.df0e_T)))]
[plt.plot([1 + k, 1 + k], [i, j], 'k.', zorder=5) if i == j else plt.plot(1 + k, i, 'k.', zorder=5)
 for i, j, k in zip(dco_type_df.df_T, dco_type_df.df0e_T, range(len(dco_type_df.df0e_T)))]

ax.set_title(r'Divergence of final DCO type in $\aleph_1$ and $\aleph_2$ data sets')
ax.set_yticklabels(['', 'BHBH', '', 'NSNS', '', 'NSBH', '', 'BHNS'])
plt.tight_layout()

[plt.savefig(f'dco_type_divergence_in_two_datasets.{i}') for i in ['pdf', 'png']]
plt.close()

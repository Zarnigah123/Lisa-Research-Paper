"""Created on Mon Jul 24 23:35:13 2023."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backend_codes import functions as fnc

df = fnc.VARIABLE_ECC_CSV_DF
df0e = fnc.ZERO_ECC_CSV_DF

# get common seeds
common_seeds1 = np.in1d(df['seed'], df0e['seed'])
common_seeds2 = np.in1d(df0e['seed'], df['seed'])

df_ = df[common_seeds1]
df0e_ = df0e[common_seeds2]

df_ = fnc.sort_reset_drop_df(df_=df_, drop=True)
df0e_ = fnc.sort_reset_drop_df(df_=df0e_, drop=True)

dco_type_df = pd.DataFrame([df_['seed'], df_['dco_type'], df0e_['dco_type']]).T
dco_type_df.columns = ['SEED', 'df_dco', 'df0e_dco']

dco_type_df['df0e_dco'] = [i.split('0e')[0] for i in dco_type_df.df0e_dco]
dco_type_df['df_T'] = [0 if i == 'BHBH' else 1 if i == 'NSNS' else 2 if i == 'BHNS' else 3
                       for i in dco_type_df.df_dco]
dco_type_df['df0e_T'] = [0 if i == 'BHBH' else 1 if i == 'NSNS' else 2 if i == 'BHNS' else 3
                         for i in dco_type_df.df0e_dco]

fig, ax = plt.subplots(1, 1, figsize=(20, 4))
ax.grid('on', zorder=-10, ls='--')
[fnc.draw_arrows([1 + k, i], [1 + k, j], ax) for i, j, k in
 zip(dco_type_df.df_T, dco_type_df.df0e_T, range(len(dco_type_df.df0e_T)))]
[plt.plot([1 + k, 1 + k], [i, j], 'k.', zorder=5) if i == j else plt.plot(1 + k, i, 'k.', zorder=5)
 for i, j, k in zip(dco_type_df.df_T, dco_type_df.df0e_T, range(len(dco_type_df.df0e_T)))]

plt.plot(1, 1, 'ro', ms=10)
plt.plot(20, 1, 'o', color='cyan', ms=10)

ax.set_title(r'Divergence of final DCO type in $\Theta_1$ and $\Theta_2$ data sets')
ax.set_yticklabels(['', 'BHBH', '', 'NSNS', '', 'NSBH', '', 'BHNS'])

plt.tight_layout()
fnc.save_figure(plt, 'dco_type_divergence_in_two_datasets')
plt.close()

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

##################################################################################################
def get_dco_type(df_, dco_type):
    df = df_[df_['dco_type'] == dco_type]
    df.reset_index(drop=True, inplace=True)

    return df

labels_ = [r'e$_\mathrm{DCO} \leq 0.1$', r'$0.25 <$ e$_\mathrm{DCO} \leq 0.9$',
           r'$0.9 <$ e$_\mathrm{DCO}$']


def seaborn_plot(df_, axes, title, log_scale=(True, False)):
    p = sns.kdeplot(data=df_, x=df_['f_orb'] * df_['SNR_harmonics'], hue='Eccentricity', log_scale=log_scale, ax=axes, hue_order=labels_, palette=['g', 'y', 'r'], fill=True, common_norm=False, common_grid=False, legend=True)
    axes.set_xlabel('Dominant Frequency [Hz]')
    axes.set_title(title)

def get_ecc_type(df_):
    df = df_.copy()
    df['Eccentricity'] = [labels_[0] if i <= 0.1
                          else labels_[1] if 0.1 < i <= 0.9
                          else labels_[2]
                          for i in df['e_dco']]

    df.reset_index(drop=True, inplace=True)

    return df

def get_ecc_proportion(df, low=0.1, high = 0.9):
    p1 = sum(df['e_dco'] < low)
    p2 = sum(np.logical_and(low < df['e_dco'], df['e_dco'] <= high))
    p3 = sum(df['e_dco'] > high)

    return p1, p2, p3


df2 = get_ecc_type(df_ = df)

df_bhbh = get_dco_type(df2, 'BHBH')
df_nsns = get_dco_type(df2, 'NSNS')
df_nsbh = get_dco_type(df2, 'NSBH')
df_bhns = get_dco_type(df2, 'BHNS')

f, ax = plt.subplots(2, 2, figsize=(12, 8))

bhbh_proportions = get_ecc_proportion(df_bhbh)
nsns_proportions = get_ecc_proportion(df_nsns)
nsbh_proportions = get_ecc_proportion(df_nsbh)
bhns_proportions = get_ecc_proportion(df_bhns)

seaborn_plot(df_bhbh, axes=ax[0][0], title='BHBH')
ax[0][0].set_xlim([1e-7, 1e-1])
seaborn_plot(df_nsns, axes=ax[0][1], title='NSNS')
ax[0][1].set_xlim([1e-7, 1e-1])
seaborn_plot(df_bhns, axes=ax[1][0], title='BHNS')
ax[1][0].set_xlim([1e-7, 1e-1])
seaborn_plot(df_nsbh, axes=ax[1][1], title='NSBH')
ax[1][1].set_xlim([1e-7, 1e-1])


def plot_inset(axes, df):
    axins = axes.inset_axes(bounds=(0.1, 0.1, 0.35, 0.5))
    axins.set_ylabel('Log[N]')
    # axins.set_yticklabels([])

    axins.bar(x=range(0, 3), height=np.log10(df), color=['g', 'y', 'r'], alpha=0.25, ec='k')
    [axins.text(i, v - 0.5, str(f'{10**v:.0f}'),
                horizontalalignment='center', verticalalignment='center')
     for i, v in enumerate(np.log10(df))]
    # taken from https://stackoverflow.com/a/12998531
    axins.tick_params('x', which='both', bottom=False, top=False, labelbottom=False)
    # plt.tight_layout()


plot_inset(ax[0][0], bhbh_proportions)
plot_inset(ax[0][1], nsns_proportions)
plot_inset(ax[1][0], bhns_proportions)
plot_inset(ax[1][1], nsbh_proportions)

plt.tight_layout()
plt.savefig('dco_fdom_ecc_details.pdf')
plt.savefig('dco_fdom_ecc_details.png')
plt.close()

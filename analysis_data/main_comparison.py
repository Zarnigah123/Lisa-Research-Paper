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
    axes.arrow(a[0], a[1], b[0] - a[0], b[1] - a[1], head_width=1, head_length=0.1, length_includes_head=True,
               color='k', zorder=2)


# fig, ax = plt.subplots(1, 1, figsize=(20, 4))
# ax.grid('on', zorder=-10, ls='--')
# [draw_arrows([1 + k, i], [1 + k, j], ax) for i, j, k in
#   zip(dco_type_df.df_T, dco_type_df.df0e_T, range(len(dco_type_df.df0e_T)))]
# [plt.plot([1 + k, 1 + k], [i, j], 'k.', zorder=5) if i == j else plt.plot(1 + k, i, 'k.', zorder=5)
#   for i, j, k in zip(dco_type_df.df_T, dco_type_df.df0e_T, range(len(dco_type_df.df0e_T)))]

# ax.set_title(r'Divergence of final DCO type in $\aleph_1$ and $\aleph_2$ data sets')
# ax.set_yticklabels(['', 'BHBH', '', 'NSNS', '', 'NSBH', '', 'BHNS'])
# plt.tight_layout()

# [plt.savefig(f'dco_type_divergence_in_two_datasets.{i}') for i in ['pdf', 'png']]
# plt.close()


##################################################################################################
def get_dco_type(data_frame, dco_type):
    temp_ = data_frame[data_frame['dco_type'] == dco_type]
    temp_.reset_index(drop=True, inplace=True)

    return temp_


labels_ = [r'e$_\mathrm{DCO} \leq 0.1$', r'$0.1 <$ e$_\mathrm{DCO} \leq 0.9$', r'e$_\mathrm{DCO} > 0.9$']


def max_hist(pd_series, bins, axes):
    bars, bins_ = np.histogram(a = np.log10(pd_series), bins=bins)

    max_bars = np.where(bars == bars.max())[0]

    difference = 0.65*(bins_[1] - bins_[0])

    height_ = 10**(bins_[max_bars] + difference)

    axes.axvline(x=height_, color='k', ls='--', alpha=0.75)


    print(f'{height_[0]:.3E}')


def seaborn_plot(data_frame, axes, title, bins, log_scale=(True, False), get_legend=False):
    legend = True if get_legend else False

    p = sns.histplot(data_frame, x=data_frame['f_orb'], hue='Eccentricity',
                     bins=bins, common_norm=True, common_bins=True, element='step', log_scale=(True, True),
                     hue_order=labels_, palette=['g', 'y', 'r'], ax=axes, legend=legend)
    axes.set_ylabel('Log[Counts]')
    axes.set_xlabel('Orbital Frequency [Hz]')
    axes.set_title(title)

    max_hist(data_frame['f_orb'], bins, axes)

    if legend:
        h = p.legend_.legend_handles
        l = [t.get_text() for t in p.legend_.get_texts()]
        p.legend_.remove()

        return h, l

def get_ecc_type(data_frame):
    temp_ = data_frame.copy()
    temp_['Eccentricity'] = [labels_[0] if i <= 0.1 else labels_[1] if 0.1 < i <= 0.9 else labels_[2]
                             for i in temp_['e_dco']]

    temp_.reset_index(drop=True, inplace=True)

    return temp_


def get_ecc_proportion(data_frame, low=0.1, high=0.9):
    p1 = sum(data_frame['e_dco'] < low)
    p2 = sum(np.logical_and(low < data_frame['e_dco'], data_frame['e_dco'] <= high))
    p3 = sum(data_frame['e_dco'] > high)

    return p1, p2, p3


df2 = get_ecc_type(data_frame=df)

df_bhbh = get_dco_type(df2, 'BHBH')
df_nsns = get_dco_type(df2, 'NSNS')
df_nsbh = get_dco_type(df2, 'NSBH')
df_bhns = get_dco_type(df2, 'BHNS')

fig, ax = plt.subplots(3, 4, figsize=(16, 10))

bhbh_proportions = get_ecc_proportion(df_bhbh)
nsns_proportions = get_ecc_proportion(df_nsns)
nsbh_proportions = get_ecc_proportion(df_nsbh)
bhns_proportions = get_ecc_proportion(df_bhns)

bins = np.histogram_bin_edges(np.log10(df_bhns['f_orb']), bins=32)
bins2 = np.histogram_bin_edges(np.log10(df_bhns['f_orb'] * df_bhns['SNR_harmonics']), bins=32)

h, l = seaborn_plot(df_bhbh, axes=ax[0][0], title='BHBH', bins=bins, get_legend=True)
ax[0][0].set_xlim([1e-7, 1e-2])
seaborn_plot(df_nsns, axes=ax[0][1], title='NSNS', bins=bins)
ax[0][1].set_xlim([1e-7, 1e-2])
seaborn_plot(df_bhns, axes=ax[0][2], title='BHNS', bins=bins)
ax[0][2].set_xlim([1e-7, 1e-2])
seaborn_plot(df_nsbh, axes=ax[0][3], title='NSBH', bins=bins)
ax[0][3].set_xlim([1e-7, 1e-2])

def plot_f_orb(axes, data_frame):
    sns.histplot(data_frame, x=data_frame['f_orb']*data_frame['SNR_harmonics'], hue='Eccentricity',
                     bins=bins2, common_norm=True, common_bins=True, element='step', log_scale=(True, True),
                     hue_order=labels_, palette=['g', 'y', 'r'], ax=axes, legend=False)
    axes.set_xlabel('Dominant Frequency [Hz]')
    axes.set_ylabel('Log[Counts]')

    max_hist(data_frame['f_orb'] * data_frame['SNR_harmonics'], bins2, axes)


def bar_plot(axes, data_frame):
    axes.bar(x=range(0, 3), height=np.log10(data_frame), color=['g', 'y', 'r'], alpha=0.25, ec='k')
    axes.set_ylabel('Log[Counts]')

    [axes.text(i, v - 0.5, str(f'{10 ** v:.0f}'), horizontalalignment='center', verticalalignment='center')
      for i, v in enumerate(np.log10(data_frame))]

    # taken from https://stackoverflow.com/a/12998531
    axes.tick_params('x', which='both', bottom=False, top=False, labelbottom=False)


plot_f_orb(axes=ax[1][0], data_frame=df_bhbh)
plot_f_orb(axes=ax[1][1], data_frame=df_nsns)
plot_f_orb(axes=ax[1][2], data_frame=df_bhns)
plot_f_orb(axes=ax[1][3], data_frame=df_nsbh)

bar_plot(axes=ax[2][0], data_frame=bhbh_proportions)
bar_plot(axes=ax[2][1], data_frame=nsns_proportions)
bar_plot(axes=ax[2][2], data_frame=bhns_proportions)
bar_plot(axes=ax[2][3], data_frame=nsbh_proportions)

[i.set_ylabel('') for j in ax for i in j[1:]]
plt.tight_layout()
plt.figlegend(h, labels_, ncols=3, title='Eccentricity', loc='upper center')
fig.subplots_adjust(top=0.9)

[plt.savefig(f'dco_fdom_ecc_details.{i}') for i in ['pdf', 'png']]
plt.close()

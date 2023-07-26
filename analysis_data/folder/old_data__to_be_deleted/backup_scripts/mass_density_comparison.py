"""Created on Sat Feb 11 14:01:33 2023."""

import h5py as h5
import matplotlib.pyplot as plt
import matplotlib.ticker
import pandas as pd
import seaborn as sns

# matplotlib.rc('font', **{'size': 16})

blackhole_df = pd.DataFrame(h5.File('./../BHBH.h5', 'r')['simulation'][...])

f, ax = plt.subplots(2, 2, figsize=(10, 8))


def plot_data(dataframe, x_value, hue, palette, scale, axes_, x_label, lr='left', y_lims=None):
    sns.kdeplot(data=dataframe,
                x=x_value,
                hue=hue,
                legend=False,
                palette=palette,
                alpha=0.25,
                ax=axes_,
                clip=(min(dataframe[x_value]), max(dataframe[x_value])))

    ax2 = axes_.twinx()
    sns.kdeplot(data=dataframe,
                x=x_value,
                ax=ax2,
                color='k',
                clip=(min(dataframe[x_value]), max(dataframe[x_value])))

    axes_.set_yscale(scale)
    axes_.yaxis.set_major_formatter(matplotlib.ticker.EngFormatter(sep='\N{THIN SPACE}'))

    if y_lims is not None:
        axes_.set_ylim(top=y_lims[0])
        ax2.set_ylim(bottom=0, top=y_lims[1])

    new_yaxis = ax2.get_yticks()

    ax2.set_yticklabels([matplotlib.ticker.EngFormatter(sep='\N{THIN SPACE}').format_eng(i) for i in new_yaxis])
    axes_.set_xlabel(x_label)

    if lr == 'left':
        ax2.set_ylabel('')
    else:
        axes_.set_ylabel('')


plot_data(blackhole_df, 'm1_zams', 'galaxy_number', 'RdYlGn', 'linear', ax[0][0],
          x_label=r'$m_{1; \mathrm{ZAMS}}\ [M_\odot]$', lr='left', y_lims=[330e-6, 31e-3])
plot_data(blackhole_df, 'm2_zams', 'galaxy_number', 'RdYlGn', 'linear', ax[0][1],
          x_label=r'$m_{2; \mathrm{ZAMS}}\ [M_\odot]$', lr='right', y_lims=[330e-6, 31e-3])
plot_data(blackhole_df, 'm1_dco', 'galaxy_number', 'RdYlGn', 'linear', ax[1][0],
          x_label=r'$m_{1; \mathrm{DCO}}\ [M_\odot]$', lr='left', y_lims=[1.5e-3, 135e-3])
plot_data(blackhole_df, 'm2_dco', 'galaxy_number', 'RdYlGn', 'linear', ax[1][1],
          x_label=r'$m_{2; \mathrm{DCO}}\ [M_\odot]$', lr='right', y_lims=[1.5e-3, 135e-3])

plt.tight_layout()
plt.savefig('mass_density_comparison.pdf')
plt.close()

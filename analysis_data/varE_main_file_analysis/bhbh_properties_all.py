# -*- coding: utf-8 -*-

from h5py import File
from matplotlib import pyplot as plt
from numpy import array, log10, mean, round
from pandas import DataFrame


def average_plotting(dco_type):
    evolved_ = File(f'/home/astrophysicsandpython/Dropbox/sirasad/analysis_data/'
                    f'{dco_type}.h5')['simulation'][()]

    evolved_ = DataFrame(evolved_)

    evolved_['a_dco'] = evolved_['a_dco'].apply(lambda value_: log10(value_))
    evolved_['a_zams'] = evolved_['a_zams'].apply(lambda value_: log10(value_))
    evolved_['a_lisa'] = evolved_['a_lisa'].apply(lambda value_: log10(value_))
    evolved_['lookback_time'] = evolved_['lookback_time'].apply(lambda value_: log10(value_))
    evolved_['SNR'] = evolved_['SNR'].apply(lambda value_: log10(value_))
    evolved_['Z'] = evolved_['Z'].apply(lambda value_: log10(value_))

    def __average_plotting(par, axes, color=None, unit=''):
        group_ = evolved_.groupby('galaxy_number')[par]
        mu_ = array(group_.mean().to_list())
        sd_ = array(group_.std().to_list())

        x_val = range(len(mu_))

        axes.plot(x_val, mu_, color=color)
        axes.fill_between(x_val, mu_ + sd_, mu_ - sd_, alpha=0.25, color=color)

        axes.hlines(y=mean(mu_), xmin=0, xmax=100, color='k', ls='-.',
                    label=rf'${round(mean(mu_), 4)}$ {unit}')
        axes.legend(loc='best')

    #    def __set_label(_list):
    #        if isinstance(_list, np.float64):
    #            return fr'$10^{{' + f'{_list:.3f}' + rf'}}$'
    #        else:
    #            return [fr'$10^{{' + f'{i:.2f}' + rf'}}$' for i in _list]
    #
    #    def _set_label(axes):
    #        axes.set_yticklabels(__set_label(axes.get_yticks()))

    f, ax = plt.subplots(2, 4, figsize=(14, 6), sharex='all')

    __average_plotting('m1_dco', ax[0][0], 'r', r'$\mathrm{M}_\odot$')
    ax[0][0].set_ylabel(r'm1$_\mathrm{DCO}\ [\mathrm{M}_\odot]$')

    __average_plotting('m2_dco', ax[0][1], 'g', r'$\mathrm{M}_\odot$')
    ax[0][1].set_ylabel(r'm2$_\mathrm{DCO}\ [\mathrm{M}_\odot]$')

    __average_plotting('a_dco', ax[0][2], 'b', 'AU')
    ax[0][2].set_ylabel(r'$\log_{10}$[a$_\mathrm{DCO}]$ [AU]')

    __average_plotting('e_dco', ax[0][3], 'c', '')
    ax[0][3].set_ylabel(r'e$_\mathrm{DCO}$')

    __average_plotting('Z', ax[1][0], 'gray', '')
    ax[1][0].set_ylabel(r'$\log_{10}$[Z]')

    __average_plotting('t_evol', ax[1][1], 'maroon', 'Myr')
    ax[1][1].set_ylabel(r't$_\mathrm{evol}$ [Myr]')

    __average_plotting('lookback_time', ax[1][2], 'y', 'Gyr')
    ax[1][2].set_ylabel(r'$\log_{10}$[t$_\mathrm{lookback}$] [Gyr]')

    __average_plotting('SNR', ax[1][3], 'gold', '')
    ax[1][3].set_ylabel('SNR')

    for x in ax[:, ][1]:
        x.set_xlabel('Number of galaxies')

    plt.tight_layout()
    plt.savefig(f'{dco_type}_n_galaxy_mean_plot.pdf')
    plt.savefig(f'{dco_type}_n_galaxy_mean_plot.png')
    plt.close()


[average_plotting(i) for i in ['BHBH', 'NSNS', 'BHNS', 'NSBH']]
# [average_plotting(i) for i in ['BHBH']]

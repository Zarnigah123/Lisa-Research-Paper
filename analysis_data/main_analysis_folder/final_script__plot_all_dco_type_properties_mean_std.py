# -*- coding: utf-8 -*-

from h5py import File
from matplotlib import pyplot as plt
from numpy import array, log10, mean, round, std
from pandas import DataFrame


def average_plotting(dco_type):
    evolved_ = File(f'/home/astrophysicsandpython/Dropbox/sirasad/analysis_data/'
                    f'{dco_type}.h5')['simulation'][()]

    evolved_ = DataFrame(evolved_)

    evolved_ = evolved_[['m1_dco', 'm2_dco', 'm_chirp', 'a_dco', 'e_dco', 'Z', 't_evol',
                         'lookback_time', 'SNR', 'SNR_harmonics', 'galaxy_number']]

    evolved_['a_dco'] = evolved_['a_dco'].apply(lambda value_: log10(value_))
    evolved_['lookback_time'] = evolved_['lookback_time'].apply(lambda value_: log10(value_))
    evolved_['SNR'] = evolved_['SNR'].apply(lambda value_: log10(value_))
    evolved_['SNR_harmonics'] = evolved_['SNR_harmonics'].apply(lambda value_: log10(value_))

    def __average_plotting(par, axes, color=None, unit=''):
        group_ = evolved_.groupby('galaxy_number')[par]
        mu_ = array(group_.mean().to_list())
        sd_ = array(group_.std().to_list())

        x_val = range(len(mu_))

        axes.plot(x_val, mu_, color=color)
        axes.fill_between(x_val, mu_ + sd_, mu_ - sd_, alpha=0.25, color=color)

        axes.hlines(y=mean(mu_), xmin=0, xmax=100, color='k', ls='-',
                    label=rf'${round(mean(mu_), 4)}$ {unit}')
        axes.hlines(y=mean(mu_) + std(mu_), xmin=0, xmax=100, color='k', ls='--', alpha=0.75)
        axes.hlines(y=mean(mu_) - std(mu_), xmin=0, xmax=100, color='k', ls='--', alpha=0.75)

        axes.legend(loc='best')

    f, ax = plt.subplots(3, 3, figsize=(10, 8), sharex='all')

    f.suptitle(f'Parameter means for {dco_type} type DCO')

    __average_plotting('m1_dco', ax[0][0], 'r', r'$\mathrm{M}_\odot$')
    ax[0][0].set_ylabel(r'm1$_\mathrm{DCO}\ [\mathrm{M}_\odot]$')

    __average_plotting('m2_dco', ax[0][1], 'g', r'$\mathrm{M}_\odot$')
    ax[0][1].set_ylabel(r'm2$_\mathrm{DCO}\ [\mathrm{M}_\odot]$')

    __average_plotting('m_chirp', ax[0][2], 'b', r'$\mathrm{M}_\odot$')
    ax[0][2].set_ylabel(r'm$_\mathrm{chirp}\ [\mathrm{M}_\odot]$')

    __average_plotting('a_dco', ax[1][0], 'c', 'AU')
    ax[1][0].set_ylabel(r'$\log_{10}$[a$_\mathrm{DCO}]$ [AU]')

    __average_plotting('e_dco', ax[1][1], 'gray', '')
    ax[1][1].set_ylabel(r'e$_\mathrm{DCO}$')

    __average_plotting('Z', ax[1][2], 'maroon', '')
    ax[1][2].set_ylabel('Z')

    __average_plotting('t_evol', ax[2][0], 'y', 'Myr')
    ax[2][0].set_ylabel(r't$_\mathrm{evol}$ [Myr]')

    __average_plotting('lookback_time', ax[2][1], 'gold', 'Gyr')
    ax[2][1].set_ylabel(r'$\log_{10}$[t$_\mathrm{lookback}$] [Gyr]')

    __average_plotting('SNR', ax[2][2], 'orange', '')
    ax[2][2].set_ylabel('SNR')

    # __average_plotting('SNR_harmonics', ax[1][4], 'pink', '')
    # ax[1][4].set_ylabel('SNR Harmonics')

    for x in ax[:, ][1]:
        x.set_xlabel('Number of galaxies')

    plt.tight_layout()
    plt.savefig(f'{dco_type}_n_galaxy_mean_plot.pdf')
    plt.savefig(f'{dco_type}_n_galaxy_mean_plot.png')
    plt.close()

    col_map = {'bhbh': 'red',
               'nsns': 'blue',
               'nsbh': 'green',
               'bhns': 'orange'}

    line_map = {'bhbh': '--',
                'nsns': '-.',
                'nsbh': '-',
                'bhns': ':'}

    f, ax = plt.subplots(1, 1, figsize=(10, 4))
    ev_ = evolved_.groupby('galaxy_number').count()['m1_dco']

    plt.hlines(y=mean(ev_), xmin=0, xmax=100, color='k', ls='-',
               label=rf'${round(mean(ev_), 4)}$ detections')
    plt.hlines(y=mean(ev_) + std(ev_), xmin=0, xmax=100, color='k', ls='--', alpha=0.75)
    plt.hlines(y=mean(ev_) - std(ev_), xmin=0, xmax=100, color='k', ls='--', alpha=0.75)
    plt.legend(loc='best')
    ev_.plot.line(marker='.', ls=line_map[dco_type.lower()], color=col_map[dco_type.lower()],
                  legend=False, zorder=10)

    ax.grid('on')
    plt.xlabel('Number of galaxies')
    plt.ylabel(f'Number of {dco_type} pair detections')
    plt.tight_layout()
    plt.savefig(f'{dco_type}_n_detections.pdf')
    plt.savefig(f'{dco_type}_n_detections.png')
    plt.close()


[average_plotting(i) for i in ['BHBH', 'NSNS', 'BHNS', 'NSBH']]
# [average_plotting(i) for i in ['BHBH']]

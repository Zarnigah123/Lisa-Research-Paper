"""Created on Sat May 27 06:37:42 2023."""

import h5py as h5
import pandas as pd
import numpy as np
import astropy.units as u
import legwork as lw
import matplotlib.pyplot as plt
import seaborn as sns

dco_types = ["BHBH", "NSNS", "BHNS", "NSBH"]

fid_sources = [{}, {}, {}, {}]

for i in range(len(dco_types)):
    with h5.File("./../{}.h5".format(dco_types[i]), "r") as f:
        data = f["simulation"][...].squeeze()

    _temp = pd.DataFrame(data)
    # _temp = _temp.drop_duplicates(['seed'])
    # _temp.reset_index(drop=True, inplace=True)

    fid_sources[i]['m1_dco'] = np.array(_temp.m1_dco)
    fid_sources[i]['m2_dco'] = np.array(_temp.m2_dco)
    fid_sources[i]['dist'] = np.array(_temp.distance)
    fid_sources[i]['e_lisa'] = np.array(_temp.e_lisa)
    fid_sources[i]['a_lisa'] = np.array(_temp.a_lisa)
    fid_sources[i]['galaxy_number'] = np.array(_temp.galaxy_number)
    fid_sources[i]['Z'] = np.array(_temp.Z)
    fid_sources[i]['SNR'] = np.array(_temp.SNR)
    fid_sources[i]['weight'] = np.array(_temp.weight)
    fid_sources[i]['seeds'] = np.array(_temp.seed)
    fid_sources[i]['freq'] = np.array(_temp.f_orb) * u.Hz
    fid_sources[i]['mc_dco'] = np.array(_temp.m_chirp)
    fid_sources[i]['max_snr_harmonic'] = np.array(_temp.SNR_harmonics)

frequency_range = np.logspace(np.log10(3e-5), np.log10(0.1), 1000) * u.Hz

all_f_dom = np.concatenate([i['freq'] * i['max_snr_harmonic'] for i in fid_sources])
all_harmonics = np.concatenate([i['max_snr_harmonic'] for i in fid_sources])
all_snr = np.concatenate([i['SNR'] for i in fid_sources])
all_weight = np.concatenate([i['weight'] for i in fid_sources])
all_ecc = np.concatenate([i['e_lisa'] for i in fid_sources])
all_asd = all_snr * np.sqrt(lw.psd.power_spectral_density(all_f_dom))
all_mc = np.concatenate([i['mc_dco'] for i in fid_sources])
all_mr = np.concatenate([i['m1_dco'] / i['m2_dco'] for i in fid_sources])
all_Z = np.concatenate([i['Z'] for i in fid_sources])
all_seed = np.concatenate([i['seeds'] for i in fid_sources])
all_galaxy = np.concatenate([i['galaxy_number'] for i in fid_sources])

all_f = np.concatenate([i['freq'] for i in fid_sources])

_types = [[dco_types[i]] * len(v['m1_dco']) for i, v in enumerate(fid_sources)]
all_types = np.concatenate([i for i in _types])

s_df = pd.DataFrame([all_f_dom.value, all_asd.to(u.Hz**-0.5).value, all_f.value, all_ecc, all_Z, all_types,
                     all_snr, all_seed, all_galaxy]).T
s_df.columns = ['f_dom', 'ASD', 'f_orb', 'ecc', 'Z', 'types', 'SNR', 'seeds', 'gal_number']

# separate the dataframe
s_df_bhbh = s_df[s_df.types == dco_types[0]]
s_df_nsns = s_df[s_df.types == dco_types[1]]
s_df_bhns = s_df[s_df.types == dco_types[2]]
s_df_nsbh = s_df[s_df.types == dco_types[3]]

freq = np.logspace(np.log10(1e-5), np.log10(1), 1000) * u.Hz

_, ax = lw.visualisation.plot_sensitivity_curve(frequency_range=freq)
ax.grid(True, zorder=-1)
p = sns.scatterplot(data=s_df, x='f_dom', y='ASD', hue='ecc', size='Z', palette='RdYlGn_r', ax=ax, lw=0, ec="k", zorder=2, style='types')
handles, labels = p.get_legend_handles_labels()
labels[0] = 'Eccentricity'
labels[5] = 'Metallicity'
labels[-5] = 'DCO Type'
p.legend_.remove()
ax.legend(handles, labels, loc='best')
ax.set_title('LISA detectable DCOs')
plt.tight_layout()
plt.savefig('all_dco_snr_plotting.pdf')
plt.savefig('all_dco_snr_plotting.png')
plt.close()

fig_, ax = plt.subplots(2, 2, figsize=(12, 10), sharex=True)

x, w = 0.04, 0.1
y, height = 0.3, 0.05

ylim_ = [3.1936014014661533e-21, 2.986243916404203e-13]

for x_ in ax:
    for x__ in x_:
        lw.visualisation.plot_sensitivity_curve(frequency_range=freq, fig=fig_, ax=x__)


def make_scatter_plot(df, dco_type, marker, axes):
    axes.grid(True, zorder=-1)
    p_ = sns.scatterplot(data=df, x='f_dom', y='ASD', hue='ecc', size='Z', marker=marker,
                          palette='RdYlGn_r', ax=axes, lw=0, ec="k", zorder=2)
    cax = axes.inset_axes([x, w, y, height])
    norm = plt.Normalize(df['ecc'].min(), df['ecc'].max())
    sm = plt.cm.ScalarMappable(cmap='RdYlGn_r', norm=norm)
    sm.set_array([])
    axes.figure.colorbar(sm, cax=cax, orientation='horizontal')
    handles_, labels_ = p_.get_legend_handles_labels()
    labels_[5] = 'Metallicity'
    p_.legend_.remove()
    axes.set_xlabel('Dominant Frequency [Hz]')
    axes.legend(handles_[5:-1], labels_[5:-1], loc='best')
    axes.annotate(f'{dco_type}: n={len(df)}', xy=(2 * x + w, (y - y / 2.) + (height / 2)),
                  xycoords="axes fraction", color="black", ha="center", va="center")
    if dco_type != 'BHNS':
        axes.set_ylim(ylim_)


make_scatter_plot(s_df_bhbh, 'BHBH', 'o', ax[0][0])
make_scatter_plot(s_df_nsns, 'NSNS', 'X', ax[0][1])
make_scatter_plot(s_df_bhns, 'BHNS', 'P', ax[1][0])
make_scatter_plot(s_df_nsbh, 'NSBH', 's', ax[1][1])

[i[1].set_ylabel('') for i in ax]
ax[0][0].set_xlabel('')
ax[0][1].set_xlabel('')

plt.tight_layout()
plt.savefig('dco_typewise_snr.pdf')
plt.savefig('dco_typewise_snr.png')
plt.close()

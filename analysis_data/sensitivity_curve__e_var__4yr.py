"""
Created on Thu Jan  6 17:44:10 2022

@author: syedalimohsinbukhari
"""

import statistics as stats

import astropy.units as u
import h5py as h5
import legwork
import legwork.psd as psd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec


def plot_sc_with_grid_lines1(frequency_range, fig, ax, mc_mean, mc_min, ecc, line_scale=1,
                             line_rot=10, loc=-75):
    legwork.visualisation.plot_sensitivity_curve(show=False, frequency_range=frequency_range,
                                                 fig=fig, ax=ax,
                                                 fill=True)

    times = np.logspace(0, -8, 9) * u.Gyr
    athings = np.power(4 * legwork.utils.beta(mc_min * 2**(1 / 5), mc_min * 2**(1 / 5)) * times,
                       1 / 4)
    freqs = legwork.utils.get_f_orb_from_a(a=athings, m_1=mc_min * 2**(1 / 5),
                                           m_2=mc_min * 2**(1 / 5))
    hide_height = np.sqrt(legwork.psd.lisa_psd(freqs))

    for i in range(len(times)):
        ax.plot([freqs[i].value, freqs[i].value], [hide_height[i].value, 1e-13], color="grey",
                lw=line_scale, zorder=0,
                linestyle="dotted")
        ax.annotate(r"$10^{{{0:1.0f}}}$ yr".format(np.log10(times[i].to(u.yr).value)),
                    xy=(freqs[i].value, 7.5e-14),
                    va="top", ha="center", rotation=90, fontsize=10 * line_scale, color="grey",
                    bbox=dict(boxstyle="round", ec="white", fc="white", pad=0.0))

    line_length = 1000
    lines_f_range = np.logspace(-6.5, -1.4, line_length) * u.Hz
    for dist in [0.1, 0.5, 8, 30]:
        dist_line_signal = np.sqrt(4 * u.yr).to(u.Hz**(-1 / 2)) * legwork.strain.h_0_n(
            m_c=np.repeat(mc_mean, line_length), dist=np.repeat(dist, line_length) * u.kpc,
            f_orb=lines_f_range,
            n=2, ecc=np.full(line_length, np.mean(ecc)))[:, 0, 0]
        mask = dist_line_signal > np.sqrt(legwork.psd.lisa_psd(lines_f_range * 2))
        ax.plot(lines_f_range[mask] * 2, dist_line_signal[mask], color="grey", linestyle="dotted",
                zorder=0,
                lw=line_scale)
        ax.annotate("{} kpc".format(dist),
                    xy=(lines_f_range[mask][-20].value,
                        dist_line_signal[mask][-20].value * (0.45 + line_scale / 20)),
                    xycoords="data", color="grey", rotation=line_rot * 1.2, ha="right", va="center",
                    fontsize=10 * line_scale,
                    bbox=dict(boxstyle="round", ec="white", fc="white", pad=0.0))

    ax.annotate(r"$\langle \mathcal{{M}}_c \rangle = {{{0:1.1f}}} \, {{\rm M_{{\odot}}}}$".format(
        mc_mean.value),
        xy=(1e-2, 3e-20), ha="center", va="center", fontsize=12 * line_scale,
        bbox=dict(boxstyle="round", ec="white", fc="white", pad=0.0))

    return fig, ax


def make_sensitivity_curve(f_range, axes):
    [legwork.visualisation.plot_sensitivity_curve(f_range, ax=i, fill=True) for i in axes]


def most_common(lst):
    return max(set(lst), key=lst.count)


dco_types = ["BHBH", "NSNS", "BHNS", "NSBH"]

fid_sources = [{}, {}, {}, {}]

for i in range(len(dco_types)):
    with h5.File("{}.h5".format(dco_types[i]), "r") as f:
        data = f["simulation"][...].squeeze()

    _temp = pd.DataFrame(data)
    _temp = _temp.drop_duplicates(['seed'])
    _temp.reset_index(drop=True, inplace=True)

    _m1, _m2, _e = np.array(_temp.m1_dco), np.array(_temp.m2_dco), np.array(_temp.e_dco)
    _a, _dist = np.array(_temp.a_dco), np.array(_temp.distance)
    fid_sources[i]['m1_dco'] = _m1
    fid_sources[i]['m2_dco'] = _m2
    fid_sources[i]['dist'] = _dist
    fid_sources[i]['e_lisa'] = _e
    fid_sources[i]['a_lisa'] = _a
    fid_sources[i]['galaxy_number'] = np.array(_temp.galaxy_number)
    fid_sources[i]['Z'] = np.array(_temp.Z)
    fid_sources[i]['SNR'] = np.array(_temp.SNR)
    fid_sources[i]['weight'] = np.array(_temp.weight)
    fid_sources[i]['seeds'] = np.array(_temp.seed)
    fid_sources[i]['freq'] = legwork.utils.get_f_orb_from_a(_a*u.AU, _m1*u.Msun, _m2*u.Msun)
    fid_sources[i]['mc_dco'] = legwork.utils.chirp_mass(_m1, _m2)
    sources = legwork.source.Source(m_1=_m1 * u.Msun,
                                    m_2=_m2 * u.Msun,
                                    dist=_dist * u.kpc,
                                    ecc=_e,
                                    a=_a * u.AU, n_proc=5)
    sources.get_snr(verbose=True)
    fid_sources[i]['max_snr_harmonic'] = sources.max_snr_harmonic

frequency_range = np.logspace(np.log10(3e-5), np.log10(1), 1000) * u.Hz

all_f_dom = np.concatenate([i['freq'] * i['max_snr_harmonic'] for i in fid_sources])
all_harmonics = np.concatenate([i['max_snr_harmonic'] for i in fid_sources])
all_snr = np.concatenate([i['SNR'] for i in fid_sources])
all_weight = np.concatenate([i['weight'] for i in fid_sources])
all_ecc = np.concatenate([i['e_lisa'] for i in fid_sources])
all_asd = all_snr * np.sqrt(psd.power_spectral_density(all_f_dom))
all_mc = np.concatenate([i['mc_dco'] for i in fid_sources])
all_mr = np.concatenate([i['m1_dco'] / i['m2_dco'] for i in fid_sources])
all_Z = np.concatenate([i['Z'] for i in fid_sources])

all_f = np.concatenate([i['freq'] for i in fid_sources])

_types = [[dco_types[i]] * len(v['m1_dco']) for i, v in enumerate(fid_sources)]
all_types = np.concatenate([i for i in _types])

f, ax = plt.subplots(1, 2, figsize=(12, 6))
sns.histplot(x=np.log10(all_f_dom.value()), bins=32, ax=ax[0], hue=all_types, multiple='stack')
sns.kdeplot(x=np.log10(all_f_dom), hue=all_types, ax=ax[1], fill=True)
print(stats.mode(all_f_dom))
plt.tight_layout()
plt.savefig('dco_distribution__e_var__4yr.pdf')
plt.close()

s_df = pd.DataFrame([all_f_dom, all_asd.to(u.Hz**-0.5).value, all_ecc, all_Z, all_types]).T
s_df.columns = ['f_dom', 'ASD', 'ecc', 'Z', 'types']

# separate the dataframe
s_df_bhbh = s_df[s_df.types == 'BHBH']
s_df_bhns = s_df[s_df.types == 'BHNS']
s_df_nsns = s_df[s_df.types == 'NSNS']
s_df_nsbh = s_df[s_df.types == 'NSBH']

fig = plt.figure(figsize=(18, 25))  # , constrained_layout=True)

gs = GridSpec(3, 2, figure=fig, height_ratios=(1.5, 1, 1))
all_ax = fig.add_subplot(gs[0, :])
axes1 = [fig.add_subplot(gs[1, i]) for i in range(2)]
axes2 = [fig.add_subplot(gs[2, i]) for i in range(2)]

fig, all_ax = plot_sc_with_grid_lines1(frequency_range, fig=fig, ax=all_ax,
                                       mc_mean=np.mean(all_mc) * u.Msun,
                                       mc_min=np.min(all_mc) * u.Msun, ecc=all_ecc, line_rot=12,
                                       loc=-80)

p = sns.scatterplot(x='f_dom', y='ASD', hue='ecc', data=s_df, ax=all_ax, palette='RdYlGn_r',
                    size='Z', sizes=(50, 200),
                    style='types', markers=['P', 'X', 'o', 's'])
x, w = 0.04, 0.1
y, height = 0.3, 0.05
cax = all_ax.inset_axes([x, w, y, height])
norm = plt.Normalize(s_df['ecc'].min(), s_df['ecc'].max())
sm = plt.cm.ScalarMappable(cmap='RdYlGn_r', norm=norm)
sm.set_array([])
all_ax.figure.colorbar(sm, cax=cax, orientation='horizontal')
handles, labels = p.get_legend_handles_labels()
p.legend_.remove()
all_ax.legend(handles[5:], labels[5:], ncol=2, loc=(0.8, 0.75))
all_ax.annotate('All DCOs', xy=(2 * x + w, (y - y / 2.) + (height / 2)), xycoords="axes fraction",
                color="black",
                ha="center", va="bottom")

_, _ = plot_sc_with_grid_lines1(frequency_range, fig=fig, ax=axes1[0],
                                mc_mean=np.mean(fid_sources[0]['mc_dco']) * u.Msun,
                                mc_min=np.min(fid_sources[0]['mc_dco']) * u.Msun,
                                ecc=fid_sources[0]['e_lisa'],
                                line_rot=12, loc=-80)

p = sns.scatterplot(x='f_dom', y='ASD', hue='ecc', data=s_df_bhbh, ax=axes1[0], palette='RdYlGn_r',
                    size='Z',
                    sizes=(50, 200), style=['BHBH'] * len(fid_sources[0]['m1_dco']), markers='P')
cax = axes1[0].inset_axes([x, w, y, height])
norm = plt.Normalize(s_df_bhbh['ecc'].min(), s_df_bhbh['ecc'].max())
sm = plt.cm.ScalarMappable(cmap='RdYlGn_r', norm=norm)
sm.set_array([])
axes1[0].figure.colorbar(sm, cax=cax, orientation='horizontal')
handles, labels = p.get_legend_handles_labels()
p.legend_.remove()
axes1[0].legend(handles[5:-1], labels[5:-1], loc=(0.8, 0.75))
axes1[0].annotate('BHBH', xy=(2 * x + w, (y - y / 2.) + (height / 2)), xycoords="axes fraction",
                  color="black",
                  ha="center", va="bottom")

_, _ = plot_sc_with_grid_lines1(frequency_range, fig=fig, ax=axes1[1],
                                mc_mean=np.mean(fid_sources[1]['mc_dco']) * u.Msun,
                                mc_min=np.min(fid_sources[1]['mc_dco']) * u.Msun,
                                ecc=fid_sources[1]['e_lisa'],
                                line_rot=12, loc=-80)

p = sns.scatterplot(x='f_dom', y='ASD', hue='ecc', data=s_df_nsns, ax=axes1[1], palette='RdYlGn_r',
                    size='Z',
                    sizes=(50, 200), style=['NSNS'] * len(fid_sources[1]['m1_dco']), markers='X')
cax = axes1[1].inset_axes([x, w, y, height])
norm = plt.Normalize(s_df_nsns['ecc'].min(), s_df_nsns['ecc'].max())
sm = plt.cm.ScalarMappable(cmap='RdYlGn_r', norm=norm)
sm.set_array([])
axes1[1].figure.colorbar(sm, cax=cax, orientation='horizontal')
handles, labels = p.get_legend_handles_labels()
p.legend_.remove()
axes1[1].legend(handles[5:-1], labels[5:-1], loc=(0.8, 0.75))
axes1[1].annotate('NSNS', xy=(2 * x + w, (y - y / 2.) + (height / 2)), xycoords="axes fraction",
                  color="black",
                  ha="center", va="bottom")

_, _ = plot_sc_with_grid_lines1(frequency_range, fig=fig, ax=axes2[0],
                                mc_mean=np.mean(fid_sources[2]['mc_dco']) * u.Msun,
                                mc_min=np.min(fid_sources[2]['mc_dco']) * u.Msun,
                                ecc=fid_sources[2]['e_lisa'],
                                line_rot=12, loc=-80)

p = sns.scatterplot(x='f_dom', y='ASD', hue='ecc', data=s_df_bhns, ax=axes2[0], palette='RdYlGn_r',
                    size='Z',
                    sizes=(50, 200), style=['BHNS'] * len(fid_sources[2]['m1_dco']), markers='o')
cax = axes2[0].inset_axes([x, w, y, height])
norm = plt.Normalize(s_df_bhns['ecc'].min(), s_df_bhns['ecc'].max())
sm = plt.cm.ScalarMappable(cmap='RdYlGn_r', norm=norm)
sm.set_array([])
axes2[0].figure.colorbar(sm, cax=cax, orientation='horizontal')
handles, labels = p.get_legend_handles_labels()
p.legend_.remove()
t = ['Z']
t.extend([str(round(float(i), 4)) for i in labels[2:-1] if i != 'Z'])
axes2[0].legend(handles[5:-1], t, loc=(0.8, 0.75))
axes2[0].annotate('BHNS', xy=(2 * x + w, (y - y / 2.) + (height / 2)), xycoords="axes fraction",
                  color="black",
                  ha="center", va="bottom")

_, _ = plot_sc_with_grid_lines1(frequency_range, fig=fig, ax=axes2[1],
                                mc_mean=np.mean(fid_sources[3]['mc_dco']) * u.Msun,
                                mc_min=np.min(fid_sources[3]['mc_dco']) * u.Msun,
                                ecc=fid_sources[3]['e_lisa'],
                                line_rot=12, loc=-80)

p = sns.scatterplot(x='f_dom', y='ASD', hue='ecc', data=s_df_nsbh, ax=axes2[1], palette='RdYlGn_r',
                    size='Z',
                    sizes=(50, 200), style=['NSBH'] * len(fid_sources[3]['m1_dco']), markers='s')
cax = axes2[1].inset_axes([x, w, y, height])
norm = plt.Normalize(s_df_nsbh['ecc'].min(), s_df_nsbh['ecc'].max())
sm = plt.cm.ScalarMappable(cmap='RdYlGn_r', norm=norm)
sm.set_array([])
axes2[1].figure.colorbar(sm, cax=cax, orientation='horizontal')
handles, labels = p.get_legend_handles_labels()
p.legend_.remove()
axes2[1].legend(handles[5:-1], labels[5:-1], loc=(0.8, 0.75))
axes2[1].annotate('NSBH', xy=(2 * x + w, (y - y / 2.) + (height / 2)), xycoords="axes fraction",
                  color="black",
                  ha="center", va="bottom")

all_ax.set_ylim(top=1e-13)
[i.set_ylim(top=1e-13) for i in axes1]
[i.set_ylim(top=1e-13) for i in axes2]

plt.tight_layout()
plt.savefig('fiducial__e_var__4yr.pdf')
plt.close()

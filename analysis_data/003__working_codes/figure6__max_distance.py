"""Created on Sun Jul 16 00:43:40 2023."""

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from legwork.visualisation import plot_sensitivity_curve

from backend_codes import functions as fnc

f_orb_sensitivity = np.logspace(np.log10(1e-5), np.log10(1), 1000)
f_orb = np.logspace(np.log10(1e-5), np.log10(1), 100)

dco_labels = fnc.DCO_LABELS


def get_max_f_orb(data_frame):
    return f_orb[np.where(data_frame == max(data_frame))[0]][0]


bhbh_m, bhbh_d = fnc.get_max_distance_mask(f'{fnc.ANALYSIS_DATA_PATH}/{dco_labels[0]}_max_dist.npy')
nsns_m, nsns_d = fnc.get_max_distance_mask(f'{fnc.ANALYSIS_DATA_PATH}/{dco_labels[1]}_max_dist.npy')
bhns_m, bhns_d = fnc.get_max_distance_mask(f'{fnc.ANALYSIS_DATA_PATH}/{dco_labels[2]}_max_dist.npy')
nsbh_m, nsbh_d = fnc.get_max_distance_mask(f'{fnc.ANALYSIS_DATA_PATH}/{dco_labels[3]}_max_dist.npy')

all_mean = np.mean([bhbh_d, nsns_d, nsbh_d, bhns_d], axis=0)

f, ax2 = plt.subplots(1, 1, figsize=(8, 6))
plt.grid('on', alpha=0.25, zorder=-1)

plot_sensitivity_curve(frequency_range=f_orb_sensitivity * u.Hz, ax=ax2, fig=f)

ax = ax2.twinx()

ax.plot(f_orb[bhbh_m], bhbh_d[bhbh_m], alpha=0.5, ls=':', label=dco_labels[0])
ax.plot(f_orb[nsns_m], nsns_d[nsns_m], alpha=0.5, ls=':', label=dco_labels[1])
ax.plot(f_orb[nsbh_m], nsbh_d[nsbh_m], alpha=0.5, ls=':', label=dco_labels[2])
ax.plot(f_orb[bhns_m], bhns_d[bhns_m], alpha=0.5, ls=':', label=dco_labels[3])

ax.plot(f_orb[:-1], all_mean[:-1], 'k-', label=r'Mean d$_{max}$')

dco_labels.append('ALL')

[print(f'{j} = {get_max_f_orb(i):.3E}')
 for i, j in zip([bhbh_d, nsns_d, nsbh_d, bhns_d, all_mean], dco_labels)]

[print(f'{j} = {(max(i) * u.kpc).to("Mpc").value:.03} Mpc')
 for i, j in zip([bhbh_d, nsns_d, bhns_d, nsbh_d, all_mean], dco_labels)]

ax.set_xscale('log')
plt.yscale('log')
ax.legend(loc='best')
ax2.set_xlabel('Orbital Frequency [Hz]')
ax.set_ylabel('Distance [kpc]')

plt.tight_layout()
fnc.save_figure(pyplot_object=plt, fig_name='d_max')
plt.close()

"""Created on Sun Jul 16 00:43:40 2023."""

import numpy as np
import matplotlib.pyplot as plt
from legwork.visualisation import plot_sensitivity_curve
import astropy.units as u

f_orb_sensitivity = np.logspace(np.log10(1e-5), np.log10(1), 1000)
f_orb = np.logspace(np.log10(1e-5), np.log10(1), 100)


def get_max_distance_mask(file_name):
    file_ = np.load(file_name, allow_pickle=True)

    mean_distance = np.mean(np.concatenate(file_).ravel()[::2], axis=0)

    return mean_distance != mean_distance.min(), mean_distance

def get_max_f_orb(data_frame):
    return f_orb[np.where(data_frame == max(data_frame))[0]][0]

bhbh_m, bhbh_d = get_max_distance_mask('./BHBH_maxdist.npy')
nsns_m, nsns_d = get_max_distance_mask('./NSNS_maxdist.npy')
nsbh_m, nsbh_d = get_max_distance_mask('./NSBH_maxdist.npy')
bhns_m, bhns_d = get_max_distance_mask('./BHNS_maxdist.npy')

all_mean = np.mean([bhbh_d, nsns_d, nsbh_d, bhns_d], axis=0)

f, ax2 = plt.subplots(1, 1, figsize=(8, 6))
plt.grid('on', alpha=0.25, zorder=-1)

plot_sensitivity_curve(frequency_range=f_orb_sensitivity*u.Hz, ax=ax2, fig=f)

ax = ax2.twinx()

ax.plot(f_orb[bhbh_m], bhbh_d[bhbh_m], alpha=0.5, ls=':', label='BHBH')
ax.plot(f_orb[nsns_m], nsns_d[nsns_m], alpha=0.5, ls=':', label='NSNS')
ax.plot(f_orb[nsbh_m], nsbh_d[nsbh_m], alpha=0.5, ls=':', label='NSBH')
ax.plot(f_orb[bhns_m], bhns_d[bhns_m], alpha=0.5, ls=':', label='BHNS')

ax.plot(f_orb[:-1], all_mean[:-1], 'k-', label=r'Mean d$_{max}$')

[print(f'{j} = {get_max_f_orb(i):.3E}')
 for i, j in zip([bhbh_d, nsns_d, nsbh_d, bhns_d, all_mean], ['BHBH', 'NSNS', 'NSBH', 'BHNS', 'ALL'])]

ax.set_xscale('log'); plt.yscale('log')
ax.legend(loc='best')
ax2.set_xlabel('Dominant Frequency [Hz]')
ax.set_ylabel('Distance [kpc]')

plt.tight_layout()

[plt.savefig(f'd_max.{i}') for i in ['pdf', 'png']]
plt.close()

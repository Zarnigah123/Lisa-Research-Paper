"""Created on Wed Jul 26 13:45:34 2023"""

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from legwork.visualisation import plot_sensitivity_curve

from backend_codes import functions as fnc


def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


dco_labels = fnc.DCO_LABELS
dco_labels0e = fnc.DCO_LABELS0E

f_orb_sensitivity = np.logspace(np.log10(1e-5), np.log10(1), 1000)
f_orb = np.logspace(np.log10(1e-5), np.log10(1), 100)

npy_files = [f'{fnc.ANALYSIS_DATA_PATH}/{i}_max_dist.npy' for i in dco_labels]
npy_files0e = [f'{fnc.ANALYSIS_DATA_PATH}/{i}_max_dist.npy' for i in dco_labels0e]

bhbh_m, bhbh_d = fnc.get_max_distance_mask(npy_files[0])
nsns_m, nsns_d = fnc.get_max_distance_mask(npy_files[1])
bhns_m, bhns_d = fnc.get_max_distance_mask(npy_files[2])
nsbh_m, nsbh_d = fnc.get_max_distance_mask(npy_files[3])

bhbh_m0e, bhbh_d0e = fnc.get_max_distance_mask(npy_files0e[0])
nsns_m0e, nsns_d0e = fnc.get_max_distance_mask(npy_files0e[1])
bhns_m0e, bhns_d0e = fnc.get_max_distance_mask(npy_files0e[2])
nsbh_m0e, nsbh_d0e = fnc.get_max_distance_mask(npy_files0e[3])

npy_mean = np.mean([bhbh_d, nsns_d, bhns_d, nsbh_d], axis=0)
npy_mean0e = np.mean([bhbh_d0e, nsns_d0e, bhns_d0e, nsbh_d0e], axis=0)

difference = npy_mean0e / npy_mean

f, ax = plt.subplots(1, 1, figsize=(10, 6))

ax2 = ax.twinx()
ax3 = ax.twinx()

ax3.spines['right'].set_position(('axes', 1.1))
ax3.spines['right'].set_visible(True)

ax.grid('on', alpha=0.25, zorder=-1)

plot_sensitivity_curve(frequency_range=f_orb_sensitivity * u.Hz, fig=f, ax=ax, alpha=0.25)
ax.set_xlabel('Orbital Frequency [Hz]')

p2, = ax2.plot(f_orb[:-1], npy_mean[:-1], 'k-', label=r'Mean d$_{max}\ \Theta_1$')
ax2.plot(f_orb[:-1], npy_mean0e[:-1], 'k-.', label=r'Mean d$_{max}\ \Theta_2$')
ax2.set_yscale('log')
ax2.set_ylabel('Distance [kpc]')
ax2.legend(loc='upper left')

p3, = ax3.plot(f_orb[:-1], difference[:-1], 'r-', label='Max distance ratio')
ax3.plot(f_orb[:-1], (bhbh_d0e / bhbh_d)[:-1], 'b--')
ax3.plot(f_orb[:-1], (nsns_d0e / nsns_d)[:-1], 'c--')
ax3.set_ylabel(r'$\frac{d_\mathrm{max}\ \Theta_2}{d_\mathrm{max}\ \Theta_1}$', fontsize=16)
ax3.legend(loc='upper right')

ax.yaxis.label.set_color("#18068b")
ax2.yaxis.label.set_color(p2.get_color())
ax3.yaxis.label.set_color(p3.get_color())

ax.tick_params(axis='y', colors='#18068b')
ax2.tick_params(axis='y', colors=p2.get_color())
ax3.tick_params(axis='y', colors=p3.get_color())

plt.tight_layout()
# fnc.save_figure(pyplot_object=plt, fig_name='d_max_difference')
# plt.close()

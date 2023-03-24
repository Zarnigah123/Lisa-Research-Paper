"""Created on Feb 09 23:42:10 2023."""

import h5py as h5
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utilities import get_mean_std

matplotlib.rc('font', **{'size': 14})

blackhole_df = pd.DataFrame(h5.File('./BHBH.h5', 'r')['simulation'][...])

m1z_m, m1z_s = get_mean_std(blackhole_df, 'm1_zams')
m2z_m, m2z_s = get_mean_std(blackhole_df, 'm2_zams')

m1d_m, m1d_s = get_mean_std(blackhole_df, 'm1_dco')
m2d_m, m2d_s = get_mean_std(blackhole_df, 'm2_dco')

f, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

x = np.arange(len(m1z_m))
ax[0].plot(x, m1z_m, 'b.-', label=r'$\mu_{m_1; \mathrm{ZAMS}}$')
ax[0].fill_between(x, m1z_m - m1z_s, m1z_m + m1z_s, color='b', alpha=0.1)
ax[0].plot(x, m2z_m, 'g.-', label=r'$\mu_{m_2; \mathrm{ZAMS}}$')
ax[0].fill_between(x, m2z_m - m2z_s, m2z_m + m2z_s, color='g', alpha=0.1)
ax[0].legend(loc='best')

ax[1].plot(x, m1d_m, 'b.-.', label=r'$\mu_{m_1; \mathrm{DCO}}$')
ax[1].fill_between(x, m1d_m - m1d_s, m1d_m + m1d_s, color='b', alpha=0.1)
ax[1].plot(x, m2d_m, 'g.-.', label=r'$\mu_{m_2; \mathrm{DCO}}$')
ax[1].fill_between(x, m2d_m - m2d_s, m2d_m + m2d_s, color='g', alpha=0.1)
ax[1].legend(loc='best')

[i.set_ylabel(r'Mass $[M_\odot]$') for i in ax]

plt.xlabel('Number of galaxies')
plt.tight_layout()

plt.savefig('mass_mean_std_comparison.pdf')

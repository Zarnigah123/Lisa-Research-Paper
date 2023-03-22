"""Created on Feb 09 23:42:10 2023."""

import h5py as h5
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

from utilities import get_mean_std

matplotlib.rc('font', **{'size': 14})

blackhole_df = pd.DataFrame(h5.File('./../BHBH.h5', 'r')['simulation'][...])
bh_ns_df = pd.DataFrame(h5.File('./../BHNS.h5', 'r')['simulation'][...])
ns_ns_df = pd.DataFrame(h5.File('./../NSNS.h5', 'r')['simulation'][...])
ns_bh_df = pd.DataFrame(h5.File('./../NSBH.h5', 'r')['simulation'][...])

az_m__bhbh, az_s__bhbh = get_mean_std(blackhole_df, 'a_zams')
ad_m__bhbh, ad_s__bhbh = get_mean_std(blackhole_df, 'a_dco')
al_m__bhbh, al_s__bhbh = get_mean_std(blackhole_df, 'a_lisa')

az_m__bhns, az_s__bhns = get_mean_std(bh_ns_df, 'a_zams')

az_m__nsns, az_s__nsns = get_mean_std(ns_ns_df, 'a_zams')

az_m__nsbh, az_m__nsbh = get_mean_std(ns_bh_df, 'a_zams')

f, ax = plt.subplots(1, 1, figsize=(10, 6))

x__bhbh = np.arange(len(az_m__bhbh))
x__bhns = np.arange(len(az_m__bhns))
x__nsns = np.arange(len(az_m__nsns))
x__nsbh = np.arange(len(az_m__nsbh))

plt.grid('on')
ax.plot(x__bhbh, az_m__bhbh, 'r*', label=r'BHBH')

# ax[1].plot(x__bhbh, ad_m__bhbh, 'g.-.', label=r'$\mu_{a; \mathrm{DCO}}$')
# ax[1].fill_between(x__bhbh, ad_m__bhbh - ad_s__bhbh, ad_m__bhbh + ad_s__bhbh, color='g', alpha=0.1)

ax.plot(x__bhns, az_m__bhns, 'b', marker='d', label=r'BHNS', ls='')
# ax[1].fill_between(x__bhns, az_m__bhns, 'g.-.', label=r'$\mu_{a; \mathrm{ZAMS}}$')
ax.plot(x__nsns, az_m__nsns, 'c', marker='h', ls='', label=r'NSNS')
ax.plot(x__nsbh, az_m__nsbh, color='maroon', marker='D', ls='', label='NSBH')  # , ls='-')#, ls='-.')
# ax[1].plot(x__bhbh, al_m__bhbh, 'b.-.', label=r'$\mu_{a; \mathrm{LISA}}$')
# ax[1].fill_between(x__bhbh, al_m__bhbh - al_s__bhbh, al_m__bhbh + al_s__bhbh, color='b', alpha=0.1)

# [i.set_ylabel(r'Semimajor axis [AU]') for i in ax]
ax.set_yscale('log')
plt.legend(loc='best')

# [i.legend(loc='best') for i in ax]

plt.xlabel('Number of galaxies')
plt.tight_layout()
plt.show()

# plt.savefig('semimajoraxis_mean_std_comparison.pdf')
# plt.close()

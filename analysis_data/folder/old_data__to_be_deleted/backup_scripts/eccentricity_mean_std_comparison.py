"""Created on Feb 09 23:42:10 2023."""

import h5py as h5
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utilities import get_mean_std

matplotlib.rc('font', **{'size': 14})

blackhole_df = pd.DataFrame(h5.File('./../BHBH.h5', 'r')['simulation'][...])

ez_m, ez_s = get_mean_std(blackhole_df, 'e_zams')
ed_m, ed_s = get_mean_std(blackhole_df, 'e_dco')
el_m, el_s = get_mean_std(blackhole_df, 'e_lisa')

plt.figure(figsize=(10, 6))

x = np.arange(len(ez_m))
plt.plot(x, ez_m, 'r.-.', label=r'$\mu_{e; \mathrm{ZAMS}}$')
plt.fill_between(x, ez_m - ez_s, ez_m + ez_s, color='r', alpha=0.1)

plt.plot(x, ed_m, 'g.-.', label=r'$\mu_{e; \mathrm{DCO}}$')
plt.fill_between(x, ed_m - ed_s, ed_m + ed_s, color='g', alpha=0.1)

plt.plot(x, el_m, 'b.-.', label=r'$\mu_{e; \mathrm{LISA}}$')
plt.fill_between(x, el_m - el_s, el_m + el_s, color='b', alpha=0.1)

plt.ylabel(r'Eccentricity')

plt.legend(loc='best')

plt.xlabel('Number of galaxies')
plt.tight_layout()
plt.show()

plt.savefig('eccentricity_mean_std_comparison.pdf')
plt.close()

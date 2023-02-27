"""Created on Feb 12 13:34:36 2023."""

from matplotlib import pyplot as plt

from utilities import get_mean_std, H5Files, plot_mass

h5_files = H5Files('./../')

bhbh_df = h5_files.get_bhbh_file
nsns_df = h5_files.get_bhns_file
bhns_df = h5_files.get_bhns_file
nsbh_df = h5_files.get_nsbh_file

m1z_m_bhbh, m1z_s_bhbh = get_mean_std(bhbh_df, 'm1_zams')
m2z_m_bhbh, m2z_s_bhbh = get_mean_std(bhbh_df, 'm2_zams')

m1d_m_bhbh, m1d_s_bhbh = get_mean_std(bhbh_df, 'm1_dco')
m2d_m_bhbh, m2d_s_bhbh = get_mean_std(bhbh_df, 'm2_dco')

m1d_m_nsns, m1d_s_nsns = get_mean_std(nsns_df, 'm1_dco')
m2d_m_nsns, m2d_s_nsns = get_mean_std(nsns_df, 'm2_dco')

m1d_m_bhns, m1d_s_bhns = get_mean_std(bhns_df, 'm1_dco')
m2d_m_bhns, m2d_s_bhns = get_mean_std(bhns_df, 'm2_dco')

m1d_m_nsbh, m1d_s_nsbh = get_mean_std(nsbh_df, 'm1_dco')
m2d_m_nsbh, m2d_s_nsbh = get_mean_std(nsbh_df, 'm2_dco')

x = range(1, 101)

f, ax = plt.subplots(4, 1, sharex='all', figsize=(12, 10))

plot_mass(x, [m1d_m_bhbh, m2d_m_bhbh], [m1d_s_bhbh, m2d_s_bhbh], [r'$m_1$ DCO', r'$m_2$ DCO'],
          ['r', 'g'], '-.', ax[0])
ax0 = ax[0].twinx()
plot_mass(x, [m1z_m_bhbh, m2z_m_bhbh], [m1z_s_bhbh, m2z_s_bhbh], [r'$m_1$ ZAMS', r'$m_2$ ZAMS'],
          ['c', 'orange'], ':', ax0)

plot_mass(x, [m1d_m_nsns, m2d_m_nsns], [m1d_s_nsns, m2d_s_nsns], [r'$m_1$ NSNS', r'$m_2$ NSNS'],
          ['r', 'g'], '-.', ax[1])
plot_mass(x, [m1d_m_bhns, m2d_m_bhns], [m1d_s_bhns, m2d_s_bhns], [r'$m_1$ BHNS', r'$m_2$ BHNS'],
          ['r', 'g'], '-.', ax[2])
plot_mass(x, [m1d_m_nsbh, m2d_m_nsbh], [m1d_s_nsbh, m2d_s_nsbh], [r'$m_1$ NSBH', r'$m_2$ NSBH'],
          ['r', 'g'], '-.', ax[3])

plt.xlabel('Number of galaxies')
[i.legend(loc='upper left') for i in ax]
ax0.legend(loc='upper right')
plt.tight_layout()

# plt.savefig('mass_comparison.pdf')
# plt.close()
plt.show()

"""Created on Sat Mar  4 16:00:42 2023."""

import matplotlib.pyplot as plt

import bootstrap as bstrp
from utilities import H5Files

h5_files = H5Files('./../')

bhbh_df = h5_files.get_bhbh_file
nsns_df = h5_files.get_bhns_file
bhns_df = h5_files.get_bhns_file
nsbh_df = h5_files.get_nsbh_file

f, ax = plt.subplots(4, 1, figsize=(8, 12))
bstrp.bootstrapped_kde(bhbh_df['m1_zams'], weights=bhbh_df['weight'], seeds=bhbh_df['seed'],
                       ax=ax[0], color='blue', label='m1-ZAMS')
ax2 = ax[0].twinx()
bstrp.bootstrapped_kde(bhbh_df['m1_dco'], weights=bhbh_df['weight'], seeds=bhbh_df['seed'],
                       ax=ax2, color='red', label='m1_DCO')

bstrp.bootstrapped_kde(bhbh_df['m2_zams'], weights=bhbh_df['weight'], seeds=bhbh_df['seed'],
                       ax=ax[1], color='blue', label='m1-ZAMS')
ax2 = ax[1].twinx()
bstrp.bootstrapped_kde(bhbh_df['m2_dco'], weights=bhbh_df['weight'], seeds=bhbh_df['seed'],
                       ax=ax2, color='red', label='m1_DCO')

bstrp.bootstrapped_kde(bhbh_df['a_zams'], weights=bhbh_df['weight'], seeds=bhbh_df['seed'],
                       ax=ax[2], color='blue', label='m1-ZAMS')
ax2 = ax[2].twinx()
bstrp.bootstrapped_kde(bhbh_df['a_dco'], weights=bhbh_df['weight'], seeds=bhbh_df['seed'],
                       ax=ax2, color='red', label='m1_DCO')

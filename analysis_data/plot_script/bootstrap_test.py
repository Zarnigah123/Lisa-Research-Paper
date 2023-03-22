"""Created on Feb 17 15:03:20 2023."""
import matplotlib.pyplot as plt

import bootstrap as bstrp
from utilities import H5Files

h5_files = H5Files('./../')

bhbh_df = h5_files.get_bhbh_file
nsns_df = h5_files.get_bhns_file
bhns_df = h5_files.get_bhns_file
nsbh_df = h5_files.get_nsbh_file

f, ax = plt.subplots(1, 1, figsize=(8, 6))
bstrp.bootstrapped_kde(bhbh_df['a_zams'], weights=bhbh_df['weight'], seeds=bhbh_df['seed'],
                       ax=ax, bootstraps=500)
bstrp.bootstrapped_kde(bhbh_df['a_dco'], bhbh_df['weight'], bhbh_df['seed'], ax, color='r')
plt.show()

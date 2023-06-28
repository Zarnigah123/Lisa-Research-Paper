"""Created on Sat Mar 25 04:58:20 2023."""

import h5py as h5
import numpy as np

# h5out = h5.File('/media/astrophysicsandpython/DATA_DRIVE0/h5out.h5')
h5out = h5.File('/media/astrophysicsandpython/DATA_DRIVE0/h5out0e.h5')

bse_sys_par = h5out['BSE_System_Parameters.csv']
bse_dcos = h5out['BSE_Double_Compact_Objects.csv']

# get the number of binaries in the file
total_binaries_in_the_file = len(bse_sys_par['SEED'][()])

# get the number of dcos in the file
total_dcos_in_the_file = len(bse_dcos['SEED'][()])

# how many dcos merge in hubble time
merge_mask = bse_dcos['Merges_Hubble_Time'][()] == 1
total_dcos_that_merge_in_H = merge_mask.sum()

# types of binaries that merge within hubble time
stellar_type_1 = bse_dcos['Stellar_Type(1)'][()]
stellar_type_2 = bse_dcos['Stellar_Type(2)'][()]

st1_merge = stellar_type_1[merge_mask]
st2_merge = stellar_type_2[merge_mask]

def dco_mask(st1, st2, merged=True):
    if merged:
        return np.logical_and(st1_merge == st1, st2_merge == st2)
    else:
        return np.logical_and(stellar_type_1 == st1, stellar_type_2 == st2)

# bhbh pairs
bhbh_mask = dco_mask(14, 14)
bhbh_binaries = len(st1_merge[bhbh_mask])

# nsns pairs
nsns_mask = dco_mask(13, 13)
nsns_binaries = len(st1_merge[nsns_mask])

# bhns pairs
bhns_mask = dco_mask(14, 13)
bhns_binaries = len(st1_merge[bhns_mask])

# nsbh pairs
nsbh_mask = dco_mask(13, 14)
nsbh_binaires = len(st1_merge[nsbh_mask])

# total nsbh pairs
total_nsbh_pairs = bhns_binaries + nsbh_binaires

# dco pairs that didn't merge
bhbh_binaries_all = len(stellar_type_1[dco_mask(14, 14, False)])
nsns_binaries_all = len(stellar_type_1[dco_mask(13, 13, False)])
bhns_binaries_all = len(stellar_type_1[dco_mask(14, 13, False)])
nsbh_binaires_all = len(stellar_type_1[dco_mask(13, 14, False)])


def get_percentage(num, den):
    return round((num/den)*100, 2)

"""
Created on Mon Dec 13 16:11:59 2021

@author: syedalimohsinbukhari
"""

import h5py
import numpy as np
import pandas as pd

from utilities import make_h5, separate_binaries


def separate_dco_types(merged_h5_file, merge_within_hubble_time=True, separate_nsbh_binaries=True):
    if separate_nsbh_binaries:
        dco_order = ['NSNS', 'BHBH', 'NSBH', 'BHNS']
    else:
        dco_order = ['NSNS', 'BHBH', 'NSBH']

    data = h5py.File(merged_h5_file, 'r')
    dco_data = data['BSE_Double_Compact_Objects.csv']
    zams_data = data['BSE_System_Parameters.csv']

    dco_keys = np.array(list(dco_data.keys()))
    metallicity_keys = np.array(list(zams_data.keys()))

    dco_dictionary, dco_units = {}, []

    for i in dco_keys:
        _get = dco_data.get(f'{i}')
        dco_dictionary[f'{i}'] = _get[()]
        dco_units.append((_get.attrs['units']).astype(str))

    metallicity_dictionary = {}

    for j in metallicity_keys:
        _get = zams_data.get(f'{j}')
        metallicity_dictionary[f'{j}'] = _get[()]

    dco_df = pd.DataFrame(dco_dictionary)

    if merge_within_hubble_time:
        dco_df = dco_df[dco_df['Merges_Hubble_Time'] == 1]
        dco_df.reset_index(drop=True, inplace=True)

    if separate_nsbh_binaries:
        _merged = [separate_binaries(dco_df, i) for i in [(13, 13), (14, 14), (13, 14), (14, 13)]]
    else:
        _merged = [separate_binaries(dco_df, i) for i in [(13, 13), (14, 14)]]
        _merged.append(separate_binaries(dco_df, [13, 14], combine_nsbh=True))

    metallicity_df = pd.DataFrame(metallicity_dictionary)

    _dco_keep = ['Mass(1)', 'Mass(2)', 'Eccentricity@DCO', 'SemiMajorAxis@DCO', 'Time', 'Coalescence_Time', 'SEED',
                 'Stellar_Type(1)', 'Stellar_Type(2)']

    _metallicity_keep = ['Mass@ZAMS(1)', 'Mass@ZAMS(2)', 'Eccentricity@ZAMS', 'Metallicity@ZAMS(1)',
                         'SemiMajorAxis@ZAMS', 'SEED']

    _keep = [i[_dco_keep] for i in _merged]

    metallicity_df = metallicity_df[_metallicity_keep]

    merged_within_hubble_time = [pd.merge(i, metallicity_df, on='SEED') for i in _keep]

    _new_keys = ['m1_dco', 'm2_dco', 'e_dco', 'a_dco', 't_evolution__Myr', 't_merge__Myr', 'seed', 's1_type', 's2_type', 'm1_zams', 'm2_zams', 'e_zams', 'z_zams', 'a_zams']

    for i in merged_within_hubble_time:
        i.columns = _new_keys

    _new_order = ['m1_zams', 'm2_zams', 'a_zams', 'e_zams', 'z_zams', 'seed', 's1_type', 's2_type', 'm1_dco', 'm2_dco', 'a_dco', 'e_dco',
                  't_evolution__Myr', 't_merge__Myr']

    [make_h5(f'{dco_order[i]}@DCO.h5', dataframe=v, order=_new_order) for i, v in enumerate(merged_within_hubble_time)]


if __name__ == '__main__':
    separate_dco_types('h5out.h5', separate_nsbh_binaries=False)

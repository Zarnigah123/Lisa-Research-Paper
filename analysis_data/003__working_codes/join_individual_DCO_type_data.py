"""Created on Feb 09 19:51:29 2023."""

import glob

import h5py
import pandas as pd

from backend_codes import functions as fnc

keys_ = ['m1_zams', 'm1_dco', 'm2_zams', 'm2_dco', 'm_chirp',
         'a_zams', 'a_dco', 'a_lisa', 'f_orb',
         'e_zams', 'e_dco', 'e_lisa',
         't_evol', 't_merge', 'lookback_time',
         'Z', 'component',
         'SNR', 'SNR_harmonics', 'weight',
         'distance', 'R', 'z', 'theta', 'seed',
         'galaxy_number']


def main(file_name, is_variable=True):
    if is_variable:
        dir_ = f'{fnc.VARIABLE_ECC_PATH}/new{file_name}'
    else:
        dir_ = f'{fnc.ZERO_ECC_PATH}/new{file_name}0e'

    files = glob.glob(f'{dir_}/*.h5')
    files.sort()

    h5pd_list = []
    for i, file_ in enumerate(files):
        df = pd.DataFrame(h5py.File(file_, 'r')['simulation'][...].tolist())
        df['galaxy_name'] = int(i + 1)
        h5pd_list.append(df)

    h5pd = pd.concat(h5pd_list)

    if is_variable:
        fnc.make_h5(f'{fnc.ANALYSIS_DATA_PATH}/{file_name.upper()}.h5', h5pd, keys=keys_)
    else:
        fnc.make_h5(f'{fnc.ANALYSIS_DATA_PATH}/{file_name.upper()}0e.h5', h5pd, keys=keys_)


if __name__ == '__main__':
    dco_types = ['bhbh', 'nsns', 'nsbh', 'bhns']
    ecc_type = [True, False]

    for ecc_ in ecc_type:
        for dco_type in dco_types:
            main(file_name=dco_type, is_variable=ecc_)

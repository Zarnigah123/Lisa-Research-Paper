"""Created on Feb 09 19:51:29 2023."""

import glob

import h5py
import pandas as pd

from utilities import make_h5

keys_ = ['m1_zams', 'm1_dco', 'm2_zams', 'm2_dco', 'm_chirp',
	 'a_zams', 'a_dco', 'a_lisa', 'f_orb',
         'e_zams', 'e_dco', 'e_lisa', 't_evol', 't_merge', 'lookback_time',
         'Z', 'component', 'SNR', 'SNR_harmonics', 'distance', 'R', 'z', 'theta', 'weight', 'seed',
         'galaxy_number']


def main(file_name):
    dir_ = f'./../new{file_name}'
    files = glob.glob(f'{dir_}/*.h5')
    files.sort()

    h5pd_list = []
    for i, file_ in enumerate(files):
        df = pd.DataFrame(h5py.File(file_, 'r')['simulation'][...].tolist())
        df['galaxy_name'] = int(i + 1)
        h5pd_list.append(df)

    h5pd = pd.concat(h5pd_list)

    make_h5(f'{file_name.upper()}.h5', h5pd, keys=keys_)


if __name__ == '__main__':
    main('bhbh')
    main('nsns')
    main('nsbh')
    main('bhns')

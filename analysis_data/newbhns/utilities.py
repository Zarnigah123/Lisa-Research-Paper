"""
Created on Tue Dec 14 00:13:51 2021

@author: syedalimohsinbukhari
"""

import math
import os

import h5py
import numpy as np
import pandas as pd
from astropy.io.misc.hdf5 import write_table_hdf5
from astropy.table import Table


def separate_binaries(df, order, combine_nsbh=False):
    if combine_nsbh:
        _df1 = df[(df['Stellar_Type(1)'] == order[0]) & (df['Stellar_Type(2)'] == order[1])]
        _df2 = df[(df['Stellar_Type(1)'] == order[1]) & (df['Stellar_Type(2)'] == order[0])]
        _df = pd.concat([_df1, _df2], ignore_index=True)
    else:
        _df = df[(df['Stellar_Type(1)'] == order[0]) & (df['Stellar_Type(2)'] == order[1])]

    _df.reset_index(drop=True, inplace=True)

    return _df


def make_h5(file_name, dataframe, order=None, keys=None, folder_name='simulation'):
    if order is None:
        new_dataframe = dataframe
        order = keys
    else:
        new_dataframe = dataframe.reindex(columns=order)  # taken from https://stackoverflow.com/a/47467999/3212945

    _h5 = h5py.File(file_name, 'w')
    write_table_hdf5(Table(data=np.stack(np.array(new_dataframe.T), axis=1), names=order), _h5, folder_name)
    _h5.close()


# def find_neighbours(df, value, column_name):
#     # taken from https://stackoverflow.com/a/53553226/3212945
#     low_ind = df[df[column_name] < value][column_name].astype(float).idxmax()
#     upp_ind = df[df[column_name] > value][column_name].astype(float).idxmin()
#     test1, test2 = abs(value - df.iloc[low_ind][column_name]), abs(value - df.iloc[upp_ind][column_name])
#
#     return [low_ind if test1 < test2 else upp_ind][0]


def find_nearest(array, value):
    # taken from https://stackoverflow.com/a/26026189/3212945
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])):
        return idx - 1
    else:
        return idx


def change_to_float(df):
    _keys = df.keys()
    for i in _keys:
        df[f'{i}'] = df[f'{i}'].astype(float)
    return df


def assign_mw_coordinates(pars):
    h5_file, simulated_mw, gal_number, in_dir, out_dir = pars

    print(f'gal_#_{gal_number}')

    _gal = np.load(simulated_mw, allow_pickle=True)

    _filename = h5_file.split('@')[0] + f'@GAL_{str(gal_number)}.h5'
    df = np.array(h5py.File(h5_file, 'r')['simulation'][:].tolist())

    # sort on metallicity
    sim_mw = _gal[_gal[:, 2].argsort()]
    sim_mw_z, df_z = sim_mw[:, 2], df[:, 4]

    idd = [find_nearest(sim_mw_z, v) for i, v in enumerate(df_z)]

    sim_mw = sim_mw[idd]

    # replacing gal_comp
    sim_mw[:, 0] = sim_mw[:, 0] * 1e3
    sim_mw[:, -1] = np.where(sim_mw[:, -1] == 'bulge', 0, sim_mw[:, -1])
    sim_mw[:, -1] = np.where(sim_mw[:, -1] == 'low_alpha_disc', 1, sim_mw[:, -1])
    sim_mw[:, -1] = np.where(sim_mw[:, -1] == 'high_alpha_disc', 2, sim_mw[:, -1])

    # remove the galaxy's metallicity
    sim_mw = np.delete(sim_mw, 2, 1)

    _join = np.concatenate((df, sim_mw), axis=1)
    keys = ['m1_zams', 'm2_zams', 'a_zams', 'e_zams', 'z_zams', 'seed', 's1_type', 's2_type', 'm1_dco', 'm2_dco', 'a_dco', 'e_dco', 't_evolution__Myr', 't_merge__Myr', 't_lookback__Gyr', 'distance__kpc', 'component']

    _pd = pd.DataFrame(_join)
    _pd.columns = keys
    _pd = change_to_float(_pd)

    try:
        os.mkdir(f'{out_dir}')
    except Exception:
        pass

    os.chdir(out_dir)

    make_h5(_filename, _pd, _pd.keys())

    os.chdir(in_dir)


def chunks(lst, n):
    # taken from https://stackoverflow.com/a/312464/3212945
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

"""
Created on Tue Dec 14 00:13:51 2021

@author: syedalimohsinbukhari
"""

import math
import os
from typing import List, Optional, Union

import astropy.units as u
import h5py as h5
import legwork as lw
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from astropy.io.misc.hdf5 import write_table_hdf5
from astropy.table import Table

try:
    from compas_processing import get_COMPAS_vars, mask_COMPAS_data
    from galaxy import simulate_mw
except ModuleNotFoundError:
    from .compas_processing import get_COMPAS_vars, mask_COMPAS_data
    from .galaxy import simulate_mw

np.random.seed(235)

HOME = os.getenv('HOME')

PRELIMINARY_OUTPUT = 'h5out_to_csv__dcos'
DCO_LABELS = ['BHBH', 'NSNS', 'BHNS', 'NSBH']
DCO_LABELS0E = ['BHBH0e', 'NSNS0e', 'BHNS0e', 'NSBH0e']

# File paths for the merged HDF5 files
VARIABLE_ECCENTRICITY_H5_FILE = '/media/astrophysicsandpython/DATA_DRIVE0/h5out.h5'
ZERO_ECCENTRICITY_H5_FILE = '/media/astrophysicsandpython/DATA_DRIVE0/h5out0e.h5'

ANALYSIS_DATA_PATH = f'{HOME}/Dropbox/sirasad/analysis_data'

VARIABLE_ECC_PATH = f'{ANALYSIS_DATA_PATH}/001__variable_eccentricity_complete_dataset'
ZERO_ECC_PATH = f'{ANALYSIS_DATA_PATH}/002__zero_eccentricity_complete_dataset'

VARIABLE_ECC_CSV = f'{ANALYSIS_DATA_PATH}/combined_dcos.csv'
ZERO_ECC_CSV = f'{ANALYSIS_DATA_PATH}/combined_dcos0e.csv'

ALL_DCOs_VARIABLE = f'{ANALYSIS_DATA_PATH}/{PRELIMINARY_OUTPUT}.csv'
ALL_DCOs_VARIABLE__MERGED = f'{ANALYSIS_DATA_PATH}/{PRELIMINARY_OUTPUT}__merged_only.csv'

ALL_DCOs_ZERO = f'{ANALYSIS_DATA_PATH}/{PRELIMINARY_OUTPUT}0e.csv'
ALL_DCOs_ZERO__MERGED = f'{ANALYSIS_DATA_PATH}/{PRELIMINARY_OUTPUT}__merged_only0e.csv'

IMG_PATH = f'{ANALYSIS_DATA_PATH}/004__images_for_latex'

VARIABLE_ECC_CSV_DF = pd.read_csv(VARIABLE_ECC_CSV)
ZERO_ECC_CSV_DF = pd.read_csv(ZERO_ECC_CSV)


def get_separated_dcos(is_variable_ecc: Optional[bool] = True, drop_duplicates: Optional[bool] = False,
                       data_frame: Optional[pd.DataFrame] = None):
    """
    Get DataFrames of separated DCO types from the provided DataFrame.

    Args:
        is_variable_ecc (bool, optional): Whether to get DataFrames for variable eccentricity (True) or zero
                                          eccentricity (False). Default is True.
        drop_duplicates (bool, optional): Whether to drop duplicate rows based on 'seed' for each separated DataFrame.
                                          Default is False.
        data_frame (pd.DataFrame, optional): Dataframe from which the DCO separation is to be done.

    Returns:
        List[pd.DataFrame]: A list containing DataFrames of separated DCO types.
    """

    dco_types = DCO_LABELS if is_variable_ecc else DCO_LABELS0E

    if data_frame is None:
        data_frame = VARIABLE_ECC_CSV_DF if is_variable_ecc else ZERO_ECC_CSV_DF

    separated_dcos = [data_frame[data_frame.dco_type == dco_type] for dco_type in dco_types]

    if drop_duplicates:
        separated_dcos = [sort_reset_drop_df(df_=df, drop=True, drop_key='seed') for df in separated_dcos]
    else:
        separated_dcos = [sort_reset_drop_df(df_=df) for df in separated_dcos]

    return separated_dcos


def sort_reset_drop_df(df_: pd.DataFrame, sort_key: str = 'seed', drop: bool = False, drop_key: Optional[str] = 'seed'):
    """
    Sort, reset index, and optionally drop duplicate rows based on the specified keys in the DataFrame.

    Args:
        df_ (pd.DataFrame): The DataFrame to be sorted and manipulated.
        sort_key (str, optional): The column to use for sorting the DataFrame. Default is 'seed'.
        drop (bool, optional): Whether to drop duplicate rows based on 'drop_key'. Default is False.
        drop_key (Optional[str], optional): The column to use for identifying duplicates if 'drop' is True.
        Default is 'seed'.

    Returns:
        pd.DataFrame: The sorted DataFrame with a reset index and potential duplicate rows dropped.
    """

    if drop:
        df_ = df_.drop_duplicates(subset=drop_key)

    df_ = df_.sort_values(by=sort_key)
    df_.reset_index(drop=True, inplace=True)

    return df_


VARIABLE_ECC_CSV_DF = sort_reset_drop_df(df_=VARIABLE_ECC_CSV_DF)
ZERO_ECC_CSV_DF = sort_reset_drop_df(df_=ZERO_ECC_CSV_DF)


def save_figure(pyplot_object, fig_name):
    [pyplot_object.savefig(f'{IMG_PATH}/{fig_name}.{i}') for i in ['pdf', 'png']]


def draw_arrows(a, b, axes):
    axes.arrow(a[0], a[1], b[0] - a[0], b[1] - a[1], head_width=1, head_length=0.1, length_includes_head=True,
               color='k', zorder=2)


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

    _h5 = h5.File(file_name, 'w')
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
    df = np.array(h5.File(h5_file, 'r')['simulation'][:].tolist())

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
    keys = ['m1_zams', 'm2_zams', 'a_zams', 'e_zams', 'z_zams', 'seed', 's1_type', 's2_type', 'm1_dco', 'm2_dco',
            'a_dco', 'e_dco', 't_evolution__Myr', 't_merge__Myr', 't_lookback__Gyr', 'distance__kpc', 'component']

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


def make_detectable_dataset(path_to_h5_file, number_of_galaxy_instances, binary_type, observation_time,
                            milkyway_size=1_000_000, snr_cutoff=7, variable_eccentricity=True):
    with h5.File(path_to_h5_file, "r") as COMPAS_file:
        _h5 = COMPAS_file["BSE_System_Parameters.csv"]
        metallicity_ = np.array(_h5["Metallicity@ZAMS(1)"][()])

        dco_mask = mask_COMPAS_data(COMPAS_file, binary_type, (True, False, False))

        dco_mask = np.where(dco_mask == 1)[0]

        # sys_par = COMPAS_file["BSE_System_Parameters.csv"]
        zams_m_1 = _h5["Mass@ZAMS(1)"][()]
        zams_m_2 = _h5["Mass@ZAMS(2)"][()]
        zams_a = _h5["SemiMajorAxis@ZAMS"][()]
        zams_e = _h5["Eccentricity@ZAMS"][()]
        zams_seeds = _h5["SEED"][()]

        _, compas_Z = get_COMPAS_vars(COMPAS_file, "BSE_Double_Compact_Objects.csv",
                                      ["Mass(1)", "Mass(2)", "SemiMajorAxis@DCO",
                                       "Eccentricity@DCO",
                                       "Time", "SEED"],
                                      mask=dco_mask,
                                      metallicity=metallicity_)

        compas_m_1, compas_m_2, compas_a_dco, compas_e_dco, compas_t_evol, compas_seeds = _

        # add units
        compas_m_1, compas_m_2 = compas_m_1 * u.Msun, compas_m_2 * u.Msun
        compas_a_dco *= u.AU
        compas_t_evol *= u.Myr

    index_1 = np.in1d(zams_seeds, compas_seeds)

    zams_m_1 = zams_m_1[index_1]
    zams_m_2 = zams_m_2[index_1]
    zams_a = zams_a[index_1]
    zams_e = zams_e[index_1]
    zams_seeds = zams_seeds[index_1]

    temp1_ = pd.DataFrame([zams_m_1, zams_m_2, zams_a, zams_e, zams_seeds]).T
    temp1_.columns = ["Mass@ZAMS(1)", "Mass@ZAMS(2)", "SemiMajorAxis@ZAMS", "Eccentricity@ZAMS",
                      "SEEDS@ZAMS"]

    zams_m1, zams_m2, zams_a, zams_e, zams_seeds = temp1_.to_numpy().T

    # work out metallicity bins
    compas_Z_unique = np.unique(compas_Z)
    inner_bins = np.sort(np.array([compas_Z_unique[i] + (compas_Z_unique[i + 1] - compas_Z_unique[i]) / 2
                                   for i in range(len(compas_Z_unique) - 1)]))

    Z_bins = np.concatenate(([compas_Z_unique[0]], inner_bins, [compas_Z_unique[-1]]))

    # create a random number generator
    rng = np.random.default_rng()

    # prep the temporary variable for parameters
    dt = np.dtype(float)

    n_detect_list = np.zeros(number_of_galaxy_instances)

    for milky_way in range(number_of_galaxy_instances):
        print(f"number {milky_way + 1}\n")

        tau, dist, Z_unbinned, pos, component = simulate_mw(milkyway_size, ret_pos=True)

        component[component == "low_alpha_disc"] = 0
        component[component == "high_alpha_disc"] = 1
        component[component == "bulge"] = 2

        R, z, theta = pos

        # work out COMPAS limits (and limit to Z=0.022)
        min_Z_compas = np.min(compas_Z_unique)

        max_Z_compas = np.max(compas_Z_unique[compas_Z_unique <= 0.03])
        # max_Z_compas = 0.03

        # change metallicities above COMPAS limits to between solar and upper
        too_big = Z_unbinned > max_Z_compas

        Z_unbinned[too_big] = 10 ** (np.random.uniform(np.log10(0.01416),
                                                       np.log10(max_Z_compas),
                                                       len(Z_unbinned[too_big])))

        # change metallicities below COMPAS limits to lower limit
        too_small = Z_unbinned < min_Z_compas

        # print(Z_unbinned[too_small])
        Z_unbinned[too_small] = 10 ** (np.random.uniform(np.log10(min_Z_compas),
                                                         np.log10(0.01416),
                                                         len(Z_unbinned[too_small])))

        # sort by metallicity so everything matches up well
        Z_order = np.argsort(Z_unbinned)
        tau, dist, Z_unbinned, R, z, theta, component = tau[Z_order], dist[Z_order], Z_unbinned[
            Z_order], R[Z_order], z[Z_order], theta[Z_order], component[Z_order]

        # bin the metallicities using Floor's bins
        h, _ = np.histogram(Z_unbinned, bins=Z_bins)

        # draw binaries for each metallicity bin, store indices
        binaries = np.zeros(milkyway_size).astype(int)
        indices = np.arange(len(compas_m_1)).astype(int)
        total = 0

        for i, v in enumerate(h):
            if h[i] > 0:
                same_Z = compas_Z == compas_Z_unique[i]
                binaries[total:total + h[i]] = rng.choice(indices[same_Z], h[i], replace=True)
                total += h[i]

        if total != milkyway_size:
            print(compas_Z_unique)
            print(Z_bins)
            print(np.sum(h), h)
            print(min_Z_compas, max_Z_compas)
            exit("PANIC: something funky is happening with the Z bins")

        # mask parameters for binaries
        sys_m1 = zams_m1[binaries]
        sys_m2 = zams_m2[binaries]
        sys_a = zams_a[binaries]
        sys_e = zams_e[binaries]
        sys_seeds = zams_seeds[binaries]

        dco_m1 = compas_m_1[binaries]
        dco_m2 = compas_m_2[binaries]
        dco_a = compas_a_dco[binaries]
        dco_e = compas_e_dco[binaries]
        t_evol = compas_t_evol[binaries]
        Z = compas_Z[binaries]
        seed = compas_seeds[binaries]
        w = np.ones(len(dco_m1))

        print("starting t_merge")

        # work out which binaries are still in the inspiral phase
        e_dco_ = chunks(dco_e, 1000)
        a_dco_ = chunks(dco_a, 1000)
        m1 = chunks(dco_m1, 1000)
        m2 = chunks(dco_m2, 1000)

        merge = [lw.evol.get_t_merge_ecc(ecc_i=i, a_i=j, m_1=k, m_2=l).value for i, j, k, l in
                 zip(e_dco_, a_dco_, m1, m2)]

        print("t_merge ended")

        t_merge = np.array([element for innerList in merge for element in innerList])
        t_merge = t_merge * u.Gyr
        inspiral = t_merge > (tau - t_evol)

        # trim out the merged binaries
        dco_m1, dco_m2 = dco_m1[inspiral], dco_m2[inspiral]
        dco_a, dco_e = dco_a[inspiral], dco_e[inspiral]
        t_evol, t_merge = t_evol[inspiral], t_merge[inspiral]
        tau, dist, Z, component = tau[inspiral], dist[inspiral], Z[inspiral], component[inspiral]
        R, z, theta = R[inspiral], z[inspiral], theta[inspiral]
        w, seed = w[inspiral], seed[inspiral]

        sys_m1, sys_m2 = sys_m1[inspiral], sys_m2[inspiral]
        sys_a, sys_e, sys_seeds = sys_a[inspiral], sys_e[inspiral], sys_seeds[inspiral]

        # evolve binaries to LISA
        e_LISA, a_LISA, f_orb_LISA = lw.evol.evol_ecc(ecc_i=dco_e, a_i=dco_a, m_1=dco_m1,
                                                      m_2=dco_m2,
                                                      t_evol=tau - t_evol,
                                                      output_vars=["ecc", "a", "f_orb"], n_proc=3)
        # we only care about the final state
        e_LISA = e_LISA[:, -1]
        a_LISA = a_LISA[:, -1]
        f_orb_LISA = f_orb_LISA[:, -1]

        sources = lw.source.Source(m_1=dco_m1, m_2=dco_m2, ecc=e_LISA, dist=dist, f_orb=f_orb_LISA,
                                   sc_params={"t_obs": observation_time})
        snr = sources.get_snr(t_obs=4 * u.yr)
        harmonics = sources.max_snr_harmonic

        detectable = snr > snr_cutoff
        n_detect = len(snr[detectable])
        print(f"n_detect={n_detect}")
        print(binary_type, n_detect, "Properties")
        n_detect_list[milky_way] = n_detect

        dtype = [("m1_zams", dt), ("m1_dco", dt), ("m2_zams", dt), ("m2_dco", dt), ("m_chrip", dt),
                 ("a_zams", dt), ("a_dco", dt), ("a_lisa", dt), ("f_orb", dt),
                 ("e_zams", dt), ("e_dco", dt), ("e_lisa", dt),
                 ("t_evol", dt), ("t_merge", dt), ("lookback_time", dt), ("Z", dt),
                 ("component", np.dtype(int)), ("SNR", dt), ("SNR_harmonics", dt),
                 ("weight", np.dtype(int)),
                 ("distance", dt), ("R", dt), ("z", dt), ("theta", dt), ("seed", np.dtype(int))]

        to_file = np.zeros(n_detect, dtype=dtype)

        # store parameters in temporary variable
        to_file["m1_zams"] = sys_m1[detectable]
        to_file["m1_dco"] = dco_m1[detectable]
        to_file["m2_zams"] = sys_m2[detectable]
        to_file["m2_dco"] = dco_m2[detectable]
        to_file["m_chrip"] = lw.utils.chirp_mass(dco_m1[detectable], dco_m2[detectable])
        to_file["a_zams"] = sys_a[detectable]
        to_file["a_dco"] = dco_a[detectable]
        to_file["a_lisa"] = a_LISA[detectable]
        to_file["f_orb"] = f_orb_LISA[detectable]
        to_file["e_zams"] = sys_e[detectable]
        to_file["e_dco"] = dco_e[detectable]
        to_file["e_lisa"] = e_LISA[detectable]
        to_file["t_evol"] = t_evol[detectable].to('Myr')
        to_file["t_merge"] = t_merge[detectable].to('Myr')
        to_file["lookback_time"] = tau[detectable].to('Myr')
        to_file["Z"] = Z[detectable]
        to_file["component"] = component[detectable]
        to_file["SNR"] = snr[detectable]
        to_file["SNR_harmonics"] = harmonics[detectable]
        to_file["weight"] = w[detectable]
        to_file["distance"] = dist[detectable]
        to_file["R"] = R[detectable]
        to_file["z"] = z[detectable]
        to_file["theta"] = theta[detectable]

        to_file["seed"] = sys_seeds[detectable]

        df = pd.DataFrame(to_file)
        df = df.loc[~(df == 0).all(axis=1)]

        cwd = os.getcwd()

        if variable_eccentricity:
            dir_ = f'{VARIABLE_ECC_PATH}/new{binary_type.lower()}'
        else:
            dir_ = f'{ZERO_ECC_PATH}/new{binary_type.lower()}0e'

        os.makedirs(f'{dir_}', exist_ok=True)

        os.chdir(f'{dir_}')

        make_h5(f"{binary_type}__GalaxyNum_{milky_way + 1:03}.h5", df,
                keys=["m1_zams", "m1_dco", "m2_zams", "m2_dco", "m_chirp",
                      "a_zams", "a_dco", "a_lisa",
                      "f_orb",
                      "e_zams", "e_dco", "e_lisa",
                      "t_evol", "t_merge", "lookback_time",
                      "Z", "component",
                      "SNR", "SNR_harmonics",
                      "weight", "distance", "R", "z", "theta",
                      "seed"])
        os.chdir(cwd)


def separate_dco_types(merged_h5_file: str, merge_within_hubble_time: bool = True, separate_nsbh_binaries: bool = True):
    """
    Separate and save double compact objects (DCOs) into separate HDF5 files based on their types.

    Parameters:
        merged_h5_file (str):
            The path to the merged HDF5 file containing DCO and metallicity data.
        merge_within_hubble_time (bool, optional):
            If True, only include DCOs that merge within the Hubble time (t < 13.8 Gyr). Default is True.
        separate_nsbh_binaries (bool, optional):
            If True, separate NSBH binaries; otherwise, combine NSBH and BHNS binaries. Default is True.

    Returns:
        None

    Note:
        The function will create separate HDF5 files for each DCO type ('BHBH', 'NSNS', 'NSBH', and 'BHNS').
    """

    dco_order = ['BHBH', 'NSNS', 'NSBH', 'BHNS'] if separate_nsbh_binaries else ['BHBH', 'NSNS', 'NSBH']

    with h5.File(merged_h5_file, 'r') as data:
        dco_data = data['BSE_Double_Compact_Objects.csv']
        zams_data = data['BSE_System_Parameters.csv']

        dco_df = pd.DataFrame(dco_data.values()).T
        dco_df.columns = dco_df.keys()

        metallicity_df = pd.DataFrame(zams_data.values()).T
        metallicity_df.columns = zams_data.keys()

        if merge_within_hubble_time:
            dco_df = dco_df[dco_df['Merges_Hubble_Time'] == 1]
            dco_df.reset_index(drop=True, inplace=True)

        if separate_nsbh_binaries:
            _merged = [separate_binaries(dco_df, i) for i in [(14, 14), (13, 13), (13, 14), (14, 13)]]
        else:
            _merged = [separate_binaries(dco_df, i) for i in [(14, 14), (13, 13)]]
            _merged.append(separate_binaries(dco_df, [13, 14], combine_nsbh=True))

        _dco_keep = ['Mass(1)', 'Mass(2)', 'Eccentricity@DCO', 'SemiMajorAxis@DCO', 'Time', 'Coalescence_Time', 'SEED',
                     'Stellar_Type(1)', 'Stellar_Type(2)']

        _metallicity_keep = ['Mass@ZAMS(1)', 'Mass@ZAMS(2)', 'Eccentricity@ZAMS', 'Metallicity@ZAMS(1)',
                             'SemiMajorAxis@ZAMS', 'SEED']

        _keep = [i[_dco_keep] for i in _merged]

        metallicity_df = metallicity_df[_metallicity_keep]

        merged_within_hubble_time = [pd.merge(i, metallicity_df, on='SEED') for i in _keep]

        _new_keys = ['m1_dco', 'm2_dco', 'e_dco', 'a_dco', 't_evolution__Myr', 't_merge__Myr', 'seed', 's1_type',
                     's2_type',
                     'm1_zams', 'm2_zams', 'e_zams', 'z_zams', 'a_zams']

        for i in merged_within_hubble_time:
            i.columns = _new_keys

        _new_order = ['m1_zams', 'm2_zams', 'a_zams', 'e_zams', 'z_zams', 'seed', 's1_type', 's2_type', 'm1_dco',
                      'm2_dco',
                      'a_dco', 'e_dco',
                      't_evolution__Myr', 't_merge__Myr']

        [make_h5(f'{dco_order[i]}@DCO.h5', dataframe=v, order=_new_order) for i, v in
         enumerate(merged_within_hubble_time)]


def get_detection_array(data_frame: pd.DataFrame, dco_type: Union[str, None] = 'all'):
    """
    Get a detection array for a given DataFrame.

    Args:
        data_frame (pd.DataFrame): The DataFrame containing the data for which to create the detection array.
        dco_type (Union[str, None], optional): The DCO type to consider or 'all' to include all types.
                                               Default is 'all'.

    Returns:
        np.ndarray: A detection array containing the number of unique seeds for each galaxy number (1 to 100).
    """

    if dco_type in ['BHBH', 'NSNS', 'NSBH', 'BHNS', 'BHBH0e', 'NSNS0e', 'NSBH0e', 'BHNS0e']:
        df_ = data_frame[data_frame.dco_type == dco_type]
        df_.reset_index(inplace=True, drop=True)
    else:
        df_ = data_frame

    detection_array = np.array([len(np.unique(df_[df_.galaxy_number == i].seed)) for i in range(1, 101)])

    return detection_array


def get_minimum_maximum_values(data_frame: pd.DataFrame, df_type: Union[str, int]):
    """
    Get the minimum and maximum values of a given DataFrame.

    Args:
        data_frame (pd.DataFrame): The DataFrame for which to calculate the minimum and maximum values.
        df_type (Union[str, int]): A label or identifier for the DataFrame.

    Returns:
        str: A formatted string containing the DataFrame type and its minimum and maximum values.
    """

    min_value, max_value = data_frame.min(), data_frame.max()

    return f'{df_type} :: min: {min_value}, max: {max_value}'


def get_merged_dco_masks(is_variable=True):
    dco_file = h5.File(VARIABLE_ECC_CSV) if is_variable else h5.File(ZERO_ECC_CSV)
    dco_types = ['BHBH', 'NSNS', 'BHNS', 'NSBH'] if is_variable else ['BHBH0e', 'NSNS0e', 'BHNS0e', 'NSBH0e']

    masks = [dco_file['dco_type'] == i for i in dco_types]
    data = [dco_file[i] for i in masks]

    data = [i.reset_index(drop=True, inplace=True) for i in data]

    return masks, data


def get_histogram_peak_value(pd_series: pd.Series, bins: int, axes: plt.Axes):
    """
    Get the peak value of a histogram plotted on the given axes.

    Args:
        pd_series (pd.Series): The pandas Series for which to calculate the histogram.
        bins (int): The number of bins to use for the histogram.
        axes (plt.Axes): The matplotlib axes on which the histogram is plotted.

    Returns:
        None: The function does not return anything but adds an 'axvline' representing the peak value to the axes.
    """

    bars, bins_ = np.histogram(a=np.log10(pd_series), bins=bins)
    max_bars = np.where(bars == bars.max())[0]
    difference = 0.65 * (bins_[1] - bins_[0])
    height_ = 10 ** (bins_[max_bars] + difference)
    axes.axvline(x=height_[0], color='k', ls='--', alpha=0.75)
    print(f'{height_[0]:.3E}')


def seaborn_plot(data_frame: pd.DataFrame, axes: plt.Axes, title: str, bins: int, hue_labels: List[str],
                 get_legend: bool = False):
    """
    Create a seaborn histogram plot on the given axes.

    Args:
        data_frame (pd.DataFrame): The DataFrame containing the data to plot.
        axes (plt.Axes): The matplotlib axes on which the histogram is plotted.
        title (str): The title of the plot.
        bins (int): The number of bins to use for the histogram.
        hue_labels (List[str]): A list of labels for the 'Eccentricity' hue.
        get_legend (bool, optional): Whether to return the handles and labels for the legend. Default is False.

    Returns:
        Optional[Tuple[List, List]]: If get_legend is True, returns a tuple containing handles and labels for the legend.
    """

    legend = True if get_legend else False

    p = sns.histplot(data_frame, x=data_frame['f_orb'], hue='Eccentricity', bins=bins, common_norm=True,
                     common_bins=True, element='step', log_scale=(True, True), hue_order=hue_labels,
                     palette=['g', 'y', 'r'], ax=axes, legend=legend)
    axes.set_ylabel('Log[Counts]')
    axes.set_xlabel('Orbital Frequency [Hz]')
    axes.set_title(title)

    get_histogram_peak_value(data_frame['f_orb'], bins, axes)

    if legend:
        handles = p.legend_.legend_handles
        labels = [texts.get_text() for texts in p.legend_.get_texts()]
        p.legend_.remove()

        return handles, labels


def plot_f_orb(axes: plt.Axes, data_frame: pd.DataFrame, bins: int, hue_order: List[str]):
    """
    Create a seaborn histogram plot for 'dominant frequency' vs the DCOs 'ASD' on the given axes.

    Args:
        axes (plt.Axes): The matplotlib axes on which the histogram is plotted.
        data_frame (pd.DataFrame): The DataFrame containing the data.
        bins (int): The number of bins to use for the histogram.
        hue_order (List[str]): A list of labels for the 'Eccentricity' hue.

    Returns:
        None: The function does not return anything but creates the plot directly on the given axes.
    """

    sns.histplot(data_frame, x=data_frame['f_orb'] * data_frame['SNR_harmonics'], hue='Eccentricity',
                 bins=bins, common_norm=True, common_bins=True, element='step', log_scale=(True, True),
                 hue_order=hue_order, palette=['g', 'y', 'r'], ax=axes, legend=False)
    axes.set_xlabel('Dominant Frequency [Hz]')
    axes.set_ylabel('Log[Counts]')

    get_histogram_peak_value(data_frame['f_orb'] * data_frame['SNR_harmonics'], bins, axes)


def bar_plot(axes: plt.Axes, data_frame: pd.Series):
    """
    Create a bar plot with log-scaled counts from the pandas Series on the given axes.

    Args:
        axes (plt.Axes): The matplotlib axes object where the plot will be created.
        data_frame (pd.Series): The DataFrame (as a pandas Series) containing the data for the bar plot.

    Returns:
        None: The function does not return anything, but creates the plot directly on the given axes.
    """

    log_values = np.log10(data_frame)
    colors = ['g', 'y', 'r']
    axes.bar(x=range(len(data_frame)), height=log_values, color=colors, alpha=0.25, edgecolor='k')

    axes.set_ylabel('Log[Counts]')

    for i, v in enumerate(log_values):
        axes.text(i, v - 0.5, str(f'{10 ** v:.0f}'), ha='center', va='center')

    axes.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)


def apply_eccentricity_labels_in_df(data_frame: pd.DataFrame, eccentricity_labels: List[str]):
    """
    Apply eccentricity labels to the 'e_dco' values in the DataFrame.

    Args:
        data_frame (pd.DataFrame): The DataFrame containing the 'e_dco' column.
        eccentricity_labels (List[str]): A list of three labels representing the eccentricity ranges.

    Returns:
        pd.DataFrame: A new DataFrame with an additional 'Eccentricity' column containing labels based on
        'e_dco' values.
    """
    temp_df = data_frame.copy()
    temp_df['Eccentricity'] = [eccentricity_labels[0] if i <= 0.1
                               else eccentricity_labels[1] if 0.1 < i <= 0.9 else eccentricity_labels[2]
                               for i in temp_df['e_dco']]
    temp_df.reset_index(drop=True, inplace=True)

    return temp_df


def get_eccentricity_proportion(data_frame, lower_boundary=0.1, upper_boundary=0.9):
    """
    Calculate the proportions of 'e_dco' values falling below the 'lower_boundary', between 'lower_boundary' and
    'upper_boundary', and above the 'upper_boundary'.

    Args:
        data_frame (pd.DataFrame): The DataFrame containing the 'e_dco' column.
        lower_boundary (float, optional): The lower boundary for the proportion calculation. Default is 0.1.
        upper_boundary (float, optional): The upper boundary for the proportion calculation. Default is 0.9.

    Returns:
        Tuple[int, int, int]: A tuple containing the counts of values below 'below_boundary', between 'below_boundary'
        and 'above_boundary', and above 'above_boundary'.
    """

    count_below_boundary = np.sum(data_frame['e_dco'] < lower_boundary)
    count_between_boundaries = np.sum((lower_boundary < data_frame['e_dco']) & (data_frame['e_dco'] <= upper_boundary))
    count_above_boundary = np.sum(data_frame['e_dco'] > upper_boundary)

    return count_below_boundary, count_between_boundaries, count_above_boundary


def get_dco_type(data_frame: pd.DataFrame, dco_type: Union[str, int]):
    """
    Filter the DataFrame by a given 'dco_type' and reset the index.

    Args:
        data_frame (pd.DataFrame): The DataFrame to filter.
        dco_type (Union[str, int]): The value of 'dco_type' to filter by.

    Returns:
        pd.DataFrame: A new DataFrame containing rows where 'dco_type' matches the given value, with the index reset.
    """

    filtered_df = data_frame[data_frame['dco_type'] == dco_type]
    filtered_df.reset_index(drop=True, inplace=True)

    return filtered_df


def get_components(df_: pd.DataFrame):
    """
    Count occurrences of each component in the DataFrame.

    Args:
        df_ (pd.DataFrame): The DataFrame containing the 'component' column.

    Returns:
        Tuple[int, int, int]: A tuple containing the counts of components in the order (comp_0, comp_1, comp_2).
    """

    comp_0 = sum(df_['component'] == 0)
    comp_1 = sum(df_['component'] == 1)
    comp_2 = sum(df_['component'] == 2)

    return comp_0, comp_1, comp_2


def get_percentages(df_: pd.DataFrame, keyword: str):
    """
    Calculate and return the percentages of occurrences for a given keyword in the DataFrame.

    Args:
        df_ (pandas.DataFrame): The DataFrame containing the data.
        keyword (str): The column name (keyword) for which to calculate the percentages.

    Returns:
        tuple: A tuple containing percentages in the order: ('BHBH', 'NSNS', 'BHNS', and 'NSBH')
    """

    def _transform(num: float, type_: str):
        """
        Format the given numerical value as a percentage string with a specific type label.

        Args:
            num (float): The numerical value to represent as a percentage.
            type_ (str): The type label for the numerical value (e.g., 'BHBH', 'NSNS', 'NSBH', 'BHNS').

        Returns:
            str: The formatted percentage string in the format '<type_> = <percentage>%'.
        """

        formatted_percentage = f'{num * 100:.3f}%'
        result_string = f'{type_} = {formatted_percentage}'

        return result_string

    total_occurrences = sum(df_[keyword])

    per1_ = df_[keyword].loc[0] / total_occurrences
    per2_ = df_[keyword].loc[1] / total_occurrences
    per3_ = df_[keyword].loc[2] / total_occurrences
    per4_ = df_[keyword].loc[3] / total_occurrences

    formatted_per1 = _transform(num=per1_, type_='BHBH')
    formatted_per2 = _transform(num=per2_, type_='NSNS')
    formatted_per3 = _transform(num=per3_, type_='BHNS')
    formatted_per4 = _transform(num=per4_, type_='NSBH')

    return formatted_per1, formatted_per2, formatted_per3, formatted_per4


def get_max_distance_mask(file_name):
    file_ = np.load(file_name, allow_pickle=True)

    mean_distance = np.mean(np.concatenate(file_).ravel()[::2], axis=0)

    return mean_distance != min(mean_distance), mean_distance

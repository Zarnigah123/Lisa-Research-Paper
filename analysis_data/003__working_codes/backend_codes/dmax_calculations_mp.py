"""
Created on Fri Dec 31 11:23:31 2021

@author: syedalimohsinbukhari
"""

import numpy as np

import functions as fnc
from dmax_calculations import GetDMax


def create_max_distance_files(is_eccentric: bool = True):
    """
    Calculate d_max values for different DCO types and save the results to files.

    Args:
        is_eccentric (bool, optional): Whether to consider eccentric (True) or zero eccentricity (False) DCO types.
        Default is True.
    """

    dco_labels = fnc.DCO_LABELS if is_eccentric else fnc.DCO_LABELS0E

    # Get separated DataFrames for each DCO type
    bhbh_df, nsns_df, bhns_df, nsbh_df = fnc.get_separated_dcos(is_variable_ecc=is_eccentric, drop_duplicates=True)

    # Calculate and save d_max values for each DCO type
    calculate_and_save_max_distance(bhbh_df, dco_labels[0], n_proc=3)
    calculate_and_save_max_distance(nsns_df, dco_labels[1], n_proc=3)
    calculate_and_save_max_distance(nsbh_df, dco_labels[2], n_proc=3)
    calculate_and_save_max_distance(bhns_df, dco_labels[3], n_proc=3)


def calculate_and_save_max_distance(data_frame, dco_label, n_proc):
    """
    Calculate d_max values for a given DataFrame and save the results to a file.

    Args:
        data_frame (pd.DataFrame): The DataFrame containing columns 'm1_dco', 'm2_dco', 'e_dco', and 'seed'.
        dco_label (str): The label of the DCO type.
        n_proc (int): Number of processors for multiprocessing.
    """

    print(f'Starting {dco_label}\n')
    d_max_values = GetDMax(data_frame, n_proc=n_proc).get_d_max()
    np.save(f'{fnc.ANALYSIS_DATA_PATH}/{dco_label}_max_dist', d_max_values)


# create_max_distance_files(is_eccentric=True)
create_max_distance_files(is_eccentric=False)

"""Created on Sat Mar 25 04:58:20 2023."""

import h5py as h5
from numpy import logical_and, logical_or
from pandas import read_csv

from backend_codes import functions as fnc


def generate_table1_output(h5_file_to_read, csv_file_to_read):
    def get_percentage(num, den, precision=2):
        return round((num / den) * 100, precision)

    def dco_mask(dataframes, dco_types):
        return logical_and(dataframes[0] == dco_types[0], dataframes[1] == dco_types[1])

    def print_table1_values(dco_type, total_, merged_, detected_):
        print(f'{dco_type} merged/total: {merged_}/{total_} | {get_percentage(num=merged_, den=total_)}%')
        print(f'{dco_type} detected/merged: {detected_}/{merged_} | {get_percentage(num=detected_, den=merged_)}%')

    h5file = h5.File(h5_file_to_read)

    system_parameters = h5file['BSE_System_Parameters.csv']
    dco_parameters = h5file['BSE_Double_Compact_Objects.csv']

    # get the number of binaries in the file
    total_binaries_in_the_file = len(system_parameters)

    # get the number of dcos in the file
    total_dcos_in_the_file = len(dco_parameters['SEED'][()])

    # how many dcos merge in hubble time
    merge_mask = dco_parameters['Merges_Hubble_Time'][()] == 1
    total_dcos_that_merge_in_H = merge_mask.sum()

    # types of binaries that merge within hubble time
    stellar_type_1 = dco_parameters['Stellar_Type(1)'][()]
    stellar_type_2 = dco_parameters['Stellar_Type(2)'][()]

    st1_merge = stellar_type_1[merge_mask]
    st2_merge = stellar_type_2[merge_mask]

    dco_keys = [[14, 14], [13, 13], [14, 13], [13, 14]]
    dco_types = ['BHBH', 'NSNS', 'BHNS', 'NSBH']

    merged_masks = [dco_mask(dataframes=[st1_merge, st2_merge], dco_types=i) for i in dco_keys]
    total_masks = [dco_mask(dataframes=[stellar_type_1, stellar_type_2], dco_types=i) for i in dco_keys]

    bhbh_binaries = len(st1_merge[merged_masks[0]])
    nsns_binaries = len(st1_merge[merged_masks[1]])
    bhns_binaries = len(st1_merge[merged_masks[2]])
    nsbh_binaires = len(st1_merge[merged_masks[3]])

    # dco pairs that didn't merge
    bhbh_binaries_all = len(stellar_type_1[total_masks[0]])
    nsns_binaries_all = len(stellar_type_1[total_masks[1]])
    bhns_binaries_all = len(stellar_type_1[total_masks[2]])
    nsbh_binaires_all = len(stellar_type_1[total_masks[3]])

    csv_df = read_csv(csv_file_to_read)

    def get_dco_specific_data(data_frame, dco_type):
        temp_ = data_frame[logical_or(data_frame.dco_type == dco_type, data_frame.dco_type == f'{dco_type}0e')]
        temp_ = temp_.drop_duplicates(subset='seed', ignore_index=True)
        return len(temp_)

    total_ = [bhbh_binaries_all, nsns_binaries_all, bhns_binaries_all, nsbh_binaires_all]
    merged_ = [bhbh_binaries, nsns_binaries, bhns_binaries, nsbh_binaires]
    n_detections = [get_dco_specific_data(data_frame=csv_df, dco_type=i) for i in dco_types]

    print(f'Total obtained: {total_dcos_in_the_file} | '
          f'{get_percentage(num=total_dcos_in_the_file, den=total_binaries_in_the_file, precision=5)}%')
    print(f'Total merged: {total_dcos_that_merge_in_H} | '
          f'{get_percentage(num=total_dcos_that_merge_in_H, den=total_dcos_in_the_file, precision=5)}%')
    print(f'Total detections: {len(csv_df)}')

    [print_table1_values(dco_type=dco_type, total_=total_binaries, merged_=merged_binaries, detected_=n_detection)
     for dco_type, total_binaries, merged_binaries, n_detection in zip(dco_types, total_, merged_, n_detections)]


generate_table1_output(fnc.VARIABLE_ECCENTRICITY_H5_FILE, fnc.VARIABLE_ECC_CSV)
print('')
generate_table1_output(fnc.ZERO_ECCENTRICITY_H5_FILE, fnc.ZERO_ECC_CSV)

"""Created on Mon Dec 13 16:11:59 2021"""

from backend_codes import functions as fnc

# Process DCO types in the merged HDF5 file with variable eccentricity
fnc.separate_dco_types(merged_h5_file=fnc.VARIABLE_ECCENTRICITY_H5_FILE,
                       merge_within_hubble_time=True,
                       separate_nsbh_binaries=True)

# Process DCO types in the merged HDF5 file with zero eccentricity
fnc.separate_dco_types(merged_h5_file=fnc.ZERO_ECCENTRICITY_H5_FILE,
                       merge_within_hubble_time=True,
                       separate_nsbh_binaries=True)

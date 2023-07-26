"""Created on Thu Feb 10 11:31:16 2022"""

import astropy.units as u

from backend_codes import functions as fnc

# Define parameters for the datasets
n_galaxies = 100
mission_duration = 4 * u.yr
milkyway_size = 1_000_000
snr_cutoff = 7

# Define the DCO types to process
dco_types = ['BHBH', 'NSNS', 'NSBH', 'BHNS']

# Loop over DCO types and create detectable datasets with variable eccentricity
for dco_type in dco_types:
    fnc.make_detectable_dataset(path_to_h5_file=fnc.VARIABLE_ECCENTRICITY_H5_FILE,
                                number_of_galaxy_instances=n_galaxies,
                                binary_type=dco_type,
                                observation_time=mission_duration,
                                milkyway_size=milkyway_size,
                                snr_cutoff=snr_cutoff)

# Loop over DCO types and create detectable datasets with zero eccentricity
for dco_type in dco_types:
    fnc.make_detectable_dataset(path_to_h5_file=fnc.ZERO_ECCENTRICITY_H5_FILE,
                                number_of_galaxy_instances=n_galaxies,
                                binary_type=dco_type,
                                observation_time=mission_duration,
                                milkyway_size=milkyway_size,
                                snr_cutoff=snr_cutoff,
                                variable_eccentricity=False)

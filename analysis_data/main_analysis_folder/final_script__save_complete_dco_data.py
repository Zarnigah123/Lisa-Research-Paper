"""Created on Fri Mar 24 20:52:26 2023."""

import h5py as h5
import numpy as np
import pandas as pd


def condition(high_value, parameter, key_, secondary_key=None, out=None, low_value=None):
    if low_value:
        cond = np.logical_and(low_value < parameter[key_], parameter[key_] <= high_value)
    else:
        cond = parameter[key_] <= high_value

    if out:
        return parameter[cond][secondary_key]
    else:
        return parameter[cond]


file_path = '/media/astrophysicsandpython/DATA_DRIVE0/h5out.h5'

h5out = h5.File(file_path)

# get_parameters
sys_values = h5out['BSE_System_Parameters.csv']
dco_values = h5out['BSE_Double_Compact_Objects.csv']

# find the first three seed that merges
merged = np.where(dco_values['Merges_Hubble_Time'][()] == 1)[0]

all_seeds = dco_values['SEED'][()]

# get their seed number
seeds = dco_values['SEED'][()]
m1_dco = dco_values['Mass(1)'][()]
m2_dco = dco_values['Mass(2)'][()]
a_dco = dco_values['SemiMajorAxis@DCO'][()]
e_dco = dco_values['Eccentricity@DCO'][()]
stype1 = dco_values['Stellar_Type(1)'][()]
stype2 = dco_values['Stellar_Type(2)'][()]
rem_mass_prsp = [4] * len(m1_dco)
evolve_pulsars = [True] * len(m1_dco)

# get their zams parameters
dco_loc = np.in1d(sys_values['SEED'][()], all_seeds)

# get sys_parameter values
sys_seeds = sys_values['SEED'][()][dco_loc]
weights = np.array([1] * len(sys_seeds))
m1_zams = sys_values['Mass@ZAMS(1)'][()][dco_loc]
m2_zams = sys_values['Mass@ZAMS(2)'][()][dco_loc]
a_zams = sys_values['SemiMajorAxis@ZAMS'][()][dco_loc]
e_zams = sys_values['Eccentricity@ZAMS'][()][dco_loc]
z_zams = sys_values['Metallicity@ZAMS(1)'][()][dco_loc]
km1_zams = sys_values['Kick_Magnitude_Random(1)'][()][dco_loc]
km2_zams = sys_values['Kick_Magnitude_Random(2)'][()][dco_loc]
kp1_zams = sys_values['Kick_Phi(1)'][()][dco_loc]
kp2_zams = sys_values['Kick_Phi(2)'][()][dco_loc]
kt1_zams = sys_values['Kick_Theta(1)'][()][dco_loc]
kt2_zams = sys_values['Kick_Theta(2)'][()][dco_loc]
kma1_zams = sys_values['Kick_Mean_Anomaly(1)'][()][dco_loc]
kma2_zams = sys_values['Kick_Mean_Anomaly(2)'][()][dco_loc]
stype_sys1 = sys_values['Stellar_Type@ZAMS(1)'][()][dco_loc]
stype_sys2 = sys_values['Stellar_Type@ZAMS(2)'][()][dco_loc]

df = pd.DataFrame([dco_values['Merges_Hubble_Time'][()],
                   sys_seeds, weights, m1_zams, m1_dco, m2_zams, m2_dco, a_zams, a_dco, e_zams,
                   e_dco, z_zams, stype_sys1, stype1, stype_sys2, stype2, km1_zams, km2_zams,
                   kp1_zams, kp2_zams, kt1_zams, kt2_zams, kma1_zams, kma2_zams]).T

df.columns = ['Merges', 'Seed', 'Weight', 'Mass@ZAMS(1)', 'Mass@DCO(1)', 'Mass@ZAMS(2)',
              'Mass@DCO(2)', 'SemiMajorAxis@ZAMS', 'SemiMajorAxis@DCO', 'Eccentricity@ZAMS',
              'Eccentricity@DCO', 'Metallicity@ZAMS', 'StellarType@ZAMS(1)', 'StellarType@DCO(1)',
              'StellarType@ZAMS(2)', 'StellarType@DCO(2)', 'KickMagnitude(1)', 'KickMagnitude(2)',
              'KickPhi(1)', 'KickPhi(2)', 'KickTheta(1)', 'KickTheta(2)', 'KickMeanAnomaly(1)',
              'KickMeanAnomaly(2)']

df['Seed'] = df['Seed'].apply(int)
df = df.sort_values(by='Seed')
df.reset_index(drop=True, inplace=True)

df.to_csv('./combined_dataset.csv', header=True, index=False)


df_merges = df[df.Merges == 1]
df_merges.reset_index(drop=True, inplace=True)

df_merges.to_csv('./combined_dataset_merged_only.csv', header=True, index=False)

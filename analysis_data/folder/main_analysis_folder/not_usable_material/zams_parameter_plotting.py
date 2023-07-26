"""Created on Mar 09 11:02:09 2023."""

from itertools import chain

import h5py as h5
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# np.random.seed(6600000)

file_path = '/media/astrophysicsandpython/A6EA5CEFEA5CBD6D/ms_project/h5out.h5'

h5out = h5.File(file_path)

# get_parameters
sys_values = h5out['BSE_System_Parameters.csv']
dco_values = h5out['BSE_Double_Compact_Objects.csv']

# find the first three seed that merges
merged = np.where(dco_values['Merges_Hubble_Time'][()] == 1)[0]

# get their seed number
seeds = dco_values['SEED'][()][merged]
# get their dco parameters
m1_dco = dco_values['Mass(1)'][()][merged]
m2_dco = dco_values['Mass(2)'][()][merged]
a_dco = dco_values['SemiMajorAxis@DCO'][()][merged]
e_dco = dco_values['Eccentricity@DCO'][()][merged]
stype1 = dco_values['Stellar_Type(1)'][()][merged]
stype2 = dco_values['Stellar_Type(2)'][()][merged]
rem_mass_prsp = [4] * len(m1_dco)
evolve_pulsars = [True] * len(m1_dco)

# get their zams parameters
zams_loc = np.array(list(chain.from_iterable([np.where(sys_values['SEED'][()] == i)[0]
                                              for i in seeds])))
# get sys_parameter values
sys_seeds = sys_values['SEED'][()][zams_loc]
weights = np.array([1] * len(sys_seeds))
m1_zams = sys_values['Mass@ZAMS(1)'][()][zams_loc]
m2_zams = sys_values['Mass@ZAMS(2)'][()][zams_loc]
a_zams = sys_values['SemiMajorAxis@ZAMS'][()][zams_loc]
e_zams = sys_values['Eccentricity@ZAMS'][()][zams_loc]
z_zams = sys_values['Metallicity@ZAMS(1)'][()][zams_loc]
km1_zams = sys_values['Kick_Magnitude_Random(1)'][()][zams_loc]
km2_zams = sys_values['Kick_Magnitude_Random(2)'][()][zams_loc]
kp1_zams = sys_values['Kick_Phi(1)'][()][zams_loc]
kp2_zams = sys_values['Kick_Phi(2)'][()][zams_loc]
kt1_zams = sys_values['Kick_Theta(1)'][()][zams_loc]
kt2_zams = sys_values['Kick_Theta(2)'][()][zams_loc]
kma1_zams = sys_values['Kick_Mean_Anomaly(1)'][()][zams_loc]
kma2_zams = sys_values['Kick_Mean_Anomaly(2)'][()][zams_loc]
stype_sys1 = sys_values['Stellar_Type@ZAMS(1)'][()][zams_loc]
stype_sys2 = sys_values['Stellar_Type@ZAMS(2)'][()][zams_loc]

df = pd.DataFrame([sys_seeds, weights, m1_zams, m2_zams, a_zams, e_zams, z_zams,
                   stype_sys1, stype_sys2,
                   km1_zams, km2_zams, kp1_zams, kp2_zams, kt1_zams, kt2_zams,
                   kma1_zams, kma2_zams]).T
df.columns = ['Seed', 'Weight', 'Mass@ZAMS(1)', 'Mass@ZAMS(2)', 'SemiMajorAxis@ZAMS',
              'Eccentricity@ZAMS', 'Metallicity@ZAMS', 'StellarType(1)', 'StellarType(2)',
              'KickMagnitude(1)', 'KickMagnitude(2)', 'KickPhi(1)', 'KickPhi(2)',
              'KickTheta(1)', 'KickTheta(2)', 'KickMeanAnomaly(1)', 'KickMeanAnomaly(2)']


def condition(high_value, parameter, key_, secondary_key=None, out=None, low_value=None):
    if low_value:
        cond = (low_value < parameter[key_]) & (parameter[key_] <= high_value)
    else:
        cond = parameter[key_] <= high_value

    if out:
        return parameter[cond][secondary_key]
    else:
        return parameter[cond]


a = np.linspace(1e-4, 0.03, 5)

df.loc[:, 'cond'] = np.zeros(len(df['SemiMajorAxis@ZAMS']))

conds = [condition(i, df, 'Metallicity@ZAMS', low_value=j) for i, j in zip(a[1:], a[:-1])]

for i in conds:
    i['cond'] = f"{round(i['Metallicity@ZAMS'].max(), 6)}"

combined_ = pd.concat(conds)
combined_.reset_index(inplace=True, drop=True)

def make_filled_kde_plot(dataframe, x_val, hue, axes, log_scale, multiple, get_ax=False):
    legend = get_ax
    p = sns.kdeplot(dataframe, x=x_val, hue=hue, palette='RdYlGn', ax=axes, log_scale=log_scale,
                    legend=legend, multiple=multiple)

    if get_ax:
        return p


#################################################################################
# important figure
#################################################################################

f, ax = plt.subplots(2, 2, figsize=(10, 8), sharey='all')

make_filled_kde_plot(combined_, 'Mass@ZAMS(1)', 'cond', ax[0][0], (True, False), 'fill')
make_filled_kde_plot(combined_, 'Mass@ZAMS(2)', 'cond', ax[0][1], (True, False), 'fill')
make_filled_kde_plot(combined_, 'SemiMajorAxis@ZAMS', 'cond', ax[1][0], (True, False), 'fill')
p = make_filled_kde_plot(combined_, 'Eccentricity@ZAMS', 'cond', ax[1][1], (False, False), 'fill',
                         get_ax=True)
plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.legend(p.get_children()[0:4], a[1:], title='Metallicity', bbox_to_anchor=[0.5, 2.42], ncol=4)

plt.savefig('zams_parameters.pdf')
plt.close()

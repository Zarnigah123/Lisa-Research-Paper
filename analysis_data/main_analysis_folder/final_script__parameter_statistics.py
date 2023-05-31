"""Created on Tue May 23 04:07:53 2023."""

import numpy as np
import pandas as pd

data = pd.read_csv('./combined_dataset.csv',delimiter=',')


merges = data[data.Merges == 1]
merges.reset_index(inplace=True, drop=True)

st1, st2 = merges['StellarType@DCO(1)'], merges['StellarType@DCO(2)']

mask_bhbh = np.logical_and(st1 == 14, st2 == 14)

bhbh = merges[mask_bhbh]
bhbh.reset_index(inplace=True, drop=True)

mask_nsns = np.logical_and(st1 == 13, st2 == 13)

nsns = merges[mask_nsns]
nsns.reset_index(drop=True, inplace=True)

mask_nsbh = np.logical_and(st1 == 13, st2 == 14)

nsbh = merges[mask_nsbh]
nsbh.reset_index(drop=True, inplace=True)

mask_bhns = np.logical_and(st1 == 14, st2 == 13)

bhns = merges[mask_bhns]
bhns.reset_index(drop=True, inplace=True)


def get_max_min_ZAMS_DCO(df, max_=True):

    pars = ['Mass@ZAMS(1)', 'Mass@ZAMS(2)', 'Mass@DCO(1)', 'Mass@DCO(2)']

    if max_:
        m1zams = df[df['Mass@ZAMS(1)'] == max(df['Mass@ZAMS(1)'])]
        m2zams = df[df['Mass@ZAMS(2)'] == max(df['Mass@ZAMS(2)'])]
        m1dco = df[df['Mass@DCO(1)'] == max(df['Mass@DCO(1)'])]
        m2dco = df[df['Mass@DCO(2)'] == max(df['Mass@DCO(2)'])]
    else:
        m1zams = df[df['Mass@ZAMS(1)'] == min(df['Mass@ZAMS(1)'])]
        m2zams = df[df['Mass@ZAMS(2)'] == min(df['Mass@ZAMS(2)'])]
        m1dco = df[df['Mass@DCO(1)'] == min(df['Mass@DCO(1)'])]
        m2dco = df[df['Mass@DCO(2)'] == min(df['Mass@DCO(2)'])]

    m1zams_ = [round(np.array(m1zams[k])[0], 3) for k in pars]
    m2zams_ = [round(np.array(m2zams[k])[0], 3) for k in pars]
    m1dco_ = [round(np.array(m1dco[k])[0], 3) for k in pars]
    m2dco_ = [round(np.array(m2dco[k])[0], 3) for k in pars]


    print('maxM1Z,primary,secondary,primary,secondary')
    print(m1zams_[0], ',', m1zams_[0], ',', m1zams_[1], ',', m1zams_[2], ',', m1zams_[3])

    print('')

    print('maxM2Z,primary,secondary,primary,secondary')
    print(m2zams_[1], ',', m2zams_[0], ',', m2zams_[1], ',', m2zams_[2], ',', m2zams_[3])

    print('')

    print('maxM1D,primary,secondary,primary,secondary')
    print(m1dco_[2], ',', m1dco_[0], ',', m1dco_[1], ',', m1dco_[2], ',', m1dco_[3])

    print('')

    print('maxM2Z,primary,secondary,primary,secondary')
    print(m2dco_[3], ',', m2dco_[0], ',', m2dco_[1], ',', m2dco_[2], ',', m2dco_[3])

if __name__ == '__main__':
    print('\nBHBH\n')
    get_max_min_ZAMS_DCO(df=bhbh)
    get_max_min_ZAMS_DCO(df=bhbh, max_=False)
    print('\nNSNS\n')
    get_max_min_ZAMS_DCO(df=nsns)
    get_max_min_ZAMS_DCO(df=nsns, max_=False)
    print('\nBHNS\n')
    get_max_min_ZAMS_DCO(df=bhns)
    get_max_min_ZAMS_DCO(df=bhns, max_=False)
    print('\nNSBH\n')
    get_max_min_ZAMS_DCO(df=nsbh)
    get_max_min_ZAMS_DCO(df=nsbh, max_=False)

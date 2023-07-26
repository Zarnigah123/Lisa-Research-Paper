# -*- coding: utf-8 -*-

import h5py as h5
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_dco_properties(stellar_condition, dco_type, image_type=None):
    if image_type is None:
        image_type = ['pdf', 'png']

    csv_file = pd.read_csv('combined_dataset.csv')

    main_ = csv_file[['Mass@DCO(1)', 'Mass@DCO(2)', 'SemiMajorAxis@DCO', 'Eccentricity@DCO',
                      'Metallicity@ZAMS', 'StellarType@DCO(1)', 'StellarType@DCO(2)', 'cond']]

    main_.reset_index(drop=True, inplace=True)

    stellar_condition = np.logical_and(main_['StellarType@DCO(1)'] == stellar_condition[0],
                                       main_['StellarType@DCO(2)'] == stellar_condition[1])

    main_ = main_[stellar_condition]
    main_.reset_index(drop=True, inplace=True)

    main_ = main_.drop(['StellarType@DCO(1)', 'StellarType@DCO(2)', 'cond'], axis=1)

    evolved_ = h5.File(f'/home/astrophysicsandpython/Dropbox/sirasad/analysis_data/'
                       f'{dco_type}.h5')['simulation'][()]

    evolved_ = pd.DataFrame(evolved_)

    main_['SemiMajorAxis@DCO'] = main_['SemiMajorAxis@DCO'].apply(lambda x: np.log10(x))
    evolved_['a_dco'] = evolved_['a_dco'].apply(lambda x: np.log10(x))
    evolved_['a_zams'] = evolved_['a_zams'].apply(lambda x: np.log10(x))
    evolved_['a_lisa'] = evolved_['a_lisa'].apply(lambda x: np.log10(x))

    def drop_duplicates(df, key_):
        df = df[~df.duplicated(subset=[key_], keep='first')]
        df.reset_index(drop=True, inplace=True)
        return df

    def get_par_bins(df, par, n_bins=32):
        return np.histogram_bin_edges(df[par], bins=n_bins)

    def plot_overlapping_hist(pars, n_bins, df2, axes):
        sns.histplot(main_, x=pars[0], color='k', alpha=0.25, bins=n_bins,
                     log_scale=(False, True), **{'lw': 0}, ax=axes)
        sns.histplot(df2, x=pars[1], bins=n_bins, **{'lw': 0}, ax=axes)

    def wrap_sns_hist(pars, axes):
        und = drop_duplicates(evolved_, pars[1])
        m_bins = get_par_bins(main_, pars[0])
        plot_overlapping_hist(pars, m_bins, und, axes)

    f, ax = plt.subplots(2, 2, figsize=(10, 8))

    wrap_sns_hist(['Mass@DCO(1)', 'm1_dco'], ax[0][0])
    wrap_sns_hist(['Mass@DCO(2)', 'm2_dco'], ax[0][1])
    wrap_sns_hist(['SemiMajorAxis@DCO', 'a_dco'], ax[1][0])
    wrap_sns_hist(['Eccentricity@DCO', 'e_dco'], ax[1][1])

    for i1, v1 in enumerate(ax):
        for i2, v2 in enumerate(v1):
            if i2 == 1:
                v2.set_ylabel('')

    plt.tight_layout()
    [plt.savefig(f'{dco_type}_params.{i}') for i in image_type]
    plt.close()

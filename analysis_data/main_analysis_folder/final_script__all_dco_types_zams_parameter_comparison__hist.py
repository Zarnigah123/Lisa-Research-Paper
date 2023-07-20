"""Created on Fri Mar 24 23:44:55 2023."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

csv_file = pd.read_csv('combined_dataset.csv')


def merged_and_unmerged(data, line_style, axes, x_label, palette=None, ret_=False,
                        log_scale=(True, True)):
    sns_obj = sns.histplot(data, log_scale=log_scale, ax=axes,
                           common_norm=True, palette=palette, common_bins=True, multiple='dodge',
                           bins=16)

    axes.set_xlabel(x_label)

    if ret_:
        return sns_obj


f, ax = plt.subplots(2, 4, figsize=(16, 8))

masses = ['Mass@ZAMS(1)', 'Mass@DCO(1)', 'Mass@ZAMS(2)', 'Mass@DCO(2)']

label_mass = r'Mass [$M_\odot$]'

p = merged_and_unmerged([csv_file[masses[0]], csv_file[masses[1]]], '--', ax[0][0], label_mass,
                        palette='RdYlGn', ret_=True)

p = merged_and_unmerged([csv_file[masses[2]], csv_file[masses[3]]], '--', ax[0][1], label_mass,
                        palette='RdYlGn', ret_=True)

semi_axes = ['SemiMajorAxis@ZAMS', 'SemiMajorAxis@DCO']

label_semi = r'$a_\mathrm{AU}$'

p = merged_and_unmerged([csv_file[semi_axes[0]], csv_file[semi_axes[1]]], '-.', ax[0][2],
                        label_semi,
                        palette='plasma', ret_=True)

ecc = ['Eccentricity@ZAMS', 'Eccentricity@DCO']

label_ecc = r'$e$'

p = merged_and_unmerged([csv_file[ecc[0]], csv_file[ecc[1]]], '-', ax[0][3], label_ecc,
                        palette='bwr', ret_=True, log_scale=(False, True))

k_rand = ['KickMagnitude(1)', 'KickMagnitude(2)']

label_k_rand = 'Kick Random Magnitude'

p = merged_and_unmerged([csv_file[k_rand[0]], csv_file[k_rand[1]]], '--', ax[1][0], label_k_rand,
                        palette='Paired', ret_=True, log_scale=(False, True))

k_phi = ['KickPhi(1)', 'KickPhi(2)']

label_k_phi = 'Kick Phi'

p = merged_and_unmerged([csv_file[k_phi[0]], csv_file[k_phi[1]]], '-.', ax[1][1], label_k_phi,
                        palette='CMRmap', ret_=True, log_scale=(False, True))

k_theta = ['KickTheta(1)', 'KickTheta(2)']

label_k_theta = 'Kick Theta'

p = merged_and_unmerged([csv_file[k_theta[0]], csv_file[k_theta[1]]], '-', ax[1][2], label_k_theta,
                        palette='Set1', ret_=True, log_scale=(False, False))

k_mean = ['KickMeanAnomaly(1)', 'KickMeanAnomaly(2)']

label_k_mean = 'Kick Mean Anomaly'

p = merged_and_unmerged([csv_file[k_mean[0]], csv_file[k_mean[1]]], ':', ax[1][3], label_k_mean,
                        palette='gist_rainbow', ret_=True, log_scale=(False, False))

plt.tight_layout()

# plt.savefig('all_zams_params.pdf')
# plt.savefig('all_zams_params.png')
# plt.close()

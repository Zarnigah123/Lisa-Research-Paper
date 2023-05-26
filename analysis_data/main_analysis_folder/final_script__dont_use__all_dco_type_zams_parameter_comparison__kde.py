"""Created on Fri Mar 24 23:44:55 2023."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

csv_file = pd.read_csv('combined_dataset.csv')


def apply_legend(axes, seaborn_object, labels, title):
    axes.legend(seaborn_object.get_children()[0:len(labels) * 2], labels[::-1], title=title,
                loc='best')


def merged_and_unmerged(data, line_style, axes, x_label, palette=None, ret_=False,
                        log_scale=(True, True)):
    sns_obj = sns.kdeplot(data, log_scale=log_scale, **{'ls': line_style}, ax=axes,
                          common_norm=True, palette=palette, cut=0)

    axes.set_xlabel(x_label)

    if ret_:
        return sns_obj


f, ax = plt.subplots(2, 4, figsize=(16, 8))

masses = ['Mass@DCO(1)', 'Mass@ZAMS(1)', 'Mass@DCO(2)', 'Mass@ZAMS(2)']

label_mass = r'Mass [$M_\odot$]'

p = merged_and_unmerged([csv_file[masses[0]], csv_file[masses[1]]], '--', ax[0][0], label_mass,
                        palette='RdYlGn', ret_=True)

apply_legend(ax[0][0], p, masses[0:2], label_mass)

p = merged_and_unmerged([csv_file[masses[2]], csv_file[masses[3]]], '--', ax[0][1], label_mass,
                        palette='RdYlGn', ret_=True)

apply_legend(ax[0][1], p, masses[2:], label_mass)

semi_axes = ['SemiMajorAxis@DCO', 'SemiMajorAxis@ZAMS']

label_semi = r'$a_\mathrm{AU}$'

p = merged_and_unmerged([csv_file[semi_axes[0]], csv_file[semi_axes[1]]], '-.', ax[0][2],
                        label_semi,
                        palette='plasma', ret_=True)

apply_legend(ax[0][2], p, semi_axes, label_semi)

ecc = ['Eccentricity@DCO', 'Eccentricity@ZAMS']

label_ecc = r'$e$'

p = merged_and_unmerged([csv_file[ecc[0]], csv_file[ecc[1]]], '-', ax[0][3], label_ecc,
                        palette='bwr', ret_=True, log_scale=(False, True))

apply_legend(ax[0][3], p, ecc, label_ecc)

k_rand = ['KickMagnitude(2)', 'KickMagnitude(1)']

label_k_rand = 'Kick Random Magnitude'

p = merged_and_unmerged([csv_file[k_rand[0]], csv_file[k_rand[1]]], '--', ax[1][0], label_k_rand,
                        palette='Paired', ret_=True, log_scale=(False, False))

apply_legend(ax[1][0], p, k_rand, label_k_rand)

k_phi = ['KickPhi(2)', 'KickPhi(1)']

label_k_phi = 'Kick Phi'

p = merged_and_unmerged([csv_file[k_phi[0]], csv_file[k_phi[1]]], '-.', ax[1][1], label_k_phi,
                        palette='CMRmap', ret_=True, log_scale=(False, False))

apply_legend(ax[1][1], p, k_phi, label_k_phi)

k_theta = ['KickTheta(2)', 'KickTheta(1)']

label_k_theta = 'Kick Theta'

p = merged_and_unmerged([csv_file[k_theta[0]], csv_file[k_theta[1]]], '-', ax[1][2], label_k_theta,
                        palette='Set1', ret_=True, log_scale=(False, False))

apply_legend(ax[1][2], p, k_theta, label_k_theta)

k_mean = ['KickMeanAnomaly(2)', 'KickMeanAnomaly(1)']

label_k_mean = 'Kick Mean Anomaly'

p = merged_and_unmerged([csv_file[k_mean[0]], csv_file[k_mean[1]]], ':', ax[1][3], label_k_mean,
                        palette='gist_rainbow', ret_=True, log_scale=(False, False))

apply_legend(ax[1][3], p, k_mean, label_k_mean)

plt.tight_layout()

# plt.savefig('all_zams_params.pdf')
# plt.savefig('all_zams_params.png')
# plt.close()

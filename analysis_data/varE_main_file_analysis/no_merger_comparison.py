"""Created on Fri Mar 24 23:44:55 2023."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

csv_file = pd.read_csv('./combined_dataset.csv')


#################################################################################
# another important figure
#################################################################################

def apply_legend(axes, seaborn_object, labels, title):
    axes.legend(seaborn_object.get_children()[0:len(labels) * 2], labels, title=title, loc='best')


def merged_and_unmerged(data, line_style, axes, x_label, palette=None, ret_=False,
                        log_scale=(True, True)):
    sns_obj = sns.kdeplot(data, log_scale=log_scale, **{'ls': line_style}, ax=axes,
                          common_norm=True, palette=palette, cut=0)

    axes.set_xlabel(x_label)

    if ret_:
        return sns_obj


f, ax = plt.subplots(2, 4, figsize=(16, 8))

masses = ['Mass@ZAMS(1)', 'Mass@DCO(1)', 'Mass@ZAMS(2)', 'Mass@DCO(2)']

label_mass = r'Mass [$M_\odot$]'

p = merged_and_unmerged([csv_file[masses[0]], csv_file[masses[1]]], '--', ax[0][0], label_mass,
                        palette='RdYlGn', ret_=True)

apply_legend(ax[0][0], p, masses[0:2], label_mass)

p = merged_and_unmerged([csv_file[masses[2]], csv_file[masses[3]]], '--', ax[0][1], label_mass,
                        palette='RdYlGn', ret_=True)

apply_legend(ax[0][1], p, masses[2:], label_mass)

semiaxes = ['SemiMajorAxis@ZAMS', 'SemiMajorAxis@DCO']

label_semi = r'$a_\mathrm{AU}$'

p = merged_and_unmerged([csv_file[semiaxes[0]], csv_file[semiaxes[1]]], '-.', ax[0][2], label_semi,
                        palette='plasma', ret_=True)

apply_legend(ax[0][2], p, semiaxes, label_semi)

ecc = ['Eccentricity@ZAMS', 'Eccentricity@DCO']

label_ecc = r'$e$'

p = merged_and_unmerged([csv_file[ecc[0]], csv_file[ecc[1]]], '-', ax[0][3], label_ecc,
                        palette='bwr', ret_=True, log_scale=(False, True))

apply_legend(ax[0][3], p, ecc, label_ecc)

krand = ['KickMagnitude(1)', 'KickMagnitude(2)']

label_krand = 'Kick Random Magnitude'

p = merged_and_unmerged([csv_file[krand[0]], csv_file[krand[1]]], '--', ax[1][0], label_krand,
                        palette='Paired', ret_=True)

apply_legend(ax[1][0], p, krand, label_krand)

kphi = ['KickPhi(1)', 'KickPhi(2)']

label_kphi = 'Kick Phi'

p = merged_and_unmerged([csv_file[kphi[0]], csv_file[kphi[1]]], '-.', ax[1][1], label_kphi,
                        palette='CMRmap', ret_=True, log_scale=(False, False))

apply_legend(ax[1][1], p, kphi, label_kphi)

ktheta = ['KickTheta(1)', 'KickTheta(2)']

label_ktheta = 'Kick Theta'

p = merged_and_unmerged([csv_file[ktheta[0]], csv_file[ktheta[1]]], '-', ax[1][2], label_ktheta,
                        palette='Set1', ret_=True, log_scale=(False, False))

apply_legend(ax[1][2], p, ktheta, label_ktheta)

kmean = ['KickMeanAnomaly(1)', 'KickMeanAnomaly(2)']

label_kmean = 'Kick Mean Anomaly'

p = merged_and_unmerged([csv_file[kmean[0]], csv_file[kmean[1]]], ':', ax[1][3], label_kmean,
                        palette='gist_rainbow', ret_=True, log_scale=(False, False))

apply_legend(ax[1][3], p, kmean, label_kmean)

plt.tight_layout()

plt.savefig('all_zams_params.pdf')
plt.close()

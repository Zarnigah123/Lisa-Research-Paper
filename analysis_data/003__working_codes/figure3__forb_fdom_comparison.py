"""Created on Thu Jun 22 14:05:47 2023."""

import matplotlib.pyplot as plt
import numpy as np

from backend_codes import functions as fnc

df = fnc.VARIABLE_ECC_CSV_DF

df = fnc.sort_reset_drop_df(df_=df, sort_key='seed')

labels_ = [r'e$_\mathrm{DCO} \leq 0.1$', r'$0.1 <$ e$_\mathrm{DCO} \leq 0.9$', r'e$_\mathrm{DCO} > 0.9$']

df2 = fnc.apply_eccentricity_labels_in_df(data_frame=df, eccentricity_labels=labels_)

df_bhbh = fnc.get_dco_type(df2, 'BHBH')
df_nsns = fnc.get_dco_type(df2, 'NSNS')
df_bhns = fnc.get_dco_type(df2, 'BHNS')
df_nsbh = fnc.get_dco_type(df2, 'NSBH')

fig, ax = plt.subplots(3, 4, figsize=(16, 10))

bhbh_proportions = fnc.get_eccentricity_proportion(data_frame=df_bhbh)
nsns_proportions = fnc.get_eccentricity_proportion(data_frame=df_nsns)
bhns_proportions = fnc.get_eccentricity_proportion(data_frame=df_bhns)
nsbh_proportions = fnc.get_eccentricity_proportion(data_frame=df_nsbh)

bins = np.histogram_bin_edges(np.log10(df_bhns['f_orb']), bins=32)
bins2 = np.histogram_bin_edges(np.log10(df_bhns['f_orb'] * df_bhns['SNR_harmonics']), bins=32)

sns_handles, _ = fnc.seaborn_plot(df_bhbh, axes=ax[0][0], title='BHBH', bins=bins, get_legend=True,
                                  hue_labels=labels_)
ax[0][0].set_xlim([1e-7, 1e-2])
fnc.seaborn_plot(df_nsns, axes=ax[0][1], title='NSNS', bins=bins, hue_labels=labels_)
ax[0][1].set_xlim([1e-7, 1e-2])
fnc.seaborn_plot(df_bhns, axes=ax[0][2], title='BHNS', bins=bins, hue_labels=labels_)
ax[0][2].set_xlim([1e-7, 1e-2])
fnc.seaborn_plot(df_nsbh, axes=ax[0][3], title='NSBH', bins=bins, hue_labels=labels_)
ax[0][3].set_xlim([1e-7, 1e-2])

fnc.plot_f_orb(axes=ax[1][0], data_frame=df_bhbh, bins=bins2, hue_order=labels_)
fnc.plot_f_orb(axes=ax[1][1], data_frame=df_nsns, bins=bins2, hue_order=labels_)
fnc.plot_f_orb(axes=ax[1][2], data_frame=df_bhns, bins=bins2, hue_order=labels_)
fnc.plot_f_orb(axes=ax[1][3], data_frame=df_nsbh, bins=bins2, hue_order=labels_)

fnc.bar_plot(axes=ax[2][0], data_frame=bhbh_proportions)
fnc.bar_plot(axes=ax[2][1], data_frame=nsns_proportions)
fnc.bar_plot(axes=ax[2][2], data_frame=bhns_proportions)
fnc.bar_plot(axes=ax[2][3], data_frame=nsbh_proportions)

[i.set_ylabel('') for j in ax for i in j[1:]]
plt.tight_layout()
plt.figlegend(sns_handles, labels_, ncols=3, title='Eccentricity', loc='upper center')
fig.subplots_adjust(top=0.9)

fnc.save_figure(pyplot_object=plt, fig_name='dco_fdom_ecc_details')
plt.close()

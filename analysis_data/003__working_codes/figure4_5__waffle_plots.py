"""Created on Sat Jun 17 17:22:17 2023."""

import matplotlib.pyplot as plt
import pandas as pd
from pywaffle import Waffle

from backend_codes import functions as fnc

bhbh, nsns, bhns, nsbh = fnc.get_separated_dcos(is_variable_ecc=True)

pct_label = ['low-alpha', 'high-alpha', 'bulge']

bhbh_comp = fnc.get_components(df_=bhbh)
nsns_comp = fnc.get_components(df_=nsns)
bhns_comp = fnc.get_components(df_=bhns)
nsbh_comp = fnc.get_components(df_=nsbh)

dict_ = {'dco_type': ['BHBH', 'NSNS', 'NSBH', 'BHNS'],
         'comp0': [bhbh_comp[0], nsns_comp[0], bhns_comp[0], nsbh_comp[0]],
         'comp1': [bhbh_comp[1], nsns_comp[1], bhns_comp[1], nsbh_comp[1]],
         'comp2': [bhbh_comp[2], nsns_comp[2], bhns_comp[2], nsbh_comp[2]],
         'colors': ['#ff028d', '#cea2fd', 'red', '#99ff99']
         }

waffle_df = pd.DataFrame(dict_)

total_ = sum(dict_['comp0'] + dict_['comp1'] + dict_['comp2'])

low_alpha_percentage = sum(dict_['comp0']) / total_
low_alpha_percentage_ = r'Low$-[\alpha/\mathrm{Fe}]$: ' + f'{low_alpha_percentage * 100:.2f}%'

high_alpha_percentage = sum(dict_['comp1']) / total_
high_alpha_percentage_ = r'High$-[\alpha/\mathrm{Fe}]$: ' + f'{high_alpha_percentage * 100:.2f}%'

bulge_percentage = sum(dict_['comp2']) / total_
bulge_percentage_ = 'Bulge population: ' + f'{bulge_percentage * 100:.2f}%'

plt.figure(FigureClass=Waffle, figsize=(10, 10), rows=5,
           plots={
               311: {
                   'values': list(waffle_df['comp0']),
                   'labels': fnc.get_percentages(waffle_df, 'comp0'),
                   'legend': {'loc': 'lower center', 'bbox_to_anchor': (0.5, -0.2), 'ncol': 4},
                   'title': {'label': r'Low$-[\alpha/\mathrm{Fe}]$ population'},
               },
               312: {
                   'values': list(waffle_df['comp1']),
                   'labels': fnc.get_percentages(waffle_df, 'comp1'),
                   'legend': {'loc': 'lower center', 'bbox_to_anchor': (0.5, -0.2), 'ncol': 4},
                   'title': {'label': r'High$-[\alpha/\mathrm{Fe}]$ population'}
               },
               313: {
                   'values': list(waffle_df['comp2']),
                   'labels': fnc.get_percentages(waffle_df, 'comp2'),
                   'legend': {'loc': 'lower center', 'bbox_to_anchor': (0.5, -0.2), 'ncol': 4},
                   'title': {'label': 'Bulge population'}
               }
           },
           columns=20, rounding_rule='nearest', cmap_name='Dark2')

plt.tight_layout()
fnc.save_figure(pyplot_object=plt, fig_name='dco_type_MW_component_distribution')
plt.close()

plt.figure(FigureClass=Waffle, figsize=(10, 3.5), rows=5,
           plots={
               111: {
                   'values': [low_alpha_percentage, high_alpha_percentage, bulge_percentage],
                   'labels': [low_alpha_percentage_, high_alpha_percentage_, bulge_percentage_],
                   'legend': {'loc': 'lower center', 'bbox_to_anchor': (0.5, -0.2), 'ncol': 4},
                   'title': {'label': 'DCO distribution in MW components'},
                   'colors': ['pink', 'lightgreen', 'lightblue']
               }
           },
           columns=20, rounding_rule='nearest')

plt.tight_layout()
fnc.save_figure(pyplot_object=plt, fig_name='dco_type_MW_distribution')
plt.close()

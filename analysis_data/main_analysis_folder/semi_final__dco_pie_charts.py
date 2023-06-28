"""Created on Sat Jun 17 17:22:17 2023."""

import pandas as pd
import matplotlib.pyplot as plt
from pywaffle import Waffle

df = pd.read_csv('./../combined_dcos.csv')

bhbh = df[df['dco_type'] == 'BHBH']
bhbh.reset_index(drop=True, inplace=True)

nsns = df[df['dco_type'] == 'NSNS']
nsns.reset_index(drop=True, inplace=True)

nsbh = df[df['dco_type'] == 'NSBH']
nsbh.reset_index(drop=True, inplace=True)

bhns = df[df['dco_type'] == 'BHNS']
bhns.reset_index(drop=True, inplace=True)


def get_components(df):
    c0 = sum(df['component'] == 0)
    c1 = sum(df['component'] == 1)
    c2 = sum(df['component'] == 2)

    return (c0, c1, c2)


def get_percentages(df_, keyword):
    def __transform(num, type_):
        return f'{type_} = {num * 100:.3f}%'

    per1_ = df_[keyword].loc[0] / sum(df_[keyword])
    per2_ = df_[keyword].loc[1] / sum(df_[keyword])
    per3_ = df_[keyword].loc[2] / sum(df_[keyword])
    per4_ = df_[keyword].loc[3] / sum(df_[keyword])

    return __transform(per1_, 'BHBH'), __transform(per2_, 'NSNS'), __transform(per3_, 'NSBH'), __transform(per4_,
                                                                                                           'BHNS')


pct_label = ['low-alpha', 'high-alpha', 'bulge']

bhbh_comp = get_components(bhbh)
nsns_comp = get_components(nsns)
nsbh_comp = get_components(nsbh)
bhns_comp = get_components(bhns)

dict_ = {'dco_type': ['BHBH', 'NSNS', 'NSBH', 'BHNS'],
         'comp0': [bhbh_comp[0], nsns_comp[0], nsbh_comp[0], bhns_comp[0]],
         'comp1': [bhbh_comp[1], nsns_comp[1], nsbh_comp[1], bhns_comp[1]],
         'comp2': [bhbh_comp[2], nsns_comp[2], nsbh_comp[2], bhns_comp[2]],
         'colors': ['#ff028d', '#cea2fd', '#99ff99', 'red']
         }

df = pd.DataFrame(dict_)

total_ = sum(dict_['comp0'] + dict_['comp1'] + dict_['comp2'])

low_alpha_percentage = sum(dict_['comp0']) / total_
low_alpha_percentage_ = r'Low$-[\alpha/\mathrm{Fe}]$: ' + f'{low_alpha_percentage * 100:.2f}%'

high_alpha_percentage = sum(dict_['comp1']) / total_
high_alpha_percentage_ = r'High$-[\alpha/\mathrm{Fe}]$: ' + f'{high_alpha_percentage * 100:.2f}%'

bulge_percentage = sum(dict_['comp2']) / total_
bulge_percentage_ = 'Bulge population: ' + f'{bulge_percentage * 100:.2f}%'

plt.figure(FigureClass=Waffle, figsize=(10, 16), rows=5,
           plots={
               411: {
                   'values': list(df['comp0']),
                   'labels': get_percentages(df, 'comp0'),
                   'legend': {'loc': 'lower center', 'bbox_to_anchor': (0.5, -0.2), 'ncol': 4},
                   'title': {'label': r'Low$-[\alpha/\mathrm{Fe}]$ population'},
               },
               412: {
                   'values': list(df['comp1']),
                   'labels': get_percentages(df, 'comp1'),
                   'legend': {'loc': 'lower center', 'bbox_to_anchor': (0.5, -0.2), 'ncol': 4},
                   'title': {'label': r'High$-[\alpha/\mathrm{Fe}]$ population'}
               },
               413: {
                   'values': list(df['comp2']),
                   'labels': get_percentages(df, 'comp2'),
                   'legend': {'loc': 'lower center', 'bbox_to_anchor': (0.5, -0.2), 'ncol': 4},
                   'title': {'label': 'Bulge population'}
               }
           },
           columns=20, rounding_rule='nearest', cmap_name='Dark2')

plt.tight_layout()
plt.savefig('dco_type_MW_component_distribution.pdf')
plt.savefig('dco_type_MW_component_distribution.png')
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
plt.savefig('dco_type_MW_distribution.pdf')
plt.savefig('dco_type_MW_distribution.png')
plt.close()

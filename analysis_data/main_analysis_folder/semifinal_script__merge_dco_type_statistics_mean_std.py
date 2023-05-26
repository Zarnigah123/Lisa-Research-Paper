"""Created on Sat Apr 29 04:41:50 2023."""

import pandas as pd

mean_ = pd.read_csv('./bhbh_mean.csv')
std_ = pd.read_csv('./bhbh_std.csv')

mean2_ = mean_

def new_cols(column_name):
    if column_name == 'galaxy_number':
        mean2_[column_name] = mean_[column_name].apply(int)
    else:
        mean2_[column_name] = ('$' + f'{mean_[column_name].apply(str):.4f}' +
                               '\pm' + std_[column_name].apply(str) + '$')

keys_ = mean_.keys()

[new_cols(i) for i in keys_]

mean2_.to_csv('./bhbh_mean_std.csv', index=False)

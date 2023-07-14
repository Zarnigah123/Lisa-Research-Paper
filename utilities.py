"""Created on Sat Apr 29 04:41:50 2023."""

import pandas as pd

mean_ = pd.read_csv('./bhbh_mean.csv')
std_ = pd.read_csv('./bhbh_std.csv')

mean2_ = mean_


def new_cols(column_name):
    mean2_[column_name] = '$' + mean_[column_name].apply(str) + '\pm' + std_[column_name].apply(str) + '$'

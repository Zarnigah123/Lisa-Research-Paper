"""Created on Feb 09 20:30:10 2023."""

import h5py as h5
import numpy as np
import pandas as pd
from astropy.io.misc.hdf5 import write_table_hdf5
from astropy.table import Table


class H5Files:

    def __init__(self, file_path):
        self.file_path = file_path

    @staticmethod
    def __get_file(file_path_):
        return pd.DataFrame(h5.File(f'{file_path_}.h5', 'r')['simulation'][...])

    @property
    def get_bhbh_file(self):
        return self.__get_file(f'{self.file_path}BHBH')

    @property
    def get_nsns_file(self):
        return self.__get_file(f'{self.file_path}NSNS')

    @property
    def get_nsbh_file(self):
        return self.__get_file(f'{self.file_path}NSBH')

    @property
    def get_bhns_file(self):
        return self.__get_file(f'{self.file_path}BHNS')

    @staticmethod
    def get_min_max(df, field):
        return min(df[field]), max(df[field])


def make_h5(file_name, dataframe, order=None, keys=None, folder_name="simulation"):
    if order is None:
        new_dataframe = dataframe
        order = keys
    else:
        new_dataframe = dataframe.reindex(
            columns=order)  # taken from https://stackoverflow.com/a/47467999/3212945

    _h5 = h5.File(file_name, "w")
    write_table_hdf5(Table(data=np.stack(np.array(new_dataframe.T), axis=1), names=order), _h5,
                     folder_name)
    _h5.close()


def get_mean_std(dataframe, key_):
    group = dataframe.groupby('galaxy_number')

    mean, std = np.array(group.mean()[key_]), np.array(group.std()[key_])

    return mean, std


def simple_plot(x, axes_, label_x, data_label1, data_label2=None, y=None, log_=False):
    axes_.plot(x[0], x[1], 'ro--', alpha=0.75, label=data_label1)
    axes_.grid('on')
    axes_.legend(loc='lower right')
    axes_.set_xlabel(label_x)
    axes_.set_ylabel(r'$\sigma_\mathrm{M_1}$ [$M_\odot$]')
    if log_:
        axes_.set_xscale(log_)
    if y is not None:
        axes__ = axes_.twinx()
        axes__.plot(y[0], y[1], 'go--', alpha=0.75, label=data_label2)
        axes__.legend(loc='upper left')
        axes__.set_ylabel(r'$\sigma_\mathrm{M_2}$ [$M_\odot$]')

        if log_:
            axes__.set_xscale(log_)

        ax1_y_lim = axes_.get_ylim()
        ax12_y_lim = axes__.get_ylim()

        axes_.set_ylim(min(ax1_y_lim[0], ax12_y_lim[0]), max(ax1_y_lim[1], ax12_y_lim[1]))
        axes__.set_ylim(min(ax1_y_lim[0], ax12_y_lim[0]), max(ax1_y_lim[1], ax12_y_lim[1]))


def plot_mass(x_val, y, y_err, data_label, color, line_style, axes_, y_label=r'Mass [$M_\odot$]'):
    axes_.plot(x_val, y[0], color=color[0], marker='.', ls=line_style, label=data_label[0])
    axes_.fill_between(x_val, y[0] - y_err[0], y[0] + y_err[0], color=color[0], alpha=0.1)
    axes_.plot(x_val, y[1], color=color[1], marker='.', ls=line_style, label=data_label[1])
    axes_.fill_between(x_val, y[1] - y_err[1], y[1] + y_err[1], color=color[1], alpha=0.1)

    axes_.set_ylabel(y_label)


def plot_semimajoraxis(x, y, yerr, data_label, axes_, y_label='', color='b', line_style='-.'):
    axes_.plot(x, y, label=data_label, ls=line_style, marker='.', color=color)
    axes_.fill_between(x, y - yerr, y + yerr, alpha=0.25, color=color)

    axes_.set_ylabel(y_label)

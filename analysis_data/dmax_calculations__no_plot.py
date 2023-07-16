"""
Created on Fri Dec 31 09:29:50 2021

@author: syedalimohsinbukhari
"""

from multiprocessing import Pool

import astropy.units as u
import legwork
import numpy as np


class GetDMax:

    def __init__(self, df, n_proc=3):
        self.df = df
        self.chunk_size = n_proc
        self.n_proc = n_proc

        self.f_orb = np.logspace(np.log10(1e-5), np.log10(1), 100) * u.Hz

    def __df_to_array(self):
        return np.array(self.df[['m1_dco', 'm2_dco', 'e_dco', 'seed']])

    @staticmethod
    def __get_chunks(par):
        array, chunk_size = par
        return [array[i:i + chunk_size] for i in range(0, len(array), chunk_size)]

    def claculate_d_max(self, array):
        _len = len(self.f_orb)

        return [legwork.source.Source(m_1=[array[0]] * _len * u.M_sun,
                                       m_2=[array[1]] * _len * u.M_sun,
                                       ecc=[array[2]] * _len,
                                       dist=[1] * _len * u.kpc,
                                       f_orb=self.f_orb,
                                       interpolate_g=True,
                                       interpolate_sc=True).get_snr_evolving(t_obs=4 * u.yr) / 7, array[3]]

    def get_d_max(self):
        chopped_array = self.__get_chunks([self.__df_to_array(), self.chunk_size])
        print(f'{len(chopped_array)} arrays to process\n')
        return [(print(f'\nstarting array # {i+1}/{len(chopped_array)}\n'), Pool(self.n_proc).map(self.claculate_d_max, v))[1]
                for i, v in enumerate(chopped_array)]

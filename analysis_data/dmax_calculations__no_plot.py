"""
Created on Fri Dec 31 09:29:50 2021

@author: syedalimohsinbukhari
"""

from multiprocessing import Pool

import astropy.units as u
import legwork
import numpy as np


class GetDMax:

    def __init__(self, df, n_proc=3, chunk_size=10):
        self.df = df
        self.chunk_size = chunk_size
        self.n_proc = n_proc

        self.f_orb = np.logspace(np.log10(1e-5), np.log10(1), 100) * u.Hz

    def __df_to_array(self, gal_number):
        _temp = self.df[self.df.galaxy_number.isin(gal_number)]
        _temp = _temp[['m1_dco', 'm2_dco', 'e_dco', 'seed']]
        print('')
        return np.array(_temp)

    @staticmethod
    def __get_chunks(par):
        array, chunk_size = par
        return [array[i:i + chunk_size] for i in range(0, len(array), chunk_size)]

    def _get_d_max(self, array):
        _len = len(self.f_orb)
        return [[legwork.source.Source(m_1=[i[0]] * _len * u.M_sun,
                                       m_2=[i[1]] * _len * u.M_sun,
                                       ecc=[i[2]] * _len,
                                       dist=[1] * _len * u.kpc,
                                       f_orb=self.f_orb,
                                       interpolate_g=True,
                                       interpolate_sc=True).get_snr_evolving(t_obs=4 * u.yr) / 7, i[3]] for i in array]

    def get_d_max(self, gal_number):
        chopped_array = self.__get_chunks([self.__df_to_array(gal_number), self.chunk_size])
        print(f'{len(chopped_array)} arrays to process\n')
        _out = [j for i in
                [(print(f'starting array # {i}'), self._get_d_max(v))[1] for i, v in enumerate(chopped_array)]
                for j in i]

        return _out

    def run(self, gal_number):
        return Pool(self.n_proc).map(GetDMax(self.df).get_d_max, gal_number)

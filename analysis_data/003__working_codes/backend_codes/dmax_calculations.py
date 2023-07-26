"""
Created on Fri Dec 31 09:29:50 2021

@author: syedalimohsinbukhari
"""

from multiprocessing import Pool

import astropy.units as u
import legwork
import numpy as np


class GetDMax:
    """
    Calculate d_max values using legwork.source.Source for a given DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing columns 'm1_dco', 'm2_dco', 'e_dco', and 'seed'.
        n_proc (int, optional): Number of processes for multiprocessing. Default is 3.

    Attributes:
        df (pd.DataFrame): The DataFrame containing columns 'm1_dco', 'm2_dco', 'e_dco', and 'seed'.
        chunk_size (int): The number of rows in each chunk for multiprocessing.
        n_proc (int): Number of processes for multiprocessing.
        f_orb (Quantity): Array of orbital frequencies in Hz.

    """

    def __init__(self, df, n_proc=3):
        self.df = df
        self.chunk_size = n_proc
        self.n_proc = n_proc

        self.f_orb = np.logspace(np.log10(1e-5), np.log10(1), 100) * u.Hz

    def _df_to_array(self):
        """
        Convert DataFrame to a numpy array.

        Returns:
            np.ndarray: Numpy array containing columns 'm1_dco', 'm2_dco', 'e_dco', and 'seed'.
        """

        return np.array(self.df[['m1_dco', 'm2_dco', 'e_dco', 'seed']])

    def calculate_d_max(self, array: np.ndarray):
        """
        Calculate d_max for a chunk of the array using legwork.source.Source.

        Args:
            array (np.ndarray): A chunk of the array containing 'm1_dco', 'm2_dco', 'e_dco', and 'seed' values.

        Returns:
            Tuple[np.ndarray, int]: A tuple containing d_max values and the corresponding seed.
        """

        _len = len(self.f_orb)

        return [legwork.source.Source(m_1=[array[0]] * _len * u.M_sun,
                                      m_2=[array[1]] * _len * u.M_sun,
                                      ecc=[array[2]] * _len,
                                      dist=[1] * _len * u.kpc,
                                      f_orb=self.f_orb,
                                      interpolate_g=True,
                                      interpolate_sc=True).get_snr_evolving(t_obs=4 * u.yr) / 7, array[3]]

    @staticmethod
    def _get_chunks(par):
        """
        Split an array into chunks.

        Args:
            par (Tuple[np.ndarray, int]): A tuple containing the array to be split and chunk size.

        Returns:
            List[np.ndarray]: List of chunks from the input array.
        """
        array, chunk_size = par
        return [array[i:i + chunk_size] for i in range(0, len(array), chunk_size)]

    def get_d_max(self):
        """
        Calculate d_max values using multiprocessing.

        Returns:
            List[Tuple[np.ndarray, int]]: A list of tuples containing d_max values and the corresponding seed.
        """
        chopped_array = self._get_chunks([self._df_to_array(), self.chunk_size])
        print(f'{len(chopped_array)} arrays to process\n')
        return [(print(f'\nstarting array # {i + 1}/{len(chopped_array)}\n'),
                 Pool(self.n_proc).map(self.calculate_d_max, v))[1]
                for i, v in enumerate(chopped_array)]

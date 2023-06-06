#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 04:03:06 2022

@author: nazeela
"""

import os
import h5py
import astropy.units as u
import legwork as lw

cur_ = os.getcwd()
path_ = 'BHBH/'

full = f'{cur_}/{path_}'

bhbh = [f for f in os.listdir(full) if f.endswith('.h5')]

for v, i in enumerate(bhbh):
    print(f'file {v}')
    h5file = h5py.File(f'{full}{i}')['simulation']
    
    seed = h5file['seed']
    m1_dco = h5file['m1_dco']*u.Msun
    m2_dco = h5file['m2_dco']*u.Msun
    a_dco = h5file['a_dco']*u.AU
    e_dco = h5file['e_dco']
    t_evolution = h5file['t_evolution__Myr']*u.Myr
    t_merge_ = h5file['t_merge__Myr']*u.Myr
    t_lookback = h5file['t_lookback__Gyr']*u.Myr
    distance = h5file['distance__kpc']*u.kpc
    component = h5file['component']
    
    t_merge = lw.evol.get_t_merge_ecc(ecc_i=e_dco, a_i=a_dco, m_1=m1_dco, m_2=m2_dco)
    insp = t_merge > (t_lookback - t_evolution)
    
    m1 = m1_dco[insp]
    m2 = m2_dco[insp]
    a = a_dco[insp]
    e = e_dco[insp]
    tevol = t_evolution[insp]
    tlb = t_lookback[insp]
    tmrg = t_merge[insp]
    dist = distance[insp]
    comp = component[insp]
    
    e_lisa, a_lisa, f_lisa = lw.evol.evol_ecc(ecc_i=e, a_i=a, m_1=m1, m_2=m2,
                                                  t_evol=tlb - tevol,
                                                  output_vars=["ecc", "a", "f_orb"])
    
    e_lisa = e_lisa[:, -1]
    a_lisa = a_lisa[:, -1]
    f_lisa = f_lisa[:, -1]
    
    sources = lw.source.Source(m_1=m1, m_2=m2, ecc=e_lisa, dist=dist, f_orb=f_lisa)
    snr = sources.get_snr(t_obs=4*u.yr)
    
    print(f'detectable = {len(snr[snr>7])}')

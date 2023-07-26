"""
Created on Mar 02 10:57:58 2022
"""

import os
from multiprocessing import Pool

import numpy as np
from analysis_data.working_codes.backend_codes.utilities import assign_mw_coordinates, chunks

cur_ = f'{os.getcwd()}/'

get_dir = '/media/nazeela/New Volume/galaxies/'
sim_MW = [f for f in os.listdir(get_dir) if f.endswith('.npy') and f.startswith('simulated_MW')]
n_gal = range(1, len(sim_MW) + 1)

dco_type = 'NSBH'

DCO_files = np.repeat(f'{dco_type}@DCO.h5', len(sim_MW))

_f = [tuple([i, f'{get_dir}{j}', int(j.split('_')[-1].split('.')[0]), cur_, f'./{dco_type}/']) for i, j, k in
      zip(DCO_files, sim_MW, n_gal)]

chunks_ = list(chunks(_f, 4))

p = [assign_mw_coordinates] * len(chunks_[0])

p *= len(chunks_)

[Pool(4).map(i, chunk) for i, chunk in zip(p, chunks_)]

"""
Created on Wed Feb 15 13:23:26 2023

@author: nazeela
"""

import os
import numpy as np
import h5py as h5

dir_ = '/home/nazeela/NewCodesAliMohsin/data/newbhbh'

f = [f for f in os.listdir(dir_) if f.endswith('.h5')]
f.sort()

print(f)

_h5 = h5.File(f'{dir_}/{f[0]}', 'r')['simulation'][...]

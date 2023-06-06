"""Created on Mon Jun  5 00:29:59 2023."""

import os
import h5py
import pandas as pd
from utilities import make_h5

f = [f for f in os.listdir(os.curdir) if f.endswith('.h5')]
f.sort()

for i in f:
    p = pd.DataFrame(h5py.File(i, 'r')['simulation'][...])
    p = p.rename(columns={'distance': 'weight1',
                          'R': 'distance1',
                          'z': 'R1',
                          'theta': 'z1',
                          'weight': 'theta1'})
    p = p.rename(columns={'distance1': 'distance',
                          'R1': 'R',
                          'z1': 'z',
                          'theta1': 'theta',
                          'weight1': 'weight'})

    make_h5(i, p, p.keys())

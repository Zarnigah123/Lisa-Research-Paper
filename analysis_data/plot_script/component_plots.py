#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 16:56:27 2023

@author: nazeela
"""

import h5py as h5

files = ['./../BHBH.h5']
# ,
#          './../NSNS.h5',
#          './../NSBH.h5',
#          './../BHNS.h5']

_h5 = h5.File(files[0])['simulation'][...].squeeze()

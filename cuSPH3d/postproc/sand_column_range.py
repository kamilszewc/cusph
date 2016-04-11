#!/usr/bin/env python
from sys import argv
import bz2
from numpy import *

from postproc.data import data

#mlab.options.offscreen = True

cx = 20.0
cy = 20.0

x, y, z, u, v, w, p = [], [], [], [], [], [], []
for arg in argv[1:]:
    f = file(arg)
    filelines = f.readlines()
    f.close()

    i = 0
    for fileline in filelines:
        if i%1 == 0:
            x.append(float(fileline.split()[0]))
            y.append(float(fileline.split()[1]))
            z.append(float(fileline.split()[2]))
            u.append(float(fileline.split()[3]))
            v.append(float(fileline.split()[4]))
            w.append(float(fileline.split()[5]))
            try:
                p.append(float(fileline.split()[-1]))
            except:
                pass
        i += 1

s = []
for ix, iy in zip(x,y):
    s.append( sqrt( (ix-cx)**2 + (iy-cy)**2 ) )
    
s.sort()

print s[-1]

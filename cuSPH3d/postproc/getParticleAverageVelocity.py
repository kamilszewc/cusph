#!/usr/bin/env python
from sys import argv
from glob import glob
from math import *
from postproc.data import data

arglist = []
for args in argv[1:]:
    for arg in glob(args):
        arglist.append(arg)
        
wl = []
tl = []
n = 0
for arg in arglist:
    d = data(arg)
    n += 1
    for vel,dens in zip(d.w,d.d):
        if dens > 1.5:
            wl.append(vel)
            tl.append(d.t)

vel = 0.0
stddev = 0.0
for w in wl:
    vel += w
vel = vel/n

for w in wl:
    stddev += (vel - w)**2
stddev = sqrt(stddev/n)
            
print vel, stddev
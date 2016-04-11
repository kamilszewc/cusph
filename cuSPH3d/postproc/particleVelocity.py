#!/usr/bin/env python
from sys import argv
from glob import glob
from matplotlib.pyplot import *
from postproc.data import data

arglist = []
for args in argv[1:]:
    for arg in glob(args):
        arglist.append(arg)
        
w = []
t = []
for arg in arglist:
    d = data(arg)
    for vel,dens in zip(d.w,d.d):
        if dens > 1.5:
            w.append(vel)
            t.append(d.t)
            
figure()
plot(t, w, '-k')
xlabel("Time")
ylabel("Velocity")
show()
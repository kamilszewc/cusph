#!/usr/bin/env python
from sys import argv
from glob import glob
from postproc.data import data

arglist = []
for args in argv[1:]:
    for arg in glob(args):
        arglist.append(arg)
        
for arg in arglist:
    d = data(arg)
    for m,d in zip(d.m,d.d):
        if d > 1.5:
            print arg, d, m
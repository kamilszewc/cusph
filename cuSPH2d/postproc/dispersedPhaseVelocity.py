#!/usr/bin/env python
from sys import argv,exit
from numpy import *
import numpy
from pylab import *
from postproc.Data import Data


for arg in argv[1:]:
    d = Data(arg)
    
    U, V = 0.0, 0.0
    N = 0
    for x, y, u, v in zip(d.pdpfX, d.pdpfY, d.pdpfU, d.pdpfV):
        if 0.7 < x < 1.3:
            U += u
            V += v
            N += 1

    U = U/N
    V = V/N

    print(d.t, U, V)

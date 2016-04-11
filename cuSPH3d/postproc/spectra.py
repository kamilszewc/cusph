#!/usr/bin/env python
# -*- coding: utf-8 -*-
# All right reserved by Kamil Szewc, Gdansk 2015

from sys import argv
import time
import numpy as np
from matplotlib.pyplot import *
from postproc.data import data
from postproc.field import field

NX, NY, NZ = 41, 41, 41

time1 = time.time()
d = data(argv[1], ("velocity","density", "mass"))
time2 = time.time()
f = field(d, NX,NY,NZ)
time3 = time.time()


e = np.zeros( (NX,NY,NZ) )
e = 0.5*(f.u**2 + f.v**2 + f.w**2)

fourier = np.fft.fftn(e)

fourier = np.abs(np.fft.fftshift(fourier))

N = NX/2 + 1
fourier1dY = np.zeros(N)
fourier1dX = np.array(range(1,N+1))
fourier1dN = np.zeros(N)
cx, cy, cz = NX/2+1, NY/2+1, NZ/2+1

for i in range(NX):
    for j in range(NY):
        for k in range(NZ):
            x = int(np.sqrt( (i-cx)**2 + (j-cy)**2 + (k-cz)**2 ))
            if x < N:
                fourier1dY[x] += fourier[i,j,k]
                fourier1dN[x] += 1
            
for i in range(len(fourier1dY)):
    if fourier1dN[i] != 0:
        fourier1dY[i] = fourier1dY[i] / fourier1dN[i]

time4 = time.time()

print "Reading data: %s" % (time2-time1)
print "Projecting: %s" % (time3-time2)
print "Fouriering: %s" % (time4-time3)
print "Total: %s" % (time4-time1)

reference53 = [60.0*i**(-5./3.) for i in fourier1dX]

figure()
loglog(fourier1dX, fourier1dY, '-', linewidth=2)
loglog(fourier1dX, reference53, '-')
show()

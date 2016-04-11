#!/usr/bin/env python
# -*- coding: utf-8 -*-
# All right reserved by Kamil Szewc, Gdansk/Nancy 2009 

from sys import argv,exit
from numpy import *
import numpy
from pylab import *
from postproc.Data import Data
from postproc.Field import Field
from postproc.Point import Point
import matplotlib

if __name__ == '__main__':
    figure(figsize=(9,7))
    ticktype=['ko', 'kd', 'k^']

    n = 0
    for arg in argv[1:]:
        d = Data(arg)

        phi = linspace(0.0, 2.0*3.1415, 64)
        xi = [0.0025*cos(p)+0.005 for p in phi]
        yi = [0.0025*sin(p)+0.005 for p in phi]
    
        concentration = []
        for x, y in zip(xi, yi):
             p = Point(x, y, d)
             concentration.append(p.cSurf)
         
        concentrationAnal = [3.0e-6 * 0.5 * ( exp(-1.0e-4 * d.t / 0.0025**2) * cos(ph) + 1.0) for ph in phi]

        if n==0: plot(phi, array(concentrationAnal)/3.0e-6, '-k', label="Analytical")
        else: plot(phi, array(concentrationAnal)/3.0e-6, '-k')

        plot(phi, array(concentration)/3.0e-6, ticktype[n], label="t/T=%s" % (d.t * 1.0e-4 / 0.0025**2))

        n += 1
    
    xlabel("$\phi$", fontsize='x-large')
    ylabel("$C(\phi,t)/ max C(\phi,0)$", fontsize='x-large')
    legend(loc=9)
    grid(True)
    xlim(0.0, 2.0*3.1415)
    xticks([0, 0.5*pi, pi, 3.0*pi/2.0, 2*pi], ['$0$', r"$\frac{1}{2} \pi$", '$\pi$', r"$\frac{3}{2} \pi$",'$2\pi$'])

    show()

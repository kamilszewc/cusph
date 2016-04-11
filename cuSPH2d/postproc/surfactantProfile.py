#!/usr/bin/env python
# -*- coding: utf-8 -*-
# All right reserved by Kamil Szewc, Gdansk 2009 

from math import *
from sys import argv,exit
import os
import numpy as np
from matplotlib.pyplot import *
from matplotlib.ticker import *
from matplotlib import rc
from postproc.Data import Data
from postproc.Field import XField, YField

def concentrationProfile(x,t):
    if x < 0.05:
        return 0.5 * ( 1.0 + erf( 0.5*(x-0.05) / sqrt(4.0e-6 * t) ) )
    else:
        return 0.5 * erfc( 0.5*(0.05-x) / sqrt(4.0e-6 * t) ) 

if __name__ == '__main__':
    style = ('ko', 'kd', 'k^', 'mo', 'yo')
	
    fig1 = figure(1)
    pic1 = fig1.add_subplot(111)

    n = 0
    for arg in argv[1:]:
        d = Data(arg)

        NX = 10000
		
        lx = d.XCV/float(d.NX)
		
        x = np.linspace(lx*0.5, lx*0.5+lx*float(d.NX-1), d.NX)
        y = 0.05

        rx = XField(x, y, d)
        x = np.array(rx.x)
        p = np.array(rx.cBulk)

        lxa = d.XCV/float(NX)
        xa = np.linspace(lxa*0.5, lxa*0.5+lxa*float(NX-1), NX)
        pa = [ concentrationProfile(xx,d.t) for xx in xa ]

        if n ==0: pic1.plot(xa / 0.1, pa, '-', label="Analytical", linewidth=1.6, color='gray')
        else: pic1.plot(xa / 0.1, pa, '-', linewidth=1.6, color='gray')

        pic1.plot(x / 0.1, p, style[n], label=r"$t=%s$" % (d.t,), markersize=5.0)

        n += 1

      #  for xx, pp in zip(x, p):
      #      print xx, pp
        

    pic1.set_xlim(0.3, 0.7)
    pic1.set_ylim(0.0, 1.0)
    pic1.legend(ncol=1, fancybox=True)
    pic1.set_ylabel(r"$C(x,t) / \max{C(x,t=0)}$", fontsize='x-large')
    pic1.set_xlabel(r"$x/L$", fontsize='x-large')

    fig1.savefig("surfDistrStep.pdf", format="pdf")
    fig1.savefig("surfDistrStep.eps", format="eps")
    show()
		

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# All right reserved by Kamil Szewc, Gdansk/Nancy 2009 

from sys import argv,exit
from numpy import *
import numpy
from pylab import *
from postproc.Data import Data
from postproc.Field import Field

if __name__ == '__main__':

    d = Data(argv[1])

    try:
        x1, y1 = [], []
        x2, y2 = [], [] 
        x3, y3 = [], []
        u1, v1 = [], []
        u2, v2 = [], []
        u3, v3 = [], []
        for i in range(d.N):
            if (d.phaseId[i] == 0):
                x1.append(d.x[i])
                y1.append(d.y[i])
                u1.append(d.u[i])
                v1.append(d.v[i])
            if (d.phaseId[i] == 1):
                x2.append(d.x[i])
                y2.append(d.y[i])
                u2.append(d.u[i])
                v2.append(d.v[i])
            if (d.phaseId[i] == 2):
                x3.append(d.x[i])
                y3.append(d.y[i])
                u3.append(d.u[i])
                v3.append(d.v[i])
    except: pass

    extent = (0,d.XCV,0,d.YCV)
    figsize = (6, 6*d.YCV/d.XCV)

    figure(figsize=array(figsize)*1.8)
    xlim(0,d.XCV)
    ylim(0,d.YCV)
    plot(x1, y1, 'co', markersize=2.1, markeredgecolor='c')
    plot(x2, y2, 'ko', markersize=2.1, markeredgecolor='k') 
    plot(x3, y3, 'ro', markersize=2.1, markeredgecolor='r')
    xlabel('x')
    ylabel('y')
    if d.T_DISPERSED_PHASE_FLUID > 0:
        plot(d.pdpfX, d.pdpfY, 'yo', markersize=2.1, markeredgecolor='y')
    tight_layout()

    show()

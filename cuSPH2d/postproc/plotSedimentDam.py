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

    figsize = (14, 1.5)

    if d.T_SOIL == 2:
        
        for i in range(d.N):
            if 0.01 < d.cs[i] < 0.99:
                d.phaseId[i] = 2
                
        figure(figsize=figsize)
        scatter(d.x, d.y, c=d.phaseId, s=6, linewidth=0)
        title("Sediment")
        xlim(0.4, d.XCV)
        ylim(0.0, 0.15)
        xlabel('x')
        ylabel('y')
        colorbar()


    show()

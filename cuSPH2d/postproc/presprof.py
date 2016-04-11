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

if __name__ == '__main__':
    style = ('k--', 'k-.', 'co', 'mo', 'yo')
    label = ('CSF', 'SSF')
	
    fig1 = figure(1)
    pic1 = fig1.add_subplot(111)

    n = 0
    for arg in argv[1:]:
        d = Data(arg)
		
        lx = d.XCV/float(d.NX)
		
        x = np.linspace(lx*0.5, lx*0.5+lx*float(d.NX-1), d.NX)
        y = 0.5

        rx = XField(x, y, d)
        x = np.array(rx.x)
        p = np.array(rx.p)

        pic1.plot(x, p-p[0], style[n], label=label[n], linewidth=2)
        n += 1
        
    xp = np.linspace(0, 1, 1000)
    pp = []
    for po in xp:
        if 0.5-0.33851375012865381 < po < 0.5+0.33851375012865381:
            pp.append(2.9540897515091933)
        else:
            pp.append(0.0)
		
    t = float(os.path.splitext(os.path.split(argv[1])[-1])[0])
		
    pic1.set_xlabel(r"$x$", size='xx-large')
    pic1.set_ylabel(r"$p-p_{ref}$", size='xx-large')
    pic1.plot(xp, pp, 'k-', label='Analytical')
    pic1.legend(loc=8)
    pic1.set_ylim(-0.5, 3.5)

    fig1.savefig('presprof.pdf', format='pdf')

    show()
		

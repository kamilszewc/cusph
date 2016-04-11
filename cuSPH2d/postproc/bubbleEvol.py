#!/usr/bin/env python
# -*- coding: utf-8 -*-
# All right reserved by Kamil Szewc, Gdansk/Nancy 2009 

from sys import argv,exit
from numpy import *
from pylab import *
from postproc.Data import Data
from postproc.Field import Field

def readdata(filename):
    f = open(filename, 'r')
    datalines = f.readlines()
    f.close()
    x, y = [], []
    for dataline in datalines:
        x.append(7.0*float(dataline.split()[0]))
        y.append(7.0*float(dataline.split()[1]))
    return x,y 

if __name__ == '__main__':
    d = Data(argv[1])

   # r = Field(d)

    extent = (0,d.XCV,0,d.XCV)

    figure(figsize=(d.XCV, d.XCV))
    x1, y1, x2, y2 = [], [], [], []
    for i in range(d.N):
        if d.phaseId[i] == 1:
            x1.append(d.x[i])
            y1.append(d.y[i])
        else:
            x2.append(d.x[i])
            y2.append(d.y[i])
    plot(x1, y1, marker='o', color='0.5', markersize=2.2, linewidth=0, markeredgecolor='0.5')
    plot(x2, y2, marker='o', color='0.8', markersize=2.2, linewidth=0, markeredgecolor='0.8')
    if (argv[-1] == '2.8'):
        x, y = readdata("postproc/resource/sus_re1000_bo200_28.dat")
	for i in range(len(x)):
            x[i] = x[i] - .45
            y[i] = y[i] + .4
	plot(x, y, 'dk', markersize=5)
	xlim(1.0, 5.0)
        ylim(2.0, 6.0)
        text(4.2, 5.6, "t=%s" % d.t, fontsize='large')
    if (argv[-1] == '4.0'):
        x, y = readdata('postproc/resource/sus_re1000_bo200_40.dat')
        for i in range(len(x)):
            x[i] = x[i] - .45
            y[i] = y[i] + .3
        plot(x, y, 'dk', markersize=5)
        xlim(1.0, 5.0)
        ylim(2.0, 6.0)
        text(4.2, 5.6, "t=%s" % d.t, fontsize='large')
    if (argv[-1] == '4.8'): 
        x, y = readdata('postproc/resource/sus_re1000_bo200_48.dat')
        for i in range(len(x)):
            x[i] = x[i] - .45
            y[i] = y[i] + .25
        plot(x, y, 'dk', markersize=5)
        xlim(1.0, 5.0)
        ylim(2.0, 6.0)
        text(4.2, 5.6, "t=%s" % d.t, fontsize='large')
    if (argv[-1] == '6.0'): 
        x, y = readdata('postproc/resource/sus_re1000_bo200_60.dat')
        for i in range(len(x)):
            x[i] = x[i] - .45
            y[i] = y[i] + .15
        plot(x, y, 'dk', markersize=5)
        xlim(1.0, 5.0)
        ylim(2.0, 6.0)
        text(4.2, 5.6, "t=%s" % d.t, fontsize='large')
    #imshow(r.d, cmap=cm.autumn, origin='lower', extent=extent)
    #imshow(r[3], cmap=cm.spring, origin='lower', extent=extent)
    #imshow(r[3], cmap=cm.summer, origin='lower', extent=extent)
    #imshow(r[3], cmap=cm.winter, origin='lower', extent=extent)
    #imshow(r[3], cmap=cm.spectral, origin='lower', extent=extent)
    #colorbar()
    #title("%d x %d particles" % (d.NX, d.NY))
    #title("Re=1000, Bo=200")
    

    #figure(figsize=(d.XCV,d.YCV))
    xlabel("x/L", fontsize='large')
    ylabel("y/L", fontsize='large')

    #title(r"CSF-SST-B, $\varepsilon=0.02$, $\sigma=0.02$", fontsize='x-large')

    tight_layout()

    savefig("bubble-5-48.eps", format='eps')
    savefig("bubble-5-48.pdf", format='pdf')
    savefig("bubble-5-48.png", dpi=600, format='png')

    show()

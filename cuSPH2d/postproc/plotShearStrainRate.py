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

    figure(figsize=(7,6))
    x1, y1, x2, y2 = [], [], [], []
    for i in range(d.N):
        if d.phaseId[i] == 1:
            x1.append(d.x[i])
            y1.append(d.y[i])
        else:
            x2.append(d.x[i])
            y2.append(d.y[i])
    #plot(x1, y1, marker='o', color='0.5', markersize=2.2, linewidth=0, markeredgecolor='0.5')
    #plot(x2, y2, marker='o', color='0.8', markersize=2.2, linewidth=0, markeredgecolor='0.8')
    scatter(d.x, d.y, c=d.str, s=6, linewidth=0)
    colorbar()
    
    if (argv[-1] == '0.2'):
        alpha = linspace(-0.5*pi,0.5*pi, 10)
        x = [cos(a)+3.0 for a in alpha]
        y = [sin(a)+2.0 for a in alpha]
	plot(x, y, 'd', color='black', markersize=7)
        text(4.2, 5.6, "t=%s" % d.t, fontsize='large')
    if (argv[-1] == '0.5'):
        alpha = linspace(-0.5*pi,0.5*pi, 10)
        x = [cos(a)+3.0 for a in alpha]
        y = [sin(a)+2.1 for a in alpha]
	plot(x, y, 'd', color='black', markersize=7)
        text(4.2, 5.6, "t=%s" % d.t, fontsize='large')
    if (argv[-1] == '2.8'):
        x, y = readdata("postproc/resource/sus_re1000_bo200_28.dat")
	for i in range(len(x)):
            x[i] = x[i] - .45
            y[i] = y[i] + .4
	plot(x, y, 'd', color='black', markersize=7)
        text(4.2, 5.6, "t=%s" % d.t, fontsize='large')
    if (argv[-1] == '4.0'):
        x, y = readdata('postproc/resource/sus_re1000_bo200_40.dat')
        for i in range(len(x)):
            x[i] = x[i] - .45
            y[i] = y[i] + .3
        plot(x, y, 'd', color='black', markersize=7)
        text(4.2, 5.6, "t=%s" % d.t, fontsize='large')
    if (argv[-1] == '4.8'): 
        x, y = readdata('postproc/resource/sus_re1000_bo200_48.dat')
        for i in range(len(x)):
            x[i] = x[i] - .45
            y[i] = y[i] + .25
        plot(x, y, 'd', color='black', markersize=7)
        text(4.2, 5.6, "t=%s" % d.t, fontsize='large')
    if (argv[-1] == '6.0'): 
        x, y = readdata('postproc/resource/sus_re1000_bo200_60.dat')
        for i in range(len(x)):
            x[i] = x[i] - .45
            y[i] = y[i] + .15
        plot(x, y, 'd', color='black', markersize=7)
        text(4.2, 5.6, "t=%s" % d.t, fontsize='large')
    
    xlim(1.0, 5.0)
    ylim(1.0, 5.0)
    
    xlabel("x/L", fontsize='large')
    ylabel("y/L", fontsize='large')
    
    text(4.2, 4.6, "t=%s" % (d.t), fontsize='x-large', bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})

    #title(r"CSF-SST-B, $\varepsilon=0.02$, $\sigma=0.02$", fontsize='x-large')

    tight_layout()

    savefig("shear-strain-rate-1-t02.eps", format='eps')
    savefig("shear-strain-rate-1-t02.pdf", format='pdf')
    savefig("shear-strain-rate-1-t02.png", dpi=600, format='png')

    show()
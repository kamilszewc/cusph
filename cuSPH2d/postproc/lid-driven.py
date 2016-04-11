#!/usr/bin/env python
# -*- coding: utf-8 -*-
# All right reserved by Kamil Szewc, Gdansk 2009 

from math import *
from sys import argv,exit
from numpy import *
from matplotlib.pyplot import *
from matplotlib.ticker import *
from matplotlib import rc
from postproc.Data import Data
from postproc.Field import XField, YField

tx     = [0.0000, 0.0547, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531, 0.5000, 0.6172, 0.7344, 0.8516, 0.9531, 0.9688, 1.0000]
tx100  = [0.0000,-0.0372,-0.0478,-0.0643,-0.1015,-0.1566,-0.2109,-0.2058,-0.1364, 0.0033, 0.2315, 0.6872, 0.7887, 1.0000]
tx1000 = [0.0000,-0.1811,-0.2222,-0.2973,-0.3829,-0.2781,-0.1065,-0.0608, 0.0570, 0.1872, 0.3330, 0.4660, 0.5749, 1.0000]
ty     = [0.0000, 0.0625, 0.0938, 0.1563, 0.2344, 0.5000, 0.8047, 0.8594, 0.9063, 0.9453, 0.9688, 1.0000]
ty100  = [0.0000, 0.0923, 0.1232, 0.1608, 0.1753, 0.0545,-0.2453,-0.2245,-0.1691,-0.1031,-0.0591, 0.0000]
ty1000 = [0.0000, 0.2749, 0.3263, 0.3710, 0.3224, 0.0253,-0.3197,-0.4267,-0.5155,-0.3919,-0.2139, 0.0000]

if __name__ == '__main__':
    style = ('ro', 'go', 'co', 'mo', 'yo')
	
    fig1 = figure(1)
    fig2 = figure(2)
    pic1 = fig1.add_subplot(111)
    pic2 = fig2.add_subplot(111)
	
    d = Data(argv[1])
		
    lx = d.XCV/float(d.NX)
    ly = d.YCV/float(d.NY)

    x = 0.5
    y = linspace(ly*0.5, ly*0.5+ly*float(d.NY-1), d.NY)

    ry = YField(x, y, d)
		
    x = linspace(lx*0.5, lx*0.5+lx*float(d.NX-1), d.NX)
    y = 0.5

    rx = XField(x, y, d)
		
    t = d.t
		
    pic1.set_xlabel(r"x", size='xx-large')
    pic1.set_ylabel(r"u_y", size='xx-large')
    pic2.set_xlabel(r"y", size='xx-large')
    pic2.set_ylabel(r"ux", size='xx-large')
    pic1.plot(rx.x, rx.v, style[0])
    pic2.plot(ry.y, ry.u, style[0])

    if len(argv) == 3 and argv[2] == '100':
        pic1.plot(ty, ty100, style[1], marker='s')
        pic2.plot(tx, tx100, style[1], marker='s')

    if len(argv) == 3 and argv[2] == '1000':
        pic1.plot(ty, ty1000, style[1], marker='s')
        pic2.plot(tx, tx1000, style[1], marker='s')
    show()
		

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# All right reserved by Kamil Szewc, Gdansk 2009 

from math import *
from sys import argv,exit
from glob import glob
from numpy import *
from matplotlib.pyplot import *
from postproc.Data import Data

if __name__ == '__main__':
    list = []
    for args in argv[1:]:
        for arg in glob(args):
            list.append(arg)

    print("#time x-center y-center x-average y-average x-max y-max x-vel y-vel")

    rxa, rya, rxmaxa, rymaxa, rua, rva, ta = [], [], [], [], [], [], []
    for arg in list:
        d = Data(arg)

        rx, ry, rxmax, rymax, rcx, rcy, ru, rv, n, m = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0
        for i in range(d.N):
            if d.c > 0.5:
                rcx += d.x[i]
                rcy += d.y[i]
                m += 1
        rcx = rcx / m
        rcy = rcy / m

        for i in range(d.N):
            if d.x[i] > rcx and d.y[i] > rcy and d.c[i] > 0.5:
                rx += d.x[i]
                ry += d.y[i]
                ru += d.u[i]
                rv += d.v[i]
                n += 1
            if d.x[i] > rxmax and d.c[i] > 0.5:
                rxmax = d.x[i]
            if d.y[i] > rymax and d.c[i] > 0.5:
                rymax = d.y[i]
        rx = -rcx + rx/float(n)
        ry = -rcy + ry/float(n)
        rxmax = -rcx + rxmax
        rymax = -rcy + rymax
        ru = ru/float(n)
        rv = rv/float(n)

        ta.append(d.t); rxa.append(rx); rya.append(ry); rua.append(ru); rva.append(rv)
        rxmaxa.append(rxmax)
        rymaxa.append(rymax)
        print(d.t, rcx/d.XCV, rcy/d.XCV, rx/d.XCV, ry/d.XCV, rxmax/d.XCV, rymax/d.XCV)

    fig1 = figure(1)
    plt1 = fig1.add_subplot(111)
    plt1.plot(ta, rxa, 'o')
    plt1.plot(ta, rya, 'o')
    plt1.set_title("Average position")

    fig2 = figure(2)
    plt2 = fig2.add_subplot(111)
    plt2.plot(ta, rxmaxa, 'o')
    plt2.plot(ta, rymaxa, 'o')
    plt2.set_title("Position")

    fig3 = figure(3)
    plt3 = fig3.add_subplot(111)
    plt3.plot(ta, rua, 'o')
    plt3.plot(ta, rva, 'o')
    plt3.set_title("Velocity")

    show()

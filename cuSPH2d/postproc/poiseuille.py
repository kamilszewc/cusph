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
from postproc.Field import YField

def poiseuille(y,t,nu,l,f,N):
    u = f*y*(l-y)/(2.0*nu)
    for n in range(N):
        u -= 4.0*f*l*l*sin(pi*y*(2*n+1)/l)*exp(-(2*n+1)*(2*n+1)*pi*pi*nu*t/(l*l))/(nu*pi*pi*pi*pow(2*n+1,3))
    return u

def show_figure(args, type_of_output, title):
#    rc('text' , usetex = True)
#    rc('text.latex', unicode = True)
    style = ('ro', 'go', 'co', 'mo', 'yo')
    label = ('WCSPH', 'ISPH')

    d=[]
    r=[]
    u0 = []
    time = ""
    t = 0.0

    fig = figure(1)
    pic1 = fig.add_subplot(211)
    pic2 = fig.add_subplot(212)
    

    for n in range(len(args)-1):
        d = Data(args[n+1])

        lx = d.XCV/float(d.NX)
        ly = d.YCV/float(d.NY)

        x = 0.4
        y = linspace(ly*0.5, ly*0.5+ly*float(d.NY-1), d.NY)

        r = YField(x, y, d)
        du = []

        if n==0:
            t = d.t

            y_a  = linspace(0.0, 1.0, 150)
            u_a  = poiseuille(y_a,t,d.nu[0],1.0,d.G_X,100)

        for i in range(len(r.y)):
            du.append(abs(r.u[i]-poiseuille(r.y[i],t,d.nu[0],1.0,d.G_X,100)))
        
        if n == 0:            
            pic1.plot(y_a, u_a, linewidth=2, label="exact")

        pic1.plot(r.y, r.u, style[n], label=label[n])
        pic2.plot(r.y, du, style[n])

    pic1.set_title(r"$t = %f$" % (t), size='x-large')
    #pic2.set_ylim(0, 0.005)
    #pic1.set_ylim(0.0, 0.14)
    pic1.set_ylabel(r"$u_x$", size='xx-large')
    pic2.set_ylabel(r"$|u_x-u^{ex}_x|$", size='xx-large')
    pic2.set_xlabel(r"$y/L$", size='xx-large')
    pic1.legend(ncol=1, fancybox=True)#, loc=8)
    setp(pic1.get_xticklabels(), visible=False)
    setp(pic1.get_xticklabels(), fontsize='large')
    setp(pic2.get_xticklabels(), fontsize='large')
    setp(pic1.get_yticklabels(), fontsize='large')
    setp(pic2.get_yticklabels(), fontsize='large')

    if type_of_output == 1:
        filename = str('%03d' % title) + '.png'
        fig.savefig(filename, dpi=80)
        fig.clf()
    else:
        show()


if __name__ == '__main__':
    args = []
    for arg in argv:
        args.append(arg)

    show_figure(args, 0, bool)



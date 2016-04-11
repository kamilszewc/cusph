#!/usr/bin/env python
# -*- coding: utf-8 -*-
# All right reserved by Kamil Szewc, Gdansk 2009 

from math import *
from sys import argv,exit
from numpy import *
import os
from matplotlib.pyplot import *
from matplotlib.ticker import *
from matplotlib import rc
from postproc.Data import Data
from postproc.Field import YField

def couette(y,t,nu,l,u_n,N):
    u = u_n*y/l
    for n in range(N):
        u+=2.0*u_n*pow(-1.0,n+1)*sin( (n+1)*pi*y/l )*exp( -nu*(n+1)*(n+1)*pi*pi*t/(l*l) )/( (n+1)*pi )
    return u

def show_figure(args, type_of_output, title):
#    rc('text' , usetex = True)
#    rc('text.latex', unicode = True)
    style = ('ro', 'g^', 'cd', 'ms', 'yH')
    label = ('WCSPH', 'ISPH')

    d=[]
    r=[]
    time = ""
    t = 0.0
	
    fig = figure(1)
    pic2 = fig.add_subplot(212)
    pic1 = fig.add_subplot(211)
	
    for n in range(len(args)-1):
        d = Data(args[n+1])

        NX = 60
        NY = 60

        lx = d.XCV/float(NX)
        ly = d.YCV/float(NY)

        x = 0.5
        y = linspace(ly*0.5, ly*0.5+ly*float(NY-1), NY)

        r = YField(x, y, d)
        du = []

        if n==0:
            t = d.t
            y_a  = linspace(0.0, 1.0, 150)
            u_a  = couette(y_a,t,d.nu[0],1.0,d.V_N,500)
			
        for i in range(len(r.y)):
            du.append( abs(r.u[i]-couette(r.y[i],t,d.nu[0],1.0,d.V_N,500)))

        if n == 0:  
            pic1.plot(y_a, u_a, linewidth=2)
			    
        pic1.plot(r.y, r.u, style[n])
        pic2.plot(r.y, du, style[n])
	
	pic1.set_title(r"$t=%0.5f, u_0=%0.4f, \nu=%0.1f, \varrho=1.0$" % (t, d.V_N, d.nu[0]), size='x-large')
    #pic2.set_ylim(0.0, 0.0000004)
    #pic1.set_ylim(0.0, 0.0001)
    pic1.set_ylabel(r"$u_x$", size='xx-large')
    pic2.set_ylabel(r"$|u_x-u_x^{ex}|$", size='xx-large')
    pic2.set_xlabel(r"$y/L$", size='xx-large')
    setp(pic1.get_xticklabels(), visible=False)
    setp(pic1.get_xticklabels(), fontsize='large')
    setp(pic2.get_xticklabels(), fontsize='large')
    setp(pic1.get_yticklabels(), fontsize='large')
    setp(pic2.get_yticklabels(), fontsize='large')
    
    pic1.ticklabel_format(style='sci', scilimits=(0,0),  axis='y')
    pic2.ticklabel_format(style='sci', scilimits=(0,0),  axis='y')

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
	
	
	
	
	
	
	
	
	

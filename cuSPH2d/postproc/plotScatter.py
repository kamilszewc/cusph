#!/usr/bin/env python
# -*- coding: utf-8 -*-
# All right reserved by Kamil Szewc, Gdansk/Nancy 2009 

from sys import argv,exit
from numpy import *
import numpy
from postproc.Data import Data
from postproc.Field import Field
import matplotlib
matplotlib.use("Qt4Agg")
from pylab import *

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

    figure(figsize=array(figsize)*3)
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

    figure()
    scatter(d.x, d.y, c=sqrt(array(d.u)**2 + array(d.v)**2), s=6, linewidths=0)
    title("Velocity")
    xlim(0.0, d.XCV)
    ylim(0.0, d.YCV)
    xlabel('x')
    ylabel('y')
    colorbar()

    figure()
    scatter(d.x, d.y, c=d.u, s=6, linewidths=0)
    title("X-Velocity")
    xlim(0.0, d.XCV)
    ylim(0.0, d.YCV)
    xlabel('x')
    ylabel('y')
    colorbar()

    figure()
    scatter(d.x, d.y, c=d.v, s=6, linewidths=0)
    title("Y-Velocity")
    xlim(0.0, d.XCV)
    ylim(0.0, d.YCV)
    xlabel('x')
    ylabel('y')
    colorbar()

    figure(figsize=figsize)
    quiver(d.x, d.y, d.u, d.v)
    title("Velocity")
    xlim(0.0, d.XCV)
    ylim(0.0, d.YCV)
    xlabel('x')
    ylabel('y')

    figure()
    scatter(d.x, d.y, c=d.d, s=6, linewidths=0)
    title("Density")
    xlim(0.0, d.XCV)
    ylim(0.0, d.YCV)
    xlabel('x')
    ylabel('y')
    colorbar()

    figure()
    scatter(d.x, d.y, c=d.m, s=6, linewidths=0)
    title("Mass")
    xlim(0.0, d.XCV)
    ylim(0.0, d.YCV)
    xlabel('x')
    ylabel('y')
    colorbar()

    figure()
    scatter(d.x, d.y, c=d.p, s=6, linewidths=0)
    title("Pressure")
    xlim(0.0, d.XCV)
    ylim(0.0, d.YCV)
    xlabel('x')
    ylabel('y')
    colorbar()

    if d.T_SURFACE_TENSION > 0:
        figure(figsize=figsize)
        quiver(d.x, d.y, d.nx, d.ny)
        title("Normal vectors")
        xlim(0.0, d.XCV)
        ylim(0.0, d.YCV)
        xlabel('x')
        ylabel('y')
    
    if d.T_STRAIN_TENSOR > 0 or d.T_TURBULENCE > 0 or d.T_SOIL > 0:
        figure()
        scatter(d.x, d.y, c=d.str, s=6, linewidth=0)
        title("Strain rate")
        xlim(0.0, d.XCV)
        ylim(0.0, d.YCV)
        xlabel('x')
        ylabel('y')
        colorbar()
        
    if d.T_TURBULENCE > 0 or d.T_SOIL > 0:
        figure()
        scatter(d.x, d.y, c=d.nut, s=6, linewidth=0)
        title("Turbulent viscosity/soil viscosity")
        xlim(0.0, d.XCV)
        ylim(0.0, d.YCV)
        xlabel('x')
        ylabel('y')
        colorbar()
        
    if d.T_SOIL == 2:
        figure()
        scatter(d.x, d.y, c=d.cs, s=6, linewidth=0)
        title("Smoothed color")
        xlim(0.0, d.XCV)
        ylim(0.0, d.YCV)
        xlabel('x')
        ylabel('y')
        colorbar()
        
        for i in range(d.N):
            if 0.01 < d.cs[i] < 0.99:
                d.phaseId[i] = 2
                
        figure()
        scatter(d.x, d.y, c=d.phaseId, s=6, linewidth=0)
        title("Sediment")
        xlim(0.0, d.XCV)
        ylim(0.0, d.YCV)
        xlabel('x')
        ylabel('y')
        colorbar()

    if d.T_SURFACTANTS > 0:   
        figure(figsize=figsize)
        quiver(d.x, d.y, d.cSurfGradX, d.cSurfGradY)
        title("Surfactant concentration gradient")
        xlim(0.0, d.XCV)
        ylim(0.0, d.YCV)
        xlabel('x')
        ylabel('y')

        figure()
        scatter(d.x, d.y, c=d.cSurf, s=6, linewidths=0)
        title("Surfactant concentration")
        xlim(0.0, d.XCV)
        ylim(0.0, d.YCV)
        xlabel('x')
        ylabel('y')
        colorbar()

        figure()
        scatter(d.x, d.y, c=d.mSurf, s=6, linewidths=0)
        title("Surfactant mass")
        xlim(0.0, d.XCV)
        ylim(0.0, d.YCV)
        xlabel('x')
        ylabel('y')
        colorbar()

        figure()
        scatter(d.x, d.y, c=d.dSurf, s=6, linewidths=0)
        title("Surfactant diffusion")
        xlim(0.0, d.XCV)
        ylim(0.0, d.YCV)
        xlabel('x')
        ylabel('y')
        colorbar()
   
        figure()
        scatter(d.x, d.y, c=d.a, s=6, linewidths=0)
        title("Area")
        xlim(0.0, d.XCV)
        ylim(0.0, d.YCV)
        xlabel('x')
        ylabel('y')
        colorbar()

    if d.T_HYDROSTATIC_PRESSURE > 0:
        figure()
        scatter(d.x, d.y, c=d.ph, s=6, linewidth=0)
        title("Hydrostatic pressure")
        xlim(0.0, d.XCV)
        ylim(0.0, d.YCV)
        xlabel('x')
        ylabel('y')
        colorbar()
    
    if d.T_DISPERSED_PHASE_FLUID > 0:
        figure()
        plot(d.pdpfX, d.pdpfY, 'ko')
        title("Dispersed phase")
        xlim(0.0, d.XCV)
        ylim(0.0, d.YCV)
        xlabel('x')
        ylabel('y')
        
        figure()
        quiver(d.pdpfX, d.pdpfY, d.pdpfU, d.pdpfV)
        title("Dispersed phase")
        xlim(0.0, d.XCV)
        ylim(0.0, d.YCV)
        xlabel('x')
        ylabel('y')
        
        figure()
        scatter(d.pdpfX, d.pdpfY, c=sqrt(array(d.pdpfU)**2 + array(d.pdpfV)**2), s=6, linewidths=0)
        title("Dispersed phase velocity")
        xlim(0.0, d.XCV)
        ylim(0.0, d.YCV)
        xlabel('x')
        ylabel('y')
        colorbar()
        
        figure()
        scatter(d.pdpfX, d.pdpfY, c=d.pdpfO, s=6, linewidths=0)
        title("Dispersed phase volume fraction")
        xlim(0.0, d.XCV)
        ylim(0.0, d.YCV)
        xlabel('x')
        ylabel('y')
        colorbar()
        
        figure()
        scatter(d.x, d.y, c=d.o, s=6, linewidths=0)
        title("Fluid volume fracion")
        xlim(0.0, d.XCV)
        ylim(0.0, d.YCV)
        xlabel('x')
        ylabel('y')
        colorbar()
        
        figure()
        scatter(d.pdpfX, d.pdpfY, c=d.pdpfD, s=6, linewidths=0)
        title("Dispersed phase density")
        xlim(0.0, d.XCV)
        ylim(0.0, d.YCV)
        xlabel('x')
        ylabel('y')
        colorbar()

        figure()
        scatter(d.pdpfX, d.pdpfY, c=d.pdpfV, s=6, linewidths=0)
        title("Dispersed phase y-velocity")
        xlim(0.0, d.XCV)
        ylim(0.0, d.YCV)
        xlabel('x')
        ylabel('y')
        colorbar()

    show()

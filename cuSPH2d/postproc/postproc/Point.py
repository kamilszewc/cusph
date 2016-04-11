#!/usr/bin/env python
# -*- coding: utf-8 -*-
# All right reserved by Kamil Szewc, Gdansk 2009 

from math import sqrt
from postproc.Kernel import *

KERNEL_TYPE = 3

class Point:

    def __init__(self, x, y, d):
        self.x, self.y, self.u, self.v, self.p, self.d, self.c, self.cBulk, self.cSurf, self.vol = x, y, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        if KERNEL_TYPE <= 3:
            xc = long( 0.5*self.x/d.H )
            yc = long( 0.5*self.y/d.H )
        else:
            xc = long( self.x/(d.H*3.0) )
            yc = long( self.y/(d.H*3.0) )
        c = xc + yc*d.NXC

        self.__cell_int(d, c)
        if (c%d.NXC > 0):
            self.__cell_int(d, c-1) #W
            if (c >= d.NXC):       self.__cell_int(d, c-d.NXC-1) #WS
            if (c < d.NC - d.NXC): self.__cell_int(d, c+d.NXC-1) #WN
        if (c%d.NXC < d.NXC-1):
            self.__cell_int(d, c+1) #E
            if (c >= d.NXC):       self.__cell_int(d, c-d.NXC+1) #ES
            if (c < d.NC - d.NXC): self.__cell_int(d, c+d.NXC+1) #EN
        if (c >= d.NXC):       self.__cell_int(d, c-d.NXC) #S
        if (c < d.NC - d.NXC): self.__cell_int(d, c+d.NXC) #N

        if d.T_BOUNDARY_PERIODICITY > 0:
            if (c%d.NXC == d.NXC-1):
                self.__cell_int(d, c-d.NXC+1)
                if (c >= d.NXC):       self.__cell_int(d, c+1-2*d.NXC)
                if (c < d.NC - d.NXC): self.__cell_int(d, c+1)
            if (c%d.NXC == 0):
                self.__cell_int(d, c+d.NXC-1)
                if (c >= d.NXC):       self.__cell_int(d, c-1)
                if (c < d.NC - d.NXC): self.__cell_int(d, c+2*d.NXC-1)

        if d.T_BOUNDARY_PERIODICITY == 1:
            if (c >= d.NC - d.NXC):
                self.__cell_int(d, c-d.NXC*(d.NYC-1) )
                if (c%d.NXC > 0):       self.__cell_int(d, c-d.NXC*(d.NYC-1)-1)
                if (c%d.NXC < d.NXC-1): self.__cell_int(d, c-d.NXC*(d.NYC-1)+1)
            if (c < d.NXC):
                self.__cell_int(d, c+d.NXC*(d.NYC-1) )
                if (c%d.NXC > 0):       self.__cell_int(d, c+d.NXC*(d.NYC-1)-1)
                if (c%d.NXC < d.NXC-1): self.__cell_int(d, c+d.NXC*(d.NYC-1)+1)
            if (c == 0): self.__cell_int(d, d.NC-1)
            if (c == d.NXC-1): self.__cell_int(d, d.NC-d.NXC)
            if (c == d.NC-1): self.__cell_int(d, 0)
            if (c == d.NC-d.NXC): self.__cell_int(d, d.NXC-1)

        #self.u = self.u / self.vol
        #self.v = self.v / self.vol
        #self.d = self.d / self.vol
        #self.p = self.p / self.vol
        #self.c = self.c / self.vol

    def __cell_int(self, d, c):
        i = d.h[c]
        while i >= 0:
            self.__inter(d, i, 0)
            if True:
                if d.T_BOUNDARY_PERIODICITY != 1:
                    if (c >= d.NC - d.NXC) and (c < d.NC): #N
                        self.__inter(d, i, 1)
                        if d.T_BOUNDARY_PERIODICITY == 0:
                            if (c%d.NXC == 0):       self.__inter(d, i, 8) #NW 8
                            if (c%d.NXC == d.NXC-1): self.__inter(d, i, 5) #NE 5
                    if (c >= 0) and (c < d.NXC): #S
                        self.__inter(d, i, 3)
                        if d.T_BOUNDARY_PERIODICITY == 0:
                            if (c%d.NXC == 0):       self.__inter(d, i, 7) #SW 7
                            if (c%d.NXC == d.NXC-1): self.__inter(d, i, 6) #SE 6
                if d.T_BOUNDARY_PERIODICITY == 0:
                    if (c%d.NXC == 0):       self.__inter(d, i, 4) #W
                    if (c%d.NXC == d.NXC-1): self.__inter(d, i, 2) #E

            i = d.l[i]


    def __inter(self, d, i, t): #funkcja zwykla !!!
        if t==0:
            x = self.x - d.x[i]
            y = self.y - d.y[i]
            u = d.u[i]
            v = d.v[i]
        if t==1: #N
            x = self.x - d.x[i]
            y = 2.0*d.YCV - self.y - d.y[i]
            u = 2.0*d.V_N - d.u[i]
            v = -d.v[i]
        if t==2: #E
            x = 2.0*d.XCV - self.x - d.x[i]
            y = self.y - d.y[i]
            u = -d.u[i]
            v = 2.0*d.V_E - d.v[i]
        if t==3: #S
            x = self.x - d.x[i]
            y = -self.y - d.y[i]
            u = 2.0*d.V_S - d.u[i]
            v = -d.v[i]
        if t==4: #W
            x = -self.x - d.x[i]
            y = self.y - d.y[i]
            u = -d.u[i]
            v = 2.0*d.V_W - d.v[i]
        if t==5: #NE
            x = 2.0*d.XCV - self.x - d.x[i]
            y = 2.0*d.YCV - self.y - d.y[i]
            u = d.u[i]
            v = d.v[i]
        if t==8: #NW
            x = -self.x - d.x[i]
            y = 2.0*d.YCV - self.y - d.y[i]
            u = d.u[i]
            v = d.v[i]
        if t==6: #SE
            x = 2.0*d.XCV - self.x - d.x[i]
            y = -self.y - d.y[i]
            u = d.u[i]
            v = d.v[i]
        if t==7: #SW
            x = -self.x - d.x[i]
            y = -self.y - d.y[i]
            u = d.u[i]
            v = d.v[i]

        if (d.T_BOUNDARY_PERIODICITY > 0) and (x >  6.0*d.H): x-=d.XCV
        if (d.T_BOUNDARY_PERIODICITY > 0) and (x < -6.0*d.H): x+=d.XCV
        if (d.T_BOUNDARY_PERIODICITY == 1) and (y >  6.0*d.H): y-=d.YCV
        if (d.T_BOUNDARY_PERIODICITY == 1) and (y < -6.0*d.H): y+=d.YCV

        q  = sqrt(x*x + y*y)/(d.sl[i])

        if q < 2.0:
            k = kernel(q, d.sl[i], KERNEL_TYPE)
            vol = d.m[i] / d.d[i]
            self.u += u*k*vol
            self.v += v*k*vol
            self.p += d.p[i]*k*vol
            #self.d += d.d[i]*k
            self.d += d.m[i]*k
            try:
                self.c += d.c[i]*k*vol
            except:
               pass 
            self.vol += vol*k
            if d.T_SURFACTANTS != 0:
                self.cBulk += d.cBulk[i]*k*vol
                self.cSurf += d.cSurf[i]*k*vol

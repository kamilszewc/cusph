#!/usr/bin/env python
# -*- coding: utf-8 -*-
# All right reserved by Kamil Szewc, Gdansk 2015

import numpy as np

def kernel(q):
    return (42.0/256.)*(q+0.5)*(2.0-q)**4

class field:
    """Field - projection class"""
    
    def __init__(self, data, *arg):
        self.NX = data.NX
        self.NY = data.NY
        self.NZ = data.NZ
        if arg != ():
            self.NX = arg[0]
            self.NY = arg[1]
            self.NZ = arg[2]
        
        self.x = np.linspace(2.0*data.H, data.XCV-2.0*data.H, self.NX)
        self.y = np.linspace(2.0*data.H, data.YCV-2.0*data.H, self.NY)
        self.z = np.linspace(2.0*data.H, data.ZCV-2.0*data.H, self.NZ)
        self.u = np.zeros( (len(self.x), len(self.y), len(self.z)) )
        self.v = np.zeros( (len(self.x), len(self.y), len(self.z)) )
        self.w = np.zeros( (len(self.x), len(self.y), len(self.z)) )
        
        for i in range(len(self.x)):
            for j in range(len(self.y)):
                for k in range(len(self.z)):
                     xc = long(0.5*self.x[i]*data.I_H)
                     yc = long(0.5*self.y[j]*data.I_H)
                     zc = long(0.5*self.z[k]*data.I_H)
                     for ic in [-1,0,1]:
                        for jc in [-1,0,1]:
                            for kc in [-1,0,1]:
                                cellNumber = (xc+ic) + (yc+jc)*data.NXC + (zc+kc)*data.NXC*data.NYC
                                try:
                                    for pid in data.particleInCell[cellNumber]:
                                        r2 = (data.x[pid]-self.x[i])**2 +(data.y[pid]-self.y[j])**2 +(data.z[pid]-self.z[k])**2
                                        if r2 < 4.0*data.H**2:
                                            help = data.KNORM * kernel( data.I_H * r2**0.5 ) * data.m[pid] / data.d[pid]
                                            self.u[i,j,k] += data.u[pid] * help
                                            self.v[i,j,k] += data.v[pid] * help
                                            self.w[i,j,k] += data.w[pid] * help
                                except:
                                    pass
                                    
        self.x, self.y, self.z = np.meshgrid(self.x, self.y, self.z)

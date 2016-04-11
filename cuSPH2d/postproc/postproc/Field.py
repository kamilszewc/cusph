#!/usr/bin/env python
# -*- coding: utf-8 -*-
# All right reserved by Kamil Szewc, Gdansk 2009 


import numpy as np
from postproc.Point import Point
try:
    from multiprocessing import Process, Queue, cpu_count
    class Multi(Process):
        pass
except ImportError:
    from threading import Thread
    from Queue import Queue
    class Multi(Thread):
        pass
    def cpu_count():
        return 1

class Field:
    def __init__(self, d, *arg):
        if arg != (): d.NX = arg[0]; d.NY = arg[1]
        self.dx = d.XCV/float(d.NX)
        self.dy = d.YCV/float(d.NY)
        self.x = np.linspace(self.dx*0.5, self.dx*0.5+self.dx*float(d.NX-1), d.NX)
        self.y = np.linspace(self.dy*0.5, self.dy*0.5+self.dy*float(d.NY-1), d.NY)
        self.u  = np.zeros( (len(self.y),len(self.x)) )
        self.v  = np.zeros( (len(self.y),len(self.x)) )
        self.p  = np.zeros( (len(self.y),len(self.x)) )
        self.d  = np.zeros( (len(self.y),len(self.x)) )
        self.c  = np.zeros( (len(self.y),len(self.x)) )

        ncpu = cpu_count()
        clen = len(self.x)*len(self.y)
        clist = range(clen)
        clen = clen/2
        clists = [ clist[n*clen:(n+1)*clen] for n in range(ncpu)[:-1] ]
        clists.append(clist[(ncpu-1)*clen:])

        qs = [Queue() for n in range(ncpu)]
        ps = [Multi(target=self._point, args=(clists[n],d,qs[n],)) for n in range(ncpu)]
        for n in range(ncpu): ps[n].start()
        for n in range(ncpu): qs[n] = qs[n].get()
        for n in range(ncpu): ps[n].join()
        for n in range(ncpu):
            self.u += qs[n][0]
            self.v += qs[n][1]
            self.p += qs[n][2]
            self.d += qs[n][3]
            self.c += qs[n][4]
        self.x,self.y = np.meshgrid(self.x,self.y)

    def _point(self, clist, d, q):
        for c in clist:
            i = c%len(self.x)
            j = ((c-i)/len(self.x))%len(self.y)
            values = Point(self.x[i], self.y[j], d)
            self.u[j,i] = values.u
            self.v[j,i] = values.v
            self.p[j,i] = values.p
            self.d[j,i] = values.d
            self.c[j,i] = values.c
        q.put((self.u,self.v,self.p,self.d,self.c))

class FieldS:
    def __init__(self, d):
        self.dx = d.XCV/float(d.NX)
        self.dy = d.YCV/float(d.NY)
        self.x = np.linspace(self.dx*0.5, self.dx*0.5+self.dx*float(d.NX-1), d.NX)
        self.y = np.linspace(self.dy*0.5, self.dy*0.5+self.dy*float(d.NY-1), d.NY)
        self.u  = np.zeros( (len(self.y),len(self.x)) )
        self.v  = np.zeros( (len(self.y),len(self.x)) )
        self.p  = np.zeros( (len(self.y),len(self.x)) )
        self.d  = np.zeros( (len(self.y),len(self.x)) )
        self.c  = np.zeros( (len(self.y),len(self.x)) )
        for i in range(d.NX):
            for j in range(d.NY):
                values = Point(self.x[i], self.y[j], d)
                self.u[j,i] = values.u
                self.v[j,i] = values.v
                self.p[j,i] = values.p
                self.d[j,i] = values.d
                self.c[j,i] = values.c


class PField:
    def __init__(self, x, y, data):
        self.x, self.y = x, y
        self.u, self.v, self.p, self.d, self.c  = [], [], [], [], []
        for i in range(len(x)):
            values = Point(x[i], y[i], data)
            self.u.append(values.u)
            self.v.append(values.v)
            self.p.append(values.p)
            self.d.append(values.d)
            self.c.append(values.c)
    
class XField:
    def __init__(self, x, y, data):
        self.x = x
        self.ny, self.u, self.v, self.p, self.d, self.c, self.cBulk = [], [], [], [], [], [], []
        for i in range(len(x)):
            values = Point(x[i], y, data)
            self.ny.append(y)
            self.u.append(values.u)
            self.v.append(values.v)
            self.p.append(values.p)
            self.d.append(values.d)
            self.c.append(values.c)
            self.cBulk.append(values.cBulk)
    
class YField:
    def __init__(self, x, y, data):
        self.y = y
        self.nx, self.u, self.v, self.p, self.d, self.c, self.cBulk = [], [], [], [], [], [], []
        for i in range(len(y)):
            values = Point(x, y[i], data)
            self.nx.append(x)
            self.u.append(values.u)
            self.v.append(values.v)
            self.p.append(values.p)
            self.d.append(values.d)
            self.c.append(values.c)
            self.cBulk.append(values.cBulk)

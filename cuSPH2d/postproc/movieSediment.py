#!/usr/bin/env python
# -*- coding: utf-8 -*-
# All right reserved by Kamil Szewc, Gdansk 2009 

from sys import argv
from numpy import *
import os
from glob import glob
try:
    from multiprocessing import Process, cpu_count
    class Multi(Process): pass
except ImportError:
    from threading import Thread
    from Queue import Queue
    class Multi(Thread): pass
    def cpu_count(): return 1
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from postproc.Data import Data

def makepng(arg, i):
    d = Data(arg)

    plt.figure(figsize=(14,6))
    plt.xlim(0,d.XCV)
    plt.ylim(0,d.YCV)
    #plt.plot(d.x, d.y, 'yo', markersize=2.1, markeredgecolor='y')
    plt.scatter(d.x, d.y, c=sqrt(array(d.u)**2 + array(d.v)**2), s=8, linewidths=0)
    plt.clim(0,0.2)
    plt.colorbar()
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    if d.T_DISPERSED_PHASE_FLUID > 0:
        plt.plot(d.pdpfX, d.pdpfY, 'ko', markersize=3.5, markeredgecolor='k')
    plt.tight_layout()

    filename = str('img%05d' % i) + '.png'
    plt.savefig(filename, dpi=300)
    print('Wrote file', filename)
    plt.close()

def queue(lista,N):
    i = N
    for arg in lista:
        print( arg, os.getpid())
        makepng(arg, i)
        i += 1
        
if __name__ == '__main__':
    #ncpu = cpu_count()
    ncpu = 6
    lista = []
    for args in argv[1:]:
        for arg in glob(args):
            lista.append(arg)

    N = len(lista)/ncpu
    lists = [ lista[n*N:(n+1)*N] for n in range(ncpu)[:-1] ]
    lists.append(lista[(ncpu-1)*N:])

    ps = [ Multi(target=queue, args=(lists[n], n*N,)) for n in range(ncpu) ]
    for n in range(ncpu): ps[n].start()
    for n in range(ncpu): ps[n].join()

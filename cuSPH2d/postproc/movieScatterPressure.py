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

    extent=(0.0, d.XCV, 0.0, d.YCV)

    plt.figure(figsize=(d.XCV/8.0, d.YCV/4))
    plt.scatter(d.x, d.y, c=array(d.p), s=2.0, linewidths=0)
    #plt.title("Surfactant mass")
    #plt.xlim(0.0, d.XCV)
    #plt.ylim(0.0, d.YCV)
    plt.xlim(20.0, 80.0)
    plt.ylim(0.0, 10.0)
    plt.xlabel('x [cm]')
    plt.ylabel('y [cm]')
    plt.clim(-100.0, 15000.0)
    plt.colorbar()
    plt.tight_layout()
    #plt.text(0.0102, 0.0106, "t=%.3f" % (d.t), fontsize='x-large')
    
    filename = str('img%05d' % i) + '.png'
    plt.savefig(filename, dpi=300)
    print('Wrote file', filename)
    plt.clf()

def queue(lista,N):
    i = N
    for arg in lista:
        print(arg, os.getpid())
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

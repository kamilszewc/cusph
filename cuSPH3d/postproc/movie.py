#!/usr/bin/env python
from sys import argv
import bz2
from numpy import *
from glob import glob
try:
    from enthought.mayavi import mlab
except:
    from mayavi import mlab
from postproc.data import data

mlab.options.offscreen = True

def function(arg, i):
    x, y, z = [], [], []

    f = file(arg)
    filelines = f.readlines()
    f.close()

    for fileline in filelines:
        x.append(float(fileline.split()[0]))
        y.append(float(fileline.split()[1]))
        z.append(float(fileline.split()[2]))

    mlab.figure(bgcolor=(1.0,1.0,1.0))
    points = mlab.points3d(x, y, z, color=(0.5,0.8,0.1), scale_factor=0.5)
#mlab.axes(points, nb_labels=5, color=(0.0, 0.0, 0.0), extent=(0.0,1.0,0.0,1.0,0.0,1.0), ranges = (0.,1.,0.,1.,0.,1.), xlabel='x', ylabel='y', zlabel='z')

#mlab.outline(points, line_width=2.1, color=(0.0, 0.0, 0.0), extent=(0,8,0,4,0,16))
#mlab.outline(points, line_width=2.1, color=(0.0, 0.0, 0.0), extent=(0,1,0,1,0,1))
    mlab.view(-90, -70, 60.0)
#f = mlab.gcf()
#camera = f.scene.camera
#camera.zoom(5.0)
    mlab.savefig("%05d.png" % (i,), magnification=3)
    print i, arg

    mlab.show()
    mlab.clf()
    
if __name__ == '__main__':
    lista = []
    for args in argv[1:]:
        for arg in glob(args):
            lista.append(arg)
    i = 0
    for arg in lista:
        function(arg, i)
        i += 1

#i!/usr/bin/env python
from sys import argv
import bz2
from numpy import *
try:
    from enthought.mayavi import mlab
except:
    from mayavi import mlab
from postproc.data import data

#mlab.options.offscreen = True

n = 0
for arg in argv[1:]:
    d = data(arg)
    print arg

    x, y, z = [], [], []
    for i in range(len(d.x)):
        if d.c[i] > 0.5:
            x.append(d.x[i])
            y.append(d.y[i])
            z.append(d.z[i])


    mlab.figure(bgcolor=(1.0,1.0,1.0))
    points = mlab.points3d(x, y, z, color=(0.5,0.8,0.1))
    #mlab.axes(points, nb_labels=5, color=(0.0, 0.0, 0.0), extent=(0.0,1.0,0.0,1.0,0.0,1.0), ranges = (0.,1.,0.,1.,0.,1.), xlabel='x', ylabel='y', zlabel='z')

    #mlab.outline(points, line_width=2.1, color=(0.0, 0.0, 0.0), extent=(0,8,0,8,0,16))
    #mlab.view(20, 90, 40)
    #f = mlab.gcf()
    #camera = f.scene.camera
    #camera.zoom(1.5)
    #mlab.savefig("%03d.png" % (n,), magnification=3) 

    mlab.show()
    mlab.clf()
    n += 1

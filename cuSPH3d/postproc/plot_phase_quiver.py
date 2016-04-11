#!/usr/bin/env python
from sys import argv
import bz2
from numpy import *
try:
    from enthought.mayavi import mlab
except:
    from mayavi import mlab
from postproc.data import data


d = data(argv[1])

x, y, z, u, v, w = [], [], [], [], [], []
for i in range(len(d.x)):
    if d.c[i] > 0.5:
        x.append(d.x[i])
        y.append(d.y[i])
        z.append(d.z[i])
        u.append(d.u[i])
        v.append(d.v[i])
        w.append(d.w[i])

mlab.figure(bgcolor=(1.0,1.0,1.0))
mlab.quiver3d(x, y, z, u, v, w)
#mlab.axes(points, nb_labels=5, color=(0.0, 0.0, 0.0), extent=(0.0,1.0,0.0,1.0,0.0,1.0), ranges = (0.,1.,0.,1.,0.,1.), xlabel='x', ylabel='y', zlabel='z')

#mlab.outline(points, line_width=2.1, color=(0.0, 0.0, 0.0), extent=(0,1,0,1,0,1))
#mlab.view(azimuth=20, distance=5)
#f = mlab.gcf()
#camera = f.scene.camera
#camera.zoom(1.5)
#mlab.savefig('droplet.png', magnification=5) 
mlab.show()


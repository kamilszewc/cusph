#!/usr/bin/env python
from sys import argv
import bz2
from numpy import *
try:
    from enthought.mayavi import mlab
except:
    from mayavi import mlab
from postproc.data import data

#mlab.options.offscreen = True

x, y, z, u, v, w, p = [], [], [], [], [], [], []
for arg in argv[1:]:
    f = file(arg)
    filelines = f.readlines()
    f.close()

    i = 0
    for fileline in filelines:
        if i%1 == 0:
            x.append(float(fileline.split()[0]))
            y.append(float(fileline.split()[1]))
            z.append(float(fileline.split()[2]))
            u.append(float(fileline.split()[3]))
            v.append(float(fileline.split()[4]))
            w.append(float(fileline.split()[5]))
            try:
                p.append(float(fileline.split()[-1]))
            except:
                pass
        i += 1

mlab.figure(bgcolor=(1.0,1.0,1.0))
points = mlab.points3d(x, y, z, color=(0.5,0.8,0.1))#, scale_factor=0.5)
#mlab.axes(points, nb_labels=5, color=(0.0, 0.0, 0.0), extent=(0.0,1.0,0.0,1.0,0.0,1.0), ranges = (0.,1.,0.,1.,0.,1.), xlabel='x', ylabel='y', zlabel='z')
#points = mlab.points3d(x, y, z, p)#, scale_mode='none', scale_factor=0.5)

#points = mlab.quiver3d(x, y, z, u, v, w)
#mlab.vectorbar(points, title='Velocity')

#mlab.outline(points, line_width=2.1, color=(0.0, 0.0, 0.0), extent=(0,8,0,4,0,16))
#mlab.outline(points, line_width=2.1, color=(0.0, 0.0, 0.0), extent=(0,1,0,1,0,1))
#mlab.view(-90, -90, 20.0)
#f = mlab.gcf()
#camera = f.scene.camera
#camera.zoom(5.0)
#mlab.savefig("%03d.png" % (0,), magnification=4)

mlab.show()
mlab.clf()

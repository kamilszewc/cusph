#!/usr/bin/env python
from sys import argv
from glob import glob
import time
from numpy import *
try:
    from enthought.mayavi import mlab
except:
    from mayavi import mlab
from postproc.data import data

#mlab.options.offscreen = True

arglist = []
for args in argv[1:]:
    for arg in glob(args):
        arglist.append(arg)

n = 0
for arg in arglist:
    time1 = time.time()
    
    d = data(arg, ('velocity','pressure', 'density'))

    mlab.figure(bgcolor=(1.0,1.0,1.0), fgcolor=(0.0,0.0,0.0))
    
    points = mlab.points3d(d.x, d.y, d.z, sqrt(d.u**2 + d.v**2 + d.w**2), scale_mode='none', resolution=4, scale_factor=0.025, vmin=0.0, vmax=1.0)
    #points = mlab.quiver3d(d.x, d.y, d.z, d.u, d.v, d.w)
    
    scalarbar=mlab.colorbar(points, title='Velocity', orientation='vertical', nb_labels=5)
    
    #mlab.axes(points, nb_labels=5, color=(0.0, 0.0, 0.0), extent=(0.0,1.0,0.0,1.0,0.0,1.0), ranges = (0.,1.,0.,1.,0.,1.), xlabel='x', ylabel='y', zlabel='z')
    mlab.outline(points, line_width=2., color=(0.0, 0.0, 0.0), extent=(0,1,0,1,0,1))
    mlab.view(azimuth=300, elevation=60, distance=3.5)
    #mlab.orientation_axes()
    
    #f = mlab.gcf()
    #camera = f.scene.camera
    #camera.zoom(1.5)
    
    #mlab.savefig("%03d.png" % (n,), magnification=3) 
    mlab.figure(bgcolor=(1.0,1.0,1.0), fgcolor=(0.0,0.0,0.0))
    points = mlab.points3d(d.x, d.y, d.z, d.p, scale_mode='none')
    scalarbar=mlab.colorbar(points, title='Pressure', orientation='vertical', nb_labels=5)
    
    mlab.figure(bgcolor=(1.0,1.0,1.0), fgcolor=(0.0,0.0,0.0))
    points = mlab.points3d(d.x, d.y, d.z, d.d, scale_mode='none')
    scalarbar=mlab.colorbar(points, title='Density', orientation='vertical', nb_labels=5)
    
    mlab.show()
    
    mlab.clf()
    
    time2 = time.time()
    
    print "Processed %s -> %03d.png in %ss" % (arg, n, time2-time1)
    n += 1


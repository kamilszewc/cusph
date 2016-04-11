#!/usr/bin/env python
# -*- coding: utf-8 -*-
# All right reserved by Kamil Szewc, Gdansk 2015

from sys import argv
import time
from numpy import *
try:
    from enthought.mayavi import mlab
except:
    from mayavi import mlab
from postproc.data import data
from postproc.field import field

time1 = time.time()
d = data(argv[1], ("velocity","density", "mass"))
time2 = time.time()
f = field(d, 10,10,10)
time3 = time.time()
print "Reading data: %s" % (time2-time1)
print "Projecting: %s" % (time3-time2)
print "Total: %s" % (time3-time1)

mlab.figure(bgcolor=(1.0,1.0,1.0))
mlab.quiver3d(f.u, f.v, f.w)
mlab.outline()

src = mlab.pipeline.vector_field(f.u, f.v, f.w)
#mlab.pipeline.vector_cut_plane(src, mask_points=2, scale_factor=3)

#magnitude = mlab.pipeline.extract_vector_norm(src)
#mlab.pipeline.iso_surface(magnitude, contours=[0.5, 0.8])

#flow = mlab.flow(f.u, f.v, f.w, seed_scale=1,
#                                seed_resolution=6,
#                                seed_visible=False,
#                                integration_direction='both')

mlab.show()

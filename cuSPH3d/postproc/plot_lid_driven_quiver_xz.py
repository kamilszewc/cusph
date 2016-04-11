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

figure=mlab.figure(bgcolor=(1.0,1.0,1.0), fgcolor=(0.0,0.0,0.0), size=(1024,768))

DX = 0.01
SHIFT = 0.01

xv, yv, zv, uv, vv, wv = [], [], [], [], [], []
for x, y, z, u, v, w in zip(d.x, d.y, d.z, d.u, d.v, d.w):
    if y+SHIFT < 0.5+DX and y+SHIFT > 0.5-DX:  
        xv.append(x)
        yv.append(y)
        zv.append(z)
        uv.append(u)
        vv.append(v)
        wv.append(w)

points = mlab.quiver3d(xv, yv, zv, uv, vv, wv, mode='arrow', resolution=8, mask_points=1, scale_mode='scalar', scale_factor=0.047)
#vectorbar=mlab.vectorbar(points, title='Velocity', orientation='vertical', nb_labels=5)
#mlab.axes(points, nb_labels=5, color=(0.0, 0.0, 0.0), extent=(0.0,1.0,0.0,1.0,0.0,1.0), ranges = (0.,1.,0.,1.,0.,1.), xlabel='x', ylabel='y', zlabel='z')
mlab.outline(points, line_width=2.1, color=(0.0, 0.0, 0.0), extent=(0,1,0,1,0,1))
mlab.view(azimuth=270, elevation=90, distance=2.8)
mlab.orientation_axes()

"""try:
    engine = mayavi.engine
except NameError:
    from mayavi.api import Engine
    engine = Engine()
    engine.start()
if len(engine.scenes) == 0:
    engine.new_scene()
# ------------------------------------------- 
module_manager = figure.scenes[0].children[0].children[0]
module_manager.scalar_lut_manager.scalar_bar_representation.minimum_size = array([1, 1])
module_manager.scalar_lut_manager.scalar_bar_representation.position2 = array([ 0.17,  0.8 ])
module_manager.scalar_lut_manager.scalar_bar_representation.position = array([ 0.82,  0.1 ])
module_manager.scalar_lut_manager.scalar_bar_representation.maximum_size = array([100000, 100000])
module_manager.scalar_lut_manager.scalar_bar.height = 0.8
module_manager.scalar_lut_manager.scalar_bar.width = 0.17
module_manager.scalar_lut_manager.scalar_bar.position2 = array([ 0.17,  0.8 ])
module_manager.scalar_lut_manager.scalar_bar.position = array([ 0.82,  0.1 ])
module_manager.scalar_lut_manager.show_scalar_bar = True
module_manager.scalar_lut_manager.show_legend = True
module_manager.scalar_lut_manager.scalar_bar.position2 = array([ 0.17,  0.8 ])
module_manager.scalar_lut_manager.scalar_bar.position = array([ 0.82,  0.1 ])
module_manager.scalar_lut_manager.scalar_bar.title = 'Velocity'
module_manager.scalar_lut_manager.scalar_bar.position2 = array([ 0.17,  0.8 ])
module_manager.scalar_lut_manager.scalar_bar.position = array([ 0.82,  0.1 ])
module_manager.scalar_lut_manager.data_name = u'Velocity'"""

mlab.show()

#mlab.savefig('lid_driven_quiver_xz.png', magnification=2) 



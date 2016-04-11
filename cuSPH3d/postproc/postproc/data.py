#!/usr/bin/env python
# -*- coding: utf-8 -*-
# All right reserved by Kamil Szewc, Gdansk 2015

import os.path
import xml.etree.ElementTree as ET
import numpy as np

class data:
    """Data - this class reads particles information from file"""

    def __init__(self, filename, *arg):
        
        if arg == (): self.fields = ("all",)
        else: self.fields = arg[0]
        
        extension = filename.split('.')[-1]
        if extension == 'bz2':
            extension = filename.split('.')[-2]

        if extension == 'xml':
            self.xmlTree = ET.parse(filename)
            self.xmlRoot = self.xmlTree.getroot()
            self.__read_param_from_xml__(filename)
            self.__read_particles_from_xml__(filename)
        else:
            print "Wrong extension"

        self.__set_linked_list__()

	if filename.split('.')[-1] == "bz2":
            self.t = os.path.split(filename)[-1].split('.')[0] + '.'
            self.t = self.t + os.path.split(filename)[-1].split('.')[1]
            self.t = float(self.t)
        else:
            self.t = float(os.path.splitext(os.path.split(filename)[-1])[0])
        
    def __set_linked_list__(self):
        self.particleInCell = [[] for i in range(self.NC)]

        for i in range(self.N):
            xc = long( 0.5*self.x[i]/self.H )
            yc = long( 0.5*self.y[i]/self.H )
            zc = long( 0.5*self.z[i]/self.H )
            c = xc + yc*self.NXC + zc*self.NXC*self.NYC
            if (xc < self.NXC) and (yc < self.NYC) and (zc < self.NZC) and (xc >= 0) and (yc >= 0) and (zc >= 0):
                self.particleInCell[c].append(i)
    
            
    def __read_param_from_xml__(self, filename):
        for child in self.xmlRoot[0][0]:
            raw = child.attrib["name"]
            value = child.text
            
            if (raw == "HDR"): self.HDR = float(value)
            if (raw == "N"): self.N = int(value)
            if (raw == "NXC"): self.NXC = int(value)
            if (raw == "NYC"): self.NYC = int(value)
            if (raw == "NZC"): self.NZC = int(value)
            if (raw == "XCV"): self.XCV = float(value)
            if (raw == "YCV"): self.YCV = float(value)
            if (raw == "ZCV"): self.ZCV = float(value)
            if (raw == "DT"): self.DT = float(value)
            if (raw == "END_TIME"): self.END_TIME = float(value)
            if (raw == "G_X"): self.G_X = float(value)
            if (raw == "G_Y"): self.G_Y = float(value)
            if (raw == "G_Z"): self.G_Z = float(value)
            if (raw == "INTERVAL_TIME"): self.INTERVAL_TIME = float(value)
            if (raw == "T_BOUNDARY_PERIODICITY"): self.T_BOUNDARY_PERIODICITY = int(value)
            if (raw == "T_INTERFACE_CORRECTION"): self.T_INTERFACE_CORRECTION = int(value)
            if (raw == "INTERFACE_CORRECTION"): self.INTERFACE_CORRECTION = float(value)
            if (raw == "T_MODEL"): self.T_MODEL = int(value)
            if (raw == "T_SURFACE_TENSION"): self.T_SURFACE_TENSION = int(value)
            if (raw == "SURFACE_TENSION"): self.SURFACE_TENSION = float(value)
            if (raw == "T_TIME_STEP"): self.T_TIME_STEP = int(value)
            if (raw == "T_HYDROSTATIC_PRESSURE"): self.T_HYDROSTATIC_PRESSURE = int(value)
            if (raw == "V_E"): self.V_E = float(value)
            if (raw == "V_N"): self.V_N = float(value)
            if (raw == "V_S"): self.V_S = float(value)
            if (raw == "V_W"): self.V_W = float(value)
            if (raw == "V_T"): self.V_T = float(value)
            if (raw == "V_B"): self.V_B = float(value)
        
        self.H = 0.5 * self.XCV / self.NXC
        self.NC = self.NXC * self.NYC * self.NZC
        self.I_H = 1.0 / self.H
        self.DH = 0.01 * self.H
        self.KNORM = (1.0/3.1415) * pow(self.I_H, 3.0)
        self.GKNORM = (1.0/3.1415) * pow(self.I_H, 5.0)
        self.DR = self.H / self.HDR

        self.NX = int(self.XCV/self.DR)
        self.NY = int(self.YCV/self.DR)
        self.NZ = int(self.ZCV/self.DR)

    def __read_particles_from_xml__(self, filename):
        for child in self.xmlRoot[1]:
            name = child.attrib["name"]
            values = child.text

            if (name == "id"):
                self.id = np.fromstring(values, dtype=int, sep=' ')
            if (name == "phase-id"):
                self.phaseId = np.fromstring(values, dtype=int, sep=' ')
            if (name == "phase-type"):
                self.phaseId = np.fromstring(values, dtype=int, sep=' ')
            if (name == "x-position"):
                self.x = np.fromstring(values, dtype=float, sep=' ')
            if (name == "y-position"):
                self.y = np.fromstring(values, dtype=float, sep=' ')
            if (name == "z-position"):
                self.z = np.fromstring(values, dtype=float, sep=' ')
            if (name == "x-velocity" and ("x-velocity" in self.fields or "velocity" in self.fields or "all" in self.fields)):
                self.u = np.fromstring(values, dtype=float, sep=' ')
            if (name == "y-velocity" and ("y-velocity" in self.fields or "velocity" in self.fields or "all" in self.fields)):
                self.v = np.fromstring(values, dtype=float, sep=' ')
            if (name == "z-velocity" and ("z-velocity" in self.fields or "velocity" in self.fields or "all" in self.fields)):
                self.w = np.fromstring(values, dtype=float, sep=' ')
            if (name == "mass" and ("mass" in self.fields or "all" in self.fields)):
                self.m = np.fromstring(values, dtype=float, sep=' ')
            if (name == "pressure" and ("pressure" in self.fields or "all" in self.fields)):
                self.p = np.fromstring(values, dtype=float, sep=' ')
            if (name == "density" and ("density" in self.fields or "all" in self.fields)):
                self.d = np.fromstring(values, dtype=float, sep=' ')
            if (name == "initial-density" and ("initial-density" in self.fields or "all" in self.fields)):
                self.di = np.fromstring(values, dtype=float, sep=' ')
            if (name == "dynamic-viscosity" and ("dynamic-viscosity" in self.fields or "all" in self.fields)):
                self.mi = np.fromstring(values, dtype=float, sep=' ')
            if (name == "gamma"and ("gamma" in self.fields or "all" in self.fields)):
                self.gamma = np.fromstring(values, dtype=float, sep=' ')
            if (name == "sound-speed" and ("sound-speed" in self.fields or "all" in self.fields)):
                self.s = np.fromstring(values, dtype=float, sep=' ')
            if (name == "equation-of-state-coefficient" and ("equation-of-state-coefficient" in self.fields or "all" in self.fields)):
                self.b = np.fromstring(values, dtype=float, sep=' ')
            if (name == "color-function" and ("color-function" in self.fields or "all" in self.fields)):
                self.c = np.fromstring(values, dtype=float, sep=' ')
            if (name == "x-normal-vector" and ("x-normal-vector" in self.fields or "normal-vector" in self.fields or "all" in self.fields)):
                self.nx = np.fromstring(values, dtype=float, sep=' ')
            if (name == "y-normal-vector" and ("y-normal-vector" in self.fields or "normal-vector" in self.fields or "all" in self.fields)):
                self.ny = np.fromstring(values, dtype=float, sep=' ')
            if (name == "z-normal-vector" and ("z-normal-vector" in self.fields or "normal-vector" in self.fields or "all" in self.fields)):
                self.nz = np.fromstring(values, dtype=float, sep=' ')
            if (name == "normal-vector-norm" and ("normal-vector-norm" in self.fields or "normal-vector" in self.fields or "all" in self.fields)):
                self.nw = np.fromstring(values, dtype=float, sep=' ')
            if (name == "curvature-influence-indicator" and ("curvature-influence-indicator" in self.fields or "all" in self.fields)):
                self.na = np.fromstring(values, dtype=int, sep=' ')
            if (name == "curvature" and ("curvature" in self.fields or "all" in self.fields)):
                self.cu = np.fromstring(values, dtype=float, sep=' ')
            if (name == "x-surface-tension" and ("x-surface-tension" in self.fields or "surface-tension" in self.fields or "all" in self.fields)):
                self.stx = np.fromstring(values, dtype=float, sep=' ')
            if (name == "y-surface-tension" and ("y-surface-tension" in self.fields or "surface-tension" in self.fields or "all" in self.fields)):
                self.sty = np.fromstring(values, dtype=float, sep=' ')
            if (name == "z-surface-tension" and ("z-surface-tension" in self.fields or "surface-tension" in self.fields or "all" in self.fields)):
                self.stz = np.fromstring(values, dtype=float, sep=' ')
            if (name == "hydrostatic-pressure" and ("hydrostatic-pressure" in self.fields or "all" in self.fields)):
                self.ph = np.fromstring(values, dtype=float, sep=' ')
        
        if ( ("kinematic-viscosity" in self.fields and "dynamic-viscosity" in self.fields and "density" in self.fields) or "all" in self.fields):
            self.nu = []
            for mi, d in zip(self.mi, self.d):           
                self.nu.append(mi/d)
            self.nu = np.array(self.nu)

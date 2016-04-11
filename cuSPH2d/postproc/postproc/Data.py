#!/usr/bin/env python
# -*- coding: utf-8 -*-
# All right reserved by Kamil Szewc, Gdansk 2009 

import os.path
from xml.dom import minidom

KERNEL_TYPE = 3

class Data:
    """Data - this class reads particles information from file"""

    def __init__(self, filename):
        extension = filename.split('.')[-1]
        if extension == 'bz2':
            extension = filename.split('.')[-2]

        if extension == 'xml': 
            self.__read_param_from_xml__(filename)
            self.__read_particles_from_xml__(filename)
        else:
            print("Wrong extension")

        self.__set_linked_list__()
	
        if filename.split('.')[-1] == "bz2":
            self.t = os.path.split(filename)[-1].split('.')[0] + '.'
            self.t = self.t + os.path.split(filename)[-1].split('.')[1]
            self.t = float(self.t)
        else:
            self.t = float(os.path.splitext(os.path.split(filename)[-1])[0])
        
    def __set_linked_list__(self):
        self.h = [-1 for i in range(self.NC)]
        self.l = [-1 for i in range(self.N)]

        for i in range(self.N):
            if KERNEL_TYPE <= 3:
                xc = int( 0.5*self.x[i]/self.H )
                yc = int( 0.5*self.y[i]/self.H )
            else:
                xc = int( self.x[i]/ (self.H*3.0) )
                yc = int( self.y[i]/ (self.H*3.0) )
            c = xc + yc*self.NXC
            if (xc < self.NXC) and (yc < self.NYC) and (xc >= 0) and (yc >= 0):
                self.l[i] = self.h[c]
                self.h[c] = i

            
    def __read_param_from_xml__(self, filename):
        xmldoc = minidom.parse(filename)
        parameterList = xmldoc.getElementsByTagName('parameter')
        for s in parameterList:
            raw = s.attributes['name'].value
            value = s.childNodes[0].nodeValue
            if (raw == "HDR"): self.HDR = float(value)
            if (raw == "N"): self.N = int(value)
            if (raw == "NXC"): self.NXC = int(value)
            if (raw == "NYC"): self.NYC = int(value)
            if (raw == "XCV"): self.XCV = float(value)
            if (raw == "YCV"): self.YCV = float(value)
            if (raw == "DT"): self.DT = float(value)
            if (raw == "END_TIME"): self.END_TIME = float(value)
            if (raw == "G_X"): self.G_X = float(value)
            if (raw == "G_Y"): self.G_Y = float(value)
            if (raw == "INTERVAL_TIME"): self.INTERVAL_TIME = float(value)
            if (raw == "T_BOUNDARY_PERIODICITY"): self.T_BOUNDARY_PERIODICITY = int(value)
            if (raw == "T_INTERFACE_CORRECTION"): self.T_INTERFACE_CORRECTION = int(value)
            if (raw == "INTERFACE_CORRECTION"): self.INTERFACE_CORRECTION = float(value)
            if (raw == "T_MODEL"): self.T_MODEL = int(value)
            if (raw == "T_SURFACE_TENSION"): self.T_SURFACE_TENSION = int(value)
            if (raw == "SURFACE_TENSION"): self.SURFACE_TENSION = float(value)
            if (raw == "T_TIME_STEP"): self.T_TIME_STEP = int(value)
            if (raw == "V_E"): self.V_E = float(value)
            if (raw == "V_N"): self.V_N = float(value)
            if (raw == "V_S"): self.V_S = float(value)
            if (raw == "V_W"): self.V_W = float(value)
            if (raw == "T_SURFACTANTS"): self.T_SURFACTANTS = int(value)
            if (raw == "T_TURBULENCE"): self.T_TURBULENCE = int(value)
            if (raw == "T_SOIL"): self.T_SOIL = int(value)
            if (raw == "T_HYDROSTATIC_PRESSURE"): self.T_HYDROSTATIC_PRESSURE = int(value)
            if (raw == "T_STRAIN_TENSOR"): self.T_STRAIN_TENSOR = int(value)
            if (raw == "T_DISPERSED_PHASE_FLUID"): self.T_DISPERSED_PHASE_FLUID = int(value)
            if (raw == "N_DISPERSED_PHASE_FLUID"): self.N_DISPERSED_PHASE_FLUID = int(value)

        
        self.H = 0.5 * self.XCV / self.NXC
        self.NC = self.NXC * self.NYC
        self.I_H = 1.0 / self.H
        self.DH = 0.01 * self.H
        self.KNORM = (1.0/3.1415) * pow(self.I_H, 2.0)
        self.GKNORM = (1.0/3.1415) * pow(self.I_H, 4.0)
        self.DR = self.H / self.HDR

        self.NX = int(self.XCV/self.DR)
        self.NY = int(self.YCV/self.DR)

    def __read_particles_from_xml__(self, filename):
        xmldoc = minidom.parse(filename)
        parameterList = xmldoc.getElementsByTagName('field')
        for s in parameterList:
            name = s.attributes['name'].value
            values = s.childNodes[0].nodeValue
            if (name == "id"):
                self.id = []
                for value in values.split():
                    self.id.append(int(value))
            if (name == "phase-id"):
                self.phaseId = []
                for value in values.split():
                    self.phaseId.append(int(value))
            if (name == "phase-type"):
                self.phaseType = []
                for value in values.split():
                    self.phaseType.append(int(value))
            if (name == "x-position"):
                self.x = []
                for value in values.split():
                    self.x.append(float(value))
            if (name == "y-position"):
                self.y = []
                for value in values.split():
                    self.y.append(float(value))
            if (name == "x-velocity"):
                self.u = []
                for value in values.split():
                    self.u.append(float(value))
            if (name == "y-velocity"):
                self.v = []
                for value in values.split():
                    self.v.append(float(value))
            if (name == "smoothing-length"):
                self.sl = []
                for value in values.split():
                    self.sl.append(float(value))
            if (name == "mass"):
                self.m = []
                for value in values.split():
                    self.m.append(float(value))
            if (name == "pressure"):
                self.p = []
                for value in values.split():
                    self.p.append(float(value))
            if (name == "density"):
                self.d = []
                for value in values.split():
                    self.d.append(float(value))
            if (name == "initial-density"):
                self.di = []
                for value in values.split():
                    self.di.append(float(value))
            if (name == "volume"):
                self.o = []
                for value in values.split():
                    self.o.append(float(value))
            if (name == "dynamic-viscosity"):
                self.mi = []
                for value in values.split():
                    self.mi.append(float(value))
            if (name == "gamma"):
                self.gamma = []
                for value in values.split():
                    self.gamma.append(float(value))
            if (name == "sound-speed"):
                self.s = []
                for value in values.split():
                    self.s.append(float(value))
            if (name == "equation-of-state-coefficient"):
                self.b = []
                for value in values.split():
                    self.b.append(float(value))
            if (name == "color-function"):
                self.c = []
                for value in values.split():
                    self.c.append(float(value))
            if (name == "smoothed-color-function"):
                self.cs = []
                for value in values.split():
                    self.cs.append(float(value))
            if (name == "x-normal-vector"):
                self.nx = []
                for value in values.split():
                    self.nx.append(float(value))
            if (name == "y-normal-vector"):
                self.ny = []
                for value in values.split():
                    self.ny.append(float(value))
            if (name == "normal-vector-norm"):
                self.nz = []
                for value in values.split():
                    self.nz.append(float(value))
            if (name == "curvature-influence-indicator"):
                self.na = []
                for value in values.split():
                    self.na.append(int(value))
            if (name == "curvature"):
                self.cu = []
                for value in values.split():
                    self.cu.append(float(value))
            if (name == "x-surface-tension"):
                self.stx = []
                for value in values.split():
                    self.stx.append(float(value))
            if (name == "y-surface-tension"):
                self.sty = []
                for value in values.split():
                    self.sty.append(float(value))
            if (name == "bulk-surfactant-mass"):
                self.mBulk = []
                for value in values.split():
                    self.mBulk.append(float(value))
            if (name == "bulk-surfactant-concentration"):
                self.cBulk = []
                for value in values.split():
                    self.cBulk.append(float(value))
            if (name == "bulk-surfactant-duffusion-coefficient"):
                self.dBulk = []
                for value in values.split():
                    self.dBulk.append(float(value))
            if (name == "surfactant-mass"):
                self.mSurf = []
                for value in values.split():
                    self.mSurf.append(float(value))
            if (name == "surfactant-concentration"):
                self.cSurf = []
                for value in values.split():
                    self.cSurf.append(float(value))
            if (name == "surfactant-diffusion-coefficient"):
                self.dSurf = []
                for value in values.split():
                    self.dSurf.append(float(value))
            if (name == "interface-area"):
                self.a = []
                for value in values.split():
                    self.a.append(float(value))
            if (name == "x-surfactant-concentration-gradient"):
                self.cSurfGradX = []
                for value in values.split():
                    self.cSurfGradX.append(float(value))
            if (name == "y-surfactant-concentration-gradient"):
                self.cSurfGradY = []
                for value in values.split():
                    self.cSurfGradY.append(float(value))
            if (name == "strain-rate"):
                self.str = []
                for value in values.split():
                    self.str.append(float(value))
            if (name == "turbulent-viscosity/soil-viscosity"):
                self.nut = []
                for value in values.split():
                    self.nut.append(float(value))
            if (name == "hydrostatic-pressure"):
                self.ph = []
                for value in values.split():
                    self.ph.append(float(value))
            if (name == "x-position-dispersed-phase-fluid"):
                self.pdpfX = []
                for value in values.split():
                    self.pdpfX.append(float(value))
            if (name == "y-position-dispersed-phase-fluid"):
                self.pdpfY = []
                for value in values.split():
                    self.pdpfY.append(float(value))
            if (name == "x-velocity-dispersed-phase-fluid"):
                self.pdpfU = []
                for value in values.split():
                    self.pdpfU.append(float(value))
            if (name == "y-velocity-dispersed-phase-fluid"):
                self.pdpfV = []
                for value in values.split():
                    self.pdpfV.append(float(value))
            if (name == "density-dispersed-phase-fluid"):
                self.pdpfD = []
                for value in values.split():
                    self.pdpfD.append(float(value))
            if (name == "initial-density-dispersed-phase-fluid"):
                self.pdpfDI = []
                for value in values.split():
                    self.pdpfDI.append(float(value))
            if (name == "volume-fraction-dispersed-phase-fluid"):
                self.pdpfO = []
                for value in values.split():
                    self.pdpfO.append(float(value))
            if (name == "mass-dispersed-phase-fluid"):
                self.pdpfM = []
                for value in values.split():
                    self.pdpfM.append(float(value))
            if (name == "pressure-dispersed-phase-fluid"):
                self.pdpfP = []
                for value in values.split():
                    self.pdpfP.append(float(value))
        
        self.nu = []
        for mi, di in zip(self.mi, self.di):           
            self.nu.append(mi/di)

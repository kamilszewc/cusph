/*
*  domain.cpp
*
*  Author: Kamil Szewc (kamil.szewc@gmail.com)
*  Modified on: 27-09-2014
*
*/

#define _CRT_SECURE_NO_WARNINGS

#include "domain.h"
#include <random>
#include <string>
#include <fstream>
#include "input.h"
#include "output.h"


Domain::Domain(const char *filename) : _outputDirectory("results/"), _time(0.0)
{
	InitDefaultParameters(2.0);
	read_parameters_from_xml(filename, &_par);

	try
	{
		_p = std::vector<Particle>(_par.N);

		if (_par.T_DISPERSED_PHASE > 0)
		{
			_pDispersedPhase = std::vector<ParticleDispersedPhase>(_par.N_DISPERSED_PHASE);
		}
	}
	catch(std::exception& ex)
	{
		std::cerr << "Exception bad_alloc caught: " << ex.what() << std::endl;
		exit(EXIT_FAILURE);
	}

	_par.NX = static_cast<int>(_par.XCV/_par.DR);
	_par.NY = static_cast<int>(_par.YCV/_par.DR);
	_par.NZ = static_cast<int>(_par.ZCV/_par.DR);

	SetParticles();
	read_particles_from_xml_file(filename, _p, &_par);
	if (_par.T_DISPERSED_PHASE > 0)
	{
		read_particles_dispersed_phase_from_xml_file(filename, _pDispersedPhase, &_par);
	}
}

Domain::Domain(int CASE, double HDR, int NXC, int NYC, int NZC) : _outputDirectory("results/"), _time(0.0)
{
	InitDefaultGeometry(CASE);
	_par.NXC = NXC;
	_par.NYC = NYC;
	_par.NZC = NZC;
	CheckConsistency();
	SetupDomain(CASE, HDR);
}

Domain::Domain(int CASE, double HDR, double H) : _outputDirectory("results/"), _time(0.0)
{
	InitDefaultGeometry(CASE);
	_par.H = H;
	_par.NXC = static_cast<int>(0.5 * _par.XCV / _par.H);
	_par.NYC = static_cast<int>(0.5 * _par.YCV / _par.H);
	_par.NZC = static_cast<int>(0.5 * _par.ZCV / _par.H);
	CheckConsistency();
	SetupDomain(CASE, HDR);
}

Domain::Domain(int CASE, double HDR, int MIN_NC) : _outputDirectory("results/"), _time(0.0)
{
	InitDefaultGeometry(CASE);
	if ( (_par.XCV <= _par.YCV) && (_par.XCV <= _par.ZCV) ) 
	{
		_par.NXC = MIN_NC;
		_par.NYC = static_cast<int>(_par.YCV / _par.XCV) * MIN_NC;
		_par.NZC = static_cast<int>(_par.ZCV / _par.XCV) * MIN_NC;
	}
	else if( (_par.YCV < _par.XCV) && (_par.YCV <= _par.ZCV) )
	{
		_par.NYC = MIN_NC;
		_par.NXC = static_cast<int>(_par.XCV / _par.YCV) * MIN_NC;
		_par.NZC = static_cast<int>(_par.ZCV / _par.YCV) * MIN_NC;
	}
	else 
	{
		_par.NZC = MIN_NC;
		_par.NXC = static_cast<int>(_par.XCV / _par.ZCV) * MIN_NC;
		_par.NYC = static_cast<int>(_par.YCV / _par.ZCV) * MIN_NC;
	}
	CheckConsistency();
	SetupDomain(CASE, HDR);
}

void Domain::SetupDomain(int CASE, double HDR)
{
	_par.NC = _par.NXC * _par.NYC * _par.NZC;

	InitDefaultParameters(HDR);

	try
	{
		_p = std::vector<Particle>(_par.N);
	}
	catch(std::bad_alloc& ex)
	{
		std::cerr << "Exception bad_alloc caught: " << ex.what() << std::endl;
		exit(EXIT_FAILURE);
	}

	InitCase(CASE);
}

void Domain::InitDefaultGeometry(int CASE)
{
	switch (CASE) {
	case 0: _par.XCV = 1.0; _par.YCV = 1.0; _par.ZCV = 1.0; break; // Static box
	case 1: _par.XCV = 1.0; _par.YCV = 1.0; _par.ZCV = 1.0; break; // Nothing
	case 2: _par.XCV = 1.0; _par.YCV = 1.0; _par.ZCV = 1.0; break; // Lid-driven
	case 3: _par.XCV = 1.0; _par.YCV = 1.0; _par.ZCV = 1.0; break; // Cube-to-droplet deformation
	case 4: _par.XCV = 8.0; _par.YCV = 8.0; _par.ZCV = 8.0; break; // Static box
	case 5: _par.XCV = 8.0; _par.YCV = 8.0; _par.ZCV = 16.0; break; // In-line coalescence
	case 6: _par.XCV = 0.005; _par.YCV = 0.005; _par.ZCV = 0.01; break; // Side-by-side coalescence
	case 7: _par.XCV = 8.0; _par.YCV = 4.0; _par.ZCV = 16.0; break;
	case 8: _par.XCV = 6.0; _par.YCV = 3.0; _par.ZCV = 3.0; break; // Side-by-side coalescence, no gravity
	case 9: _par.XCV = 1.0; _par.YCV = 1.0; _par.ZCV = 1.0; break; // Dam-breaking
	case 10: _par.XCV = 1.0; _par.YCV = 1.0; _par.ZCV = 2.0; break; // Static box with hydrostatic pressure
	case 12: _par.XCV = 1.0; _par.YCV = 1.0; _par.ZCV = 1.0; break; // Wave
	case 13: _par.XCV = 1.0; _par.YCV = 1.0; _par.ZCV = 1.0; break; // Drop deformation in shear flow
	case 14: _par.XCV = 1.0; _par.YCV = 1.0; _par.ZCV = 1.0; break; // Drop deformation in shear flow - high density ratio
	case 20: _par.XCV = 40.0; _par.YCV = 40.0; _par.ZCV = 10.0; break; // Soil
	case 25: _par.XCV = 1.0; _par.YCV = 1.0; _par.ZCV = 1.0; break; // Single solid particle
	case 26: _par.XCV = 1.0; _par.YCV = 1.0; _par.ZCV = 1.0; break; // Single solid particle
	default:
		std::cerr << "Undefined case no. " << CASE << std::endl;
		exit(EXIT_FAILURE);
	}
}

void Domain::InitDefaultParameters(double HDR)
{
	_par.HDR = HDR;
	_par.T_MODEL = 2;
	_par.V_N = 0.0; _par.V_E = 0.0; _par.V_S = 0.0; _par.V_W = 0.0; _par.V_T = 0.0; _par.V_B = 0.0;
	_par.G_X = 0.0; _par.G_Y = 0.0; _par.G_Z = 0.0;
	_par.T_BOUNDARY_PERIODICITY = 0;
	_par.DT = 0.002;
	_par.END_TIME = 10.0;
	_par.INTERVAL_TIME = _par.DT * 100.0;
	_par.T_TIME_STEP = 0;
	_par.T_INTERFACE_CORRECTION = 0;
	_par.INTERFACE_CORRECTION = 0.0;
	_par.T_SURFACE_TENSION = 0;
	_par.SURFACE_TENSION = 0.0;
	_par.T_NORMAL_VECTOR = 1;
	_par.T_NORMAL_VECTOR_TRESHOLD = 1;
	_par.T_RENORMALIZE_PRESSURE = 0;
	_par.T_XSPH = 0;
	_par.XSPH = 0.0;
	_par.T_DISPERSED_PHASE = 0;
	_par.N_DISPERSED_PHASE = 0;
	_par.T_TURBULENCE = 0;
	_par.T_SOIL = 0;
	_par.SOIL_COHESION = 0.0;
	_par.SOIL_INTERNAL_ANGLE = 0.51;
	_par.SOIL_MINIMAL_VISCOSITY = 0.01;
	_par.SOIL_MAXIMAL_VISCOSITY = 2000.0;
	_par.T_STRAIN_RATE = 0;
	_par.T_HYDROSTATIC_PRESSURE = 0;
	_par.T_SOLID_PARTICLE = 0;
	_par.T_SMOOTHING_DENSITY = 0;
	_par.H = 0.5 * _par.XCV / (double)_par.NXC;
	_par.I_H = 1.0 / _par.H;
	_par.DH = 0.01*_par.H;
	_par.KNORM = M_1_PI * pow3(_par.I_H);
	_par.GKNORM = M_1_PI * pow5(_par.I_H);
	_par.DR = _par.H / HDR;
	_par.NX = static_cast<int>(_par.XCV / _par.DR);
	_par.NY = static_cast<int>(_par.YCV / _par.DR);
	_par.NZ = static_cast<int>(_par.ZCV / _par.DR);
	_par.N = _par.NX * _par.NY * _par.NZ;
}

void Domain::InitCase(int CASE)
{
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dist(0.0, 1.0);

	switch (CASE) {
	case 0: // Static box
		_par.T_BOUNDARY_PERIODICITY = 1;
		_par.DT = 0.000025;
		_par.INTERVAL_TIME = 0.5;
		_par.END_TIME = 5.50051;
		SetParticles();
		break;
	case 1: // Lid-driven cavity
		_par.V_T = 1.0;
		_par.END_TIME = 12.0;
		_par.DT = 0.00005;
		_par.INTERVAL_TIME = 0.5;
		SetParticles();
		break;
	case 2: // Lid-driven cavity Re=1000
		_par.V_T = 1.0;
		_par.END_TIME = 10.0;
		_par.DT = 0.0005;
		_par.INTERVAL_TIME = 0.1;
		_par.T_STRAIN_RATE = 0;
		_par.T_TURBULENCE = 1;
		SetParticles();
		for (auto& p : _p)
		{
			p.nu = 0.01;
			p.mi = 0.01;
			p.phaseId = 1;
		}
		break;
	case 3: // Cube-to-droplet deformation
		_par.END_TIME = 1.0;
		_par.DT = 0.00025;
		_par.INTERVAL_TIME = 0.02;
		_par.T_SURFACE_TENSION = 1;
		_par.SURFACE_TENSION = 1.0;
		_par.T_NORMAL_VECTOR = 2;
		_par.T_NORMAL_VECTOR_TRESHOLD = 1;
		SetParticles();
		for (auto& p : _p) {
			if ( (p.pos.x > 0.2) && (p.pos.x < 0.8) &&
				 (p.pos.y > 0.2) && (p.pos.y < 0.8) &&
				 (p.pos.z > 0.2) && (p.pos.z < 0.8) )
			{
					p.phaseId = 1;
					p.c = 1.0;
			}
			p.m = _par.XCV * _par.YCV * _par.ZCV * p.d / _par.N;
			p.s = 15.0;
			p.b = pow2(p.s) * p.di / p.gamma;
			p.mi = 0.2; p.nu = 0.2;
		}
		break;
	case 4: // Static box
		_par.END_TIME = 10.0;
		_par.DT = 0.00025;
		_par.NOUT = 1000;
		_par.INTERVAL_TIME = 0.1;
		_par.G_Z = -1.0;
		SetParticles();
		for (auto& p : _p)
		{
			p.phaseId = 1;
			p.d = 1.0;
			p.di = 1.0;
			p.m = _par.XCV * _par.YCV * _par.ZCV * p.d / _par.N;
			p.c = 1.0;
			p.s = 10.0;
			p.mi = 0.202610504;
			p.nu = p.mi / p.d;
			p.gamma = 7.0;
			p.b = _p[0].s*_p[0].s*_p[0].di/_p[0].gamma;
			p.m = p.m*pow(1.0 + p.di*(_par.ZCV-p.pos.z)/p.b, 1.0/p.gamma);
		}
		break;
	case 5: // In-line coalescence
		_par.END_TIME = 10.0;
		_par.DT = 0.00025;
		_par.NOUT = 1000;
		_par.INTERVAL_TIME = 0.05;
		_par.T_SURFACE_TENSION = 2;
		_par.SURFACE_TENSION = 0.0344827586;
		_par.T_INTERFACE_CORRECTION = 1;
		_par.INTERFACE_CORRECTION = 1.2;
		_par.G_Z = -1.0;
		SetParticles();
		for (auto& p : _p)
		{
			if ( pow2(p.pos.x-4.0) + pow2(p.pos.y-4.0) + pow2(p.pos.z-3.0) < 1.0)
			{
				p.phaseId = 1;
				p.d = 0.001;
				p.di = 0.001;
				p.c = 1.0;
				p.s = 141.421356237;
				p.mi = 0.202610504/100.0;
				p.gamma = 1.4;
			}
			else if (pow2(p.pos.x-4.0) + pow2(p.pos.y-4.0) + pow2(p.pos.z-6.0) < 1.0)
			{
				p.phaseId = 2;
				p.d = 0.001;
				p.di = 0.001;
				p.c = 1.0;
				p.s = 141.421356237;
				p.mi = 0.202610504/100.0;
				p.gamma = 1.4;
			}
			else
			{
				p.phaseId = 0;
				p.d = 1.0;
				p.di = 1.0;
				p.c = 0.0;
				p.s = 10.0;
				p.mi = 0.202610504;
				p.gamma = 7.0;
			}
			p.nu = p.mi / p.d;
			p.m = _par.XCV * _par.YCV * _par.ZCV * p.d/_par.N;
			p.b = _p[0].s*_p[0].s*_p[0].di/_p[0].gamma;

			if (pow2(p.pos.x-4.0) + pow2(p.pos.y-4.0) + pow2(p.pos.z-0.3) < 1.0)
			{
				p.m = p.m*pow(1.0 + (_par.ZCV-0.3)/p.b, 1.0/1.4);
			}
			else if (pow2(p.pos.x-4.0) + pow2(p.pos.y-4.0) + pow2(p.pos.z-0.6) < 1.0)
			{
				p.m = p.m*pow(1.0 + (_par.ZCV-0.6)/p.b, 1.0/1.4);
			}
			else
			{
				p.m = p.m*pow(1.0 + p.di*(_par.ZCV-p.pos.z)/p.b, 1.0/p.gamma);
			}
		}
		break;
	case 6: // Side-by-side coalescence
		_par.END_TIME = 0.08;
		_par.DT = 0.0000001;
		_par.NOUT = 1000;
		_par.INTERVAL_TIME = 0.0001;
		_par.T_SURFACE_TENSION = 2;
		_par.SURFACE_TENSION = 0.0169;
		_par.T_INTERFACE_CORRECTION = 1;
		_par.INTERFACE_CORRECTION = 0.0005;
		_par.G_Z = -9.81;
		SetParticles();
		for (auto& p : _p)
		{
			if ( pow2(p.pos.x-0.0025-0.0009-0.0002) + pow2(p.pos.y-0.0025) + pow2(p.pos.z-0.0025) < pow2(0.0009) )
			{
				p.phaseId = 1;
				p.d = 0.711;
				p.di = 0.711;
				p.c = 1.0;
				p.s = 141.421356237;
				p.mi = 0.000001/100.0;
				p.gamma = 1.4;
			}
			else if ( pow2(p.pos.x-0.0025+0.0009+0.0002) + pow2(p.pos.y-0.0025) + pow2(p.pos.z-0.0025) < pow2(0.0009) )
			{
				p.phaseId = 2;
				p.d = 0.711;
				p.di = 0.711;
				p.c = 1.0;
				p.s = 141.421356237;
				p.mi = 0.000001/100.0;
				p.gamma = 1.4;
			} 
			else 
			{
				p.phaseId = 0;
				p.d = 817.0;
				p.di = 817.0;
				p.c = 0.0;
				p.s = 10.0;
				p.mi = 0.000001;
				p.gamma = 7.0;
			}
			p.nu = p.mi / p.d;
			p.m = _par.XCV * _par.YCV * _par.ZCV * p.d / _par.N;
			p.b = _p[0].s*_p[0].s*_p[0].di/_p[0].gamma;

			if (pow2(p.pos.x-0.0025-0.0009-0.0002) + pow2(p.pos.y-0.0025) + pow2(p.pos.z-0.0025) < pow2(0.0009))
			{
				p.m = p.m*pow(1.0 + (0.01-0.0025)/p.b, 1.0/1.4);
			}
			else if (pow2(p.pos.x-0.0025+0.0009+0.0002) + pow2(p.pos.y-0.0025) + pow2(p.pos.z-0.0025) < pow2(0.0009))
			{
				p.m = p.m*pow(1.0 + (0.01-0.0025)/p.b, 1.0/1.4);
			} 
			else 
			{
				p.m = p.m*pow(1.0 + p.di*(0.01-p.pos.z)/p.b, 1.0/p.gamma);
			}

		}
		break;
	case 7:
		_par.END_TIME = 10.0;
		_par.DT = 0.00025;
		_par.NOUT = 1000;
		_par.INTERVAL_TIME = 0.05;
		_par.T_SURFACE_TENSION = 2;
		_par.SURFACE_TENSION = 0.0344827586;
		_par.T_INTERFACE_CORRECTION = 1;
		_par.INTERFACE_CORRECTION = 0.6;
		_par.G_Z = -1.0;
		SetParticles();
		for (auto& p : _p)
		{
			double r = _par.XCV / 16.0;
			if ( pow2(p.pos.x-0.5*_par.XCV+2.0*r) + pow2(p.pos.y-0.5*_par.YCV) + pow2(p.pos.z-_par.ZCV/16.0) < pow2(r) )
			{
				p.phaseId = 1;
				p.d = 0.001;
				p.di = 0.001;
				p.c = 1.0;
				p.s = 141.421356237;
				p.mi = 0.202610504/100.0;
				p.gamma = 1.4;
			}
			else if (pow2(p.pos.x-0.5*_par.XCV-2.0*r) + pow2(p.pos.y-0.5*_par.YCV) + pow2(p.pos.z-_par.ZCV/16.0) < pow2(r) )
			{
				p.phaseId = 2;
				p.d = 0.001;
				p.di = 0.001;
				p.c = 1.0;
				p.s = 141.421356237;
				p.mi = 0.202610504/100.0;
				p.gamma = 1.4;
			}
			else
			{
				p.phaseId = 0;
				p.d = 1.0;
				p.di = 1.0;
				p.c = 0.0;
				p.s = 10.0;
				p.mi = 0.202610504;
			p.gamma = 7.0;
			}
			p.nu = p.mi / p.d;
			p.m = _par.XCV * _par.YCV * _par.ZCV * p.d / _par.N;
			p.b = _p[0].s*_p[0].s*_p[0].di/_p[0].gamma;

			if ( pow2(p.pos.x-0.5*_par.XCV+2.0*r) + pow2(p.pos.y-0.5*_par.YCV) + pow2(p.pos.z-_par.ZCV/16.0) < pow2(r) )
			{
				p.m = p.m*pow(1.0 + (_par.ZCV-_par.ZCV/16.0)/p.b, 1.0/1.4);
			}
			else if ( pow2(p.pos.x-0.5*_par.XCV-2.0*r) + pow2(p.pos.y-0.5*_par.YCV) + pow2(p.pos.z-_par.ZCV/16.0) < pow2(r) )
			{
				p.m = p.m*pow(1.0 + (_par.ZCV-_par.ZCV/16.0)/p.b, 1.0/1.4);
			}
			else
			{
				p.m = p.m*pow(1.0 + p.di*(_par.ZCV-p.pos.z)/p.b, 1.0/p.gamma);
			}
		}
		break;
	case 8: // Side-by-side coalescence, no gravity
		_par.T_BOUNDARY_PERIODICITY = 1;
		_par.END_TIME = 20.0;
		_par.DT = 0.0001;
		_par.NOUT = 1000;
		_par.INTERVAL_TIME = 0.005;
		_par.T_SURFACE_TENSION = 2;
		//_par.SURFACE_TENSION = 0.0344827586;
		_par.SURFACE_TENSION = 0.1;
		//_par.T_INTERFACE_CORRECTION = 1;
		//_par.INTERFACE_CORRECTION = 0.6;
		SetParticles();
		for (auto& p : _p)
		{
			double r = 0.5;
			if ( pow2(p.pos.x-0.5*_par.XCV+1.25*r) + pow2(p.pos.y-0.5*_par.YCV) + pow2(p.pos.z-0.5*_par.ZCV) < pow2(r) )
			{
				p.phaseId = 1;
				p.d = 0.001;
				p.di = 0.001;
				p.c = 1.0;
				p.s = 141.421356237;
				p.mi = 0.202610504/100.0;
				p.gamma = 1.4;
			}
			else if (pow2(p.pos.x-0.5*_par.XCV-1.25*r) + pow2(p.pos.y-0.5*_par.YCV) + pow2(p.pos.z-0.5*_par.ZCV) < pow2(r) )
			{
				p.phaseId = 2;
				p.d = 0.001;
				p.di = 0.001;
				p.c = 1.0;
				p.s = 141.421356237;
				p.mi = 0.202610504/100.0;
				p.gamma = 1.4;
			}
			else
			{
				p.phaseId = 0;
				p.d = 1.0;
				p.di = 1.0;
				p.c = 0.0;
				p.s = 10.0;
				p.mi = 0.202610504;
				p.gamma = 7.0;
			}
			p.nu = p.mi / p.d;
			p.m = _par.XCV * _par.YCV * _par.ZCV * p.d / _par.N;
			p.b = _p[0].s*_p[0].s*_p[0].di/_p[0].gamma;
		}
		break;
	case 9: // Dam-breaking
		_par.DT = 0.00002;
		_par.INTERVAL_TIME = 0.1;
		_par.END_TIME = 5.0;
		_par.G_Z = -1.0;
		SetParticles();
		{
			std::vector<Particle> p;
			for (int i=0; i<_par.N; i++)
			{
				_p[i].nu = 0.05;
				_p[i].mi = 0.05;
				if ( (_p[i].pos.x < 0.4) && (_p[i].pos.y > 0.6) && (_p[i].pos.z < 0.8) )
				{
					p.push_back(_p[i]);
				}
			}
			_p = p;
			_par.N = p.size();
		}
		break;
	case 10: // Static box with hydrostatic pressure
		//_par.T_BOUNDARY_PERIODICITY = 2;
		_par.G_Z = -1.0;
		_par.DT = 0.000025;
		_par.INTERVAL_TIME = 0.1;
		_par.END_TIME = 8.0;
		SetParticles();
		{
			bool isDispersedParticleSet = true;
			for (auto& p : _p)
			{
				p.nu = 1.0 / 100.0;
				p.mi = p.nu * p.d;
				p.m = _par.XCV * _par.YCV * _par.ZCV * p.d / _par.N;
				p.s = 15.0;
				p.gamma = 7.0;
				p.b = pow2(p.s) * p.di / p.gamma;
				//p.m = p.m * pow(1.0 + p.d*(2.0 - p.pos.y) / p.b, 1.0 / 7.0);
				p.d = p.d * pow(1.0 + p.d*(2.0 - p.pos.z) / p.b, 1.0 / 7.0);
				if ((isDispersedParticleSet == false) && (p.pos.x >= 0.5) && (p.pos.y >= 0.5) && (p.pos.z >= 1.5))
				{
					p.d = 10.0;
					p.di = 10.0;
					p.m = _par.XCV * _par.YCV * _par.ZCV * p.d / _par.N;
					p.b = _p[0].b;
					isDispersedParticleSet = true;
				}
			}
		}
		break;
	
	case 12: // Wave
		_par.DT = 0.00002;
		_par.T_BOUNDARY_PERIODICITY = 2;
		_par.INTERVAL_TIME = 0.1;
		_par.END_TIME = 5.0;
		_par.G_Z = -1.0;
		SetParticles();
		{
			std::vector<Particle> p;
			for (int i = 0; i<_par.N; i++)
			{
				_p[i].nu = 0.05;
				_p[i].mi = 0.05;
				if ( 0.5 + 0.1*sin(2.0*M_PI*_p[i].pos.x)*sin(2.0*M_PI*_p[i].pos.y) > _p[i].pos.z )
				{
					p.push_back(_p[i]);
				}
			}
			_p = p;
			_par.N = p.size();
			for (auto& p : _p)
			{
				p.d = p.d * pow(1.0 + p.d*(0.5 + 0.1*sin(2.0*M_PI*p.pos.x)*sin(2.0*M_PI*p.pos.y) - p.pos.z) / p.b, 1.0 / 7.0);
			}
		}
		break;
	case 13: // Drop deformation in shear flow
		_par.V_T = 1.0;
		_par.V_B = -1.0;
		_par.T_BOUNDARY_PERIODICITY = 2;
		_par.T_SURFACE_TENSION = 1;
		_par.SURFACE_TENSION = 0.25;
		_par.DT = 0.00005;
		_par.INTERVAL_TIME = 0.2;
		_par.END_TIME = 5.0;
		SetParticles();
		for (auto& p : _p)
		{
			p.d = 1.0;
			p.di = 1.0;
			if (pow2(p.pos.x - 0.5) + pow2(p.pos.y - 0.5) + pow2(p.pos.z - 0.5) < pow2(0.25))
			{
				p.phaseId = 1;
				p.c = 1.0;
				p.d = 1.0;
				p.di = 1.0;
				p.nu = 0.25;
				p.mi = 0.25;
			}
			else
			{
				p.phaseId = 0;
				p.c = 0.0;
				p.d = 1.0;
				p.di = 1.0;
				p.nu = 0.25;
				p.mi = 0.25;
			}
			p.s = 20.0;
			p.gamma = 7.0;
			p.b = pow2(_p[0].s) * _p[0].di / _p[0].gamma;
		}
		break;
	case 14: // Drop deformation in shear flow - high density ratio
		_par.V_T = 1.0;
		_par.V_B = -1.0;
		_par.T_BOUNDARY_PERIODICITY = 2;
		_par.T_SURFACE_TENSION = 1;
		_par.SURFACE_TENSION = 1.0/2.0;
		_par.DT = 0.0000025;
		_par.INTERVAL_TIME = 0.01;
		_par.END_TIME = 5.0;
		SetParticles();
		for (auto& p : _p)
		{
			if (pow2(p.pos.x - 0.5) + pow2(p.pos.y - 0.5) + pow2(p.pos.z - 0.5) < pow2(0.25))
			{
				p.phaseId = 1;
				p.d = 0.001;
				p.di = 0.001;
				p.c = 1.0;
				p.s = 141.42;
				p.nu = 100.0*0.5;
				p.gamma = 1.4;
			}
			else
			{
				p.phaseId = 0;
				p.d = 1.0;
				p.di = 1.0;
				p.c = 0.0;
				p.s = 10.0;
				p.nu = 0.5;
				p.gamma = 7.0;
			}
			p.m = _par.XCV * _par.YCV * _par.ZCV * p.d / _par.N;
			p.mi = p.nu * p.d;
			p.b = _p[0].s*_p[0].s*_p[0].di / _p[0].gamma;
		}
		break;
	case 20: // Soil
		_par.G_Z = -981.0;
		_par.DT = 0.000001;
		_par.INTERVAL_TIME = 0.005;
		_par.END_TIME = 0.25;
		_par.T_SOIL = 1;
		_par.SOIL_COHESION = 0.0;
		_par.SOIL_INTERNAL_ANGLE = 0.51;
		_par.SOIL_MINIMAL_VISCOSITY = 0.001;
		_par.SOIL_MAXIMAL_VISCOSITY = 2000.0;
		_par.T_STRAIN_RATE = 0;
		SetParticles();
		{
			real a = 0.35f;
			real r = 9.7f;

			std::vector<Particle> p;
			for (int i = 0; i < _par.N; i++)
			{
				//if (((pow2(_p[i].pos.x - _par.XCV*0.5) + pow2(_p[i].pos.y - _par.YCV*0.5) < pow2(r)) && (_p[i].pos.z < a*r + 2.0)) || (_p[i].pos.z < 2.0))
				if ((pow2(_p[i].pos.x - _par.XCV*0.5) + pow2(_p[i].pos.y - _par.YCV*0.5) < pow2(r)) && (_p[i].pos.z < a*r))
				{
					p.push_back(_p[i]);
				}
			}
			_p = p;
			_par.N = p.size();

			for (int i = 0; i < _par.N; i++)
			{
				_p[i].phaseId = 1;
				_p[i].phaseType = 1;
				_p[i].c = 1.0;
				_p[i].d = 2.6;
				_p[i].di = 2.6;
				_p[i].mi = 0.0;
				_p[i].nu = _p[i].mi / _p[i].d;
				_p[i].m = _par.XCV * _par.YCV * _par.ZCV * _p[i].d / (_par.NX * _par.NY * _par.NZ);
				_p[i].s = 1000.0;
				_p[i].gamma = 7.0;
				_p[i].b = pow2(_p[i].s) * _p[i].di / _p[i].gamma;

				//if (_p[i].pos.z < 2.0)
				//{
				//	_p[i].m = _p[i].m * pow(1.0 + _p[i].di*fabs(_par.G_Z)*(a*r - _p[i].pos.z) / _p[i].b, 1.0 / 7.0);
				//	_p[i].d = _p[i].di * pow(1.0 + _p[i].di*fabs(_par.G_Z)*(a*r - _p[i].pos.z) / _p[i].b, 1.0 / 7.0);
				//}
				//else
				//{
				//	_p[i].m = _p[i].m * pow(1.0 + _p[i].di*fabs(_par.G_Z)*(2.0 - _p[i].pos.z) / _p[i].b, 1.0 / 7.0);
				//	_p[i].d = _p[i].di * pow(1.0 + _p[i].di*fabs(_par.G_Z)*(2.0 - _p[i].pos.z) / _p[i].b, 1.0 / 7.0);
				//}
				//if (_p[i].pos.z < 2.0)
				//{
				//	_p[i].phaseId = 1;
				//	_p[i].c = 0.0;
				//}
			}
		}
		break;
	case 25: // Single solid particle
		//_par.T_BOUNDARY_PERIODICITY = 2;
		_par.G_Z = -1.0;
		_par.DT = 0.00002;
		_par.INTERVAL_TIME = 0.005;
		_par.END_TIME = 10.0;
		//_par.T_SOLID_PARTICLE = 1;
		SetParticles();
		{
			for (auto& p : _p)
			{
				p.d = 1.0;
				p.di = 1.0;
				p.c = 0.0;
				p.phaseId = 0;
				p.nu = 1.0 / 10.0;
				p.mi = p.nu * p.d;
				p.m = _par.XCV * _par.YCV * _par.ZCV * p.d / _par.N;
				p.s = 20.0;
				p.gamma = 7.0;
				p.b = pow2(p.s) * p.di / p.gamma;

				static bool isSetUp = false;
				if ( (pow2(p.pos.x - 0.5) + pow2(p.pos.y - 0.5) + pow2(p.pos.z - 0.75) < pow2(0.02)) && (isSetUp == false))
				{
					//std::cout << p.id << std::endl;
					p.phaseId = 1;
					p.c = 1.0;
					//p.phaseType = 2;
					p.d = 8.0;
					p.di = 8.0;
					p.nu = 1.0/10.0;
					p.mi = p.nu * p.d;
					p.m = _par.XCV * _par.YCV * _par.ZCV * p.d / _par.N;
					isSetUp = true;
				}
				else
				{
					p.m = p.m * pow(1.0 + (1.0 - p.pos.z) / p.b, 1.0 / 7.0);
					p.d = p.d * pow(1.0 + (1.0 - p.pos.z) / p.b, 1.0 / 7.0);
				}
				//p.m = p.m * pow(1.0 + p.d*(2.0 - p.pos.y) / p.b, 1.0 / 7.0);
				//p.d = p.d * pow(1.0 + (2.0 - p.pos.z) / p.b, 1.0 / 7.0);
			}
		}
		break;
	case 26: // Dispersed phase
		_par.G_Z = -1.0;
		_par.DT = 0.00002;
		_par.INTERVAL_TIME = 0.005;
		_par.END_TIME = 2.0;
		SetParticles();
		_par.T_DISPERSED_PHASE = 1;
		_par.N_DISPERSED_PHASE = 1;
		_pDispersedPhase = std::vector<ParticleDispersedPhase>(_par.N_DISPERSED_PHASE);
		for (auto& p : _p)
		{
			p.nu = 1.0 / 10.0;
			p.mi = p.nu * p.d;
			p.s = 20.0;
			p.gamma = 7.0;
			p.b = pow2(p.s) * p.di / p.gamma;
			p.m = p.m * pow(1.0 + (1.0 - p.pos.z) / p.b, 1.0 / 7.0);
			p.d = p.d * pow(1.0 + (1.0 - p.pos.z) / p.b, 1.0 / 7.0);
		}
		for (auto& pd : _pDispersedPhase)
		{
			pd.pos.x = 0.5;
			pd.pos.y = 0.5;
			pd.pos.z = 0.75;
			pd.vel.x = 0.0;
			pd.vel.y = 0.0;
			pd.vel.z = 0.0;
			pd.d = 8.0;
			pd.dia = 0.04f;
			pd.dFl = 0.0;
			pd.miFl = 0.0;
			pd.velFl.x = 0.0;
			pd.velFl.y = 0.0;
		}
		break;
	default:
		break;
	}

}


void Domain::SetParticles()
{
	double dx = _par.XCV / _par.NX;
	double dy = _par.YCV / _par.NY;
	double dz = _par.ZCV / _par.NZ;
	double x0 = 0.5 * dx;
	double y0 = 0.5 * dy;
	double z0 = 0.5 * dz;
	int l = 0;
	for (int i = 0; i < _par.NX; i++) 
	{
		for (int j = 0; j < _par.NY; j++) 
		{
			for (int k = 0; k < _par.NZ; k++) 
			{
				_p[l].id = l; _p[l].phaseId = 0; _p[l].phaseType = 0;
				_p[l].pos.x = x0 + dx * static_cast<double>(i); _p[l].rh_pos.x = 0.0;
				_p[l].pos.y = y0 + dx * static_cast<double>(j); _p[l].rh_pos.y = 0.0;
				_p[l].pos.z = z0 + dx * static_cast<double>(k); _p[l].rh_pos.z = 0.0;
				_p[l].vel.x = 0.0; _p[l].rh_vel.x = 0.0;
				_p[l].vel.y = 0.0; _p[l].rh_vel.y = 0.0;
				_p[l].vel.z = 0.0; _p[l].rh_vel.z = 0.0;
				_p[l].m = 1.0 / static_cast<double>(_par.N);
				_p[l].p = 0.0; _p[l].ph = 0.0;
				_p[l].d = 1.0; _p[l].di = 1.0; _p[l].rh_d = 0.0;
				_p[l].nu = 0.01; _p[l].mi = 0.01;
				_p[l].str = 0.0; _p[l].nut = 0.0;
				_p[l].gamma = 7.0; _p[l].s = 10.0;
				_p[l].b = _p[l].s * _p[l].s * _p[l].di / _p[l].gamma;
				_p[l].o = pow2(_p[l].m / _p[l].d);
				_p[l].c = 0.0;
				_p[l].n.x = 0.0; _p[l].n.y = 0.0; _p[l].n.z = 0.0; _p[l].n.w = 0.0;
				_p[l].na = 0;
				_p[l].cu = 0.0;
				_p[l].st.x = 0.0; _p[l].st.y = 0.0; _p[l].st.z = 0.0;
				_p[l].cs = 0.0;
				_p[l].cw = 0.0;
				l++;
			}
		}
	}
}

std::vector<Particle>* Domain::GetParticles()
{
	return &_p;
}

std::vector<ParticleDispersedPhase>* Domain::GetParticlesDispersedPhase()
{
	return &_pDispersedPhase;
}

Parameters* Domain::GetParameters()
{
	return &_par;
}

void Domain::SetModel(int MODEL)
{
	_par.T_MODEL = MODEL;
}

double* Domain::GetTime()
{
	return &_time;
}

const char* Domain::GetOutputDirectory()
{
	return _outputDirectory;
}

void Domain::SetOutputDirectory(const char* outputDirectory)
{
	_outputDirectory = outputDirectory;
}

void Domain::CheckConsistency()
{
	if (!IsConsistentWithSearchAlgorithm())
		{
			std::cerr << "NXC or NYC is not consistent with search algorithm." << std::endl;
			std::cerr << "Due to the GPU efficieNYC the search algorithm requires NXC" << std::endl;
			std::cerr << "and NYC to be power of 2." << std::endl;
			std::cerr << "This limitation is planned to be removed in newer releases." << std::endl;
			exit(EXIT_FAILURE);
		}
		if (!IsConsistentWithGeometry())
		{
			std::cerr << "Domain is geometrically not consistent." << std::endl;
			exit(EXIT_FAILURE);
		}
}

bool Domain::IsConsistentWithGeometry()
{
	return true;
}

bool Domain::IsConsistentWithSearchAlgorithm()
{
	if ((_par.NXC % 2 == 0) && (_par.NYC % 2 == 0) && (_par.NZC % 2 == 0))
	{
		return true;
	}
	else
	{
		return false;
	}
}

void Domain::WriteToFile(const char* filename, FileFormat fileFormat)
{
	write_to_file(filename, _p, _pDispersedPhase, &_par, fileFormat);
}

void Domain::WriteToFile(FileFormat fileFormat)
{
	std::stringstream filenameStream;
	filenameStream << _outputDirectory << std::fixed << _time;
	if (fileFormat == FileFormat::SPH) filenameStream << ".sph";
	else if (fileFormat == FileFormat::XML) filenameStream << ".xml";
	else exit(1);
	std::string filenameString = filenameStream.str();
	WriteToFile(filenameString.c_str(), fileFormat);
}

void Domain::WriteToFile()
{
	WriteToFile(FileFormat::SPH);
}


void Domain::WritePhaseDataToRawFile(const char* filename, int phaseId)
{
	write_raw_phase_data(filename, phaseId, _p, &_par);
}

void Domain::WritePhaseDataToRawFile(int phaseId)
{
	std::stringstream filenameStream;
	filenameStream << _outputDirectory << std::fixed << _time <<  "_ph_" << phaseId <<".dat";
	std::string filenameString = filenameStream.str();
	WritePhaseDataToRawFile(filenameString.c_str(), phaseId);
}

void Domain::WritePhasesToRawFile(const char* filename)
{
	write_raw_phases_data(filename, _p, &_par);
}

void Domain::WritePhasesToRawFile()
{
	std::stringstream filenameStream;
	filenameStream << _outputDirectory << std::fixed << _time << "_ph.dat";
	std::string filenameString = filenameStream.str();
	WritePhasesToRawFile(filenameString.c_str());
}

void Domain::WritePhasesToRawFiles()
{
	/*std::vector<int> phaseIds(1);
	for (Particle& p : _p)
	{
		bool isPhaseNew = true;
		for (int& phaseId : phaseIds)
		{
			if (p.phaseId == phaseId)
			{
				isPhaseNew = false;
			}
		}
	}*/

	/*std::stringstream filenameStream;
	filenameStream << _outputDirectory << std::fixed << _time << "_ph.dat";
	std::string filenameString = filenameStream.str();
	WritePhasesToRawFile(filenameString.c_str());*/
}

double Domain::GetAndWriteKinetic(const char *filename)
{
	double kinetic = 0.0;
	double c = 0.0;
	for (auto& p : _p)
	{
		{ // Checking if nan occures
			if ( (p.vel.x != p.vel.x) || (p.vel.y != p.vel.y) || (p.vel.z != p.vel.z) )
			{
				std::cerr << "(domain.cpp) Nan in velocity. Particle " << p.id << "." << std::endl;
			}
			if (p.d != p.d)
			{
				std::cerr << "(domain.cpp) Nan in density. Particle " << p.id << "." << std::endl;
			}
		}
		
		double value = 0.5 * p.m * ( pow2(p.vel.x) + pow2(p.vel.y) + pow2(p.vel.z) );
		double y = value - c;
		double t = kinetic + y;
		c = (t-kinetic) - y;
		kinetic = t;
	}

	std::ofstream file(filename, std::ios::app);

	if (file.is_open())
	{
		file << _time << " " << kinetic << std::endl;
		file.close();
	}
	else
	{
		std::cerr << "Output kinetic error: " << filename << std::endl;
		exit(EXIT_FAILURE);
	}

	return kinetic;
}

double Domain::GetAndWriteKinetic()
{
	std::stringstream filenameStream;
	filenameStream << _outputDirectory << "kinetic.dat";
	std::string filenameString = filenameStream.str();
	return GetAndWriteKinetic(filenameString.c_str());
}

int Domain::GetSizeOfParticles()
{
	return sizeof(Particle)*_par.N;
}

int Domain::GetSizeOfParameters()
{
	return sizeof(Parameters);
}

Domain::~Domain()
{}

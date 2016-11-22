/*
*  domain.cpp
*
*  Author: Kamil Szewc (kamil.szewc@gmail.com)
*  Modified on: 20-05-2015
*
*/

#define _CRT_SECURE_NO_WARNINGS

#include "domain.h"
#include <random>
#include <string>
#include <fstream>
#include "input.h"
#include "output.h"


Domain::Domain(char *filename) : _outputDirectory("results/"), _time(0.0)
{
	InitDefaultParameters(2.0);
	read_parameters_from_xml(filename, &_par);

	try
	{
		_p = std::vector<Particle>(_par.N);

		if (_par.T_DISPERSED_PHASE != 0)
		{
			_pDispersedPhase = std::vector<ParticleDispersedPhase>(_par.N_DISPERSED_PHASE);
		}
		else
		{
			_pDispersedPhase = std::vector<ParticleDispersedPhase>(1);
		}

		if (_par.T_DISPERSED_PHASE_FLUID != 0)
		{
			_pDispersedPhaseFluid = std::vector<Particle>(_par.N_DISPERSED_PHASE_FLUID);
		}
		else
		{
			_pDispersedPhaseFluid = std::vector<Particle>(1);
		}
	}
	catch(std::exception& ex)
	{
		std::cerr << "Exception bad_alloc caught: " << ex.what() << std::endl;
		exit(EXIT_FAILURE);
	}
	
	SetParticles(_p);
	read_particles_from_xml_file(filename, _p, &_par);
	if (_par.T_DISPERSED_PHASE != 0)
	{
		read_particles_dispersed_phase_from_xml_file(filename, _pDispersedPhase, &_par);
	}
	//_phasesXmlInformation = read_phases_from_xml(filename, &_par);
}

Domain::Domain(int CASE, double HDR, int NXC, int NYC) : _outputDirectory("results/"), _time(0.0)
{
	InitDefaultGeometry(CASE);
	_par.NXC = NXC;
	_par.NYC = NYC;
	CheckConsistency();
	SetupDomain(CASE, HDR);
}

Domain::Domain(int CASE, double HDR, double H) : _outputDirectory("results/"), _time(0.0)
{
	InitDefaultGeometry(CASE);
	_par.H = H;
	_par.NXC = static_cast<int>( (0.5 * _par.XCV) / _par.H);
	_par.NYC = static_cast<int>( (0.5 * _par.YCV) / _par.H);
	CheckConsistency();
	SetupDomain(CASE, HDR);
}

Domain::Domain(int CASE, double HDR, int MIN_NC) : _outputDirectory("results/"), _time(0.0)
{
	InitDefaultGeometry(CASE);
	if (_par.XCV > _par.YCV) {
		_par.NYC = MIN_NC;
		_par.NXC = static_cast<int>( (_par.XCV + 0.25*_par.YCV) / _par.YCV) * MIN_NC;
	}
	else {
		_par.NXC = MIN_NC;
		_par.NYC = static_cast<int>( (_par.YCV + 0.25*_par.XCV) / _par.XCV) * MIN_NC;
	}
	CheckConsistency();
	SetupDomain(CASE, HDR);
}

void Domain::SetupDomain(int CASE, double HDR)
{
	_par.NC = _par.NXC * _par.NYC;

	InitDefaultParameters(CASE, HDR);

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

	std::ostringstream stream;
	stream << "    <phases>\n";
	stream << "    </phases>";
	_phasesXmlInformation = stream.str();
}


void Domain::InitDefaultGeometry(int CASE)
{
	switch (CASE) {
	case 0: _par.XCV = 1.0; _par.YCV = 1.0; break; // Static box
	case 1: _par.XCV = 1.0; _par.YCV = 1.0; break; // Poiseuille flows
	case 2: _par.XCV = 2.0; _par.YCV = 1.0; break; // Couette flow
	case 3: _par.XCV = 1.0; _par.YCV = 1.0; break; // Taylor-Green vortex
	case 4: _par.XCV = 1.0; _par.YCV = 1.0; break; // Lid-driven cavity Re=1000
	case 5: _par.XCV = 1.0; _par.YCV = 2.0; break; // Rayleigh-Taylor instability
	case 6: _par.XCV = 1.0; _par.YCV = 1.0; break; // Square-droplet deformation
	case 7: _par.XCV = 50.0; _par.YCV = 50.0; break; // Square-droplet deformation no outer phase
	case 8: _par.XCV = 50.0; _par.YCV = 50.0; break; // Square-droplet deformation with outer phase
	case 9: _par.XCV = 1.0; _par.YCV = 1.0; break; // Square-droplet deformation high density ratio
	case 10: _par.XCV = 6.0; _par.YCV = 12.0; break; // Bubble rising in water
	case 11: _par.XCV = 8.0; _par.YCV = 8.0; break; // Bubble-bubble coalescence (no gravity, only CSF)
	case 12: _par.XCV = 1.0; _par.YCV = 1.0; break; // Drop deformation in shear flow
	case 13: _par.XCV = 1.0; _par.YCV = 1.0; break; // Drop deformation in shear flow - high density ratio
	case 14: _par.XCV = 1.0; _par.YCV = 1.0; break; // Bubble deformation in shear flow - high density ratio
	case 16: _par.XCV = 1.0; _par.YCV = 1.0; break; // Dispersed phase in static box
	case 17: _par.XCV = 0.01; _par.YCV = 0.001; break; // Taylor bubble
	case 20: _par.XCV = 0.1; _par.YCV = 0.1; break; // Surfactants - bulk diffusion
	case 21: _par.XCV = 0.1; _par.YCV = 0.1; break; // Surfactants - bulk diffusion
	case 22: _par.XCV = 0.1; _par.YCV = 0.1; break; // Surfactants - surface diffusion straight
	case 23: _par.XCV = 0.01; _par.YCV = 0.01; break; // Surfactants - surface diffusion droplet
	case 24: _par.XCV = 0.01; _par.YCV = 0.01; break; // Surfactants - surface diffusion droplet with surfae tension
	case 25: _par.XCV = 0.01; _par.YCV = 0.01; break; // Surfactants - surface diffusion droplet with surfae tension
	case 30: _par.XCV = 4.0; _par.YCV = 4.0; break; // Single solid particle
	case 31: _par.XCV = 4.0; _par.YCV = 8.0; break; // Square in fluid
	case 40: _par.XCV = 1.0; _par.YCV = 1.0; break; // Rotation of circle
	case 50: _par.XCV = 2.0; _par.YCV = 2.0; break;
	case 80: _par.XCV = 50.0; _par.YCV = 10.0; break; // Axisymmetric collapse of granular column (bylo 50, 10)
	case 81: _par.XCV = 50.0; _par.YCV = 10.0; break; // Axisymmetric collapse of granular column
	case 82: _par.XCV = 200.0; _par.YCV = 100.0; break; // Soil in water
	case 83: _par.XCV = 200.0; _par.YCV = 100.0; break; // Soil in water
	case 84: _par.XCV = 2.0; _par.YCV = 0.25; break; // Fraccarollo and Capart (2002) 
	case 85: _par.XCV = 8.0; _par.YCV = 16.0; break; // Falling sand
	case 86: _par.XCV = 60.0; _par.YCV = 20.0; break; // Falling sand 2
	case 100: _par.XCV = 1.0; _par.YCV = 1.0; break; // Dispersed phase in static box
	case 101: _par.XCV = 1.0; _par.YCV = 1.0; break; // Dispersed phase in static box
	case 102: _par.XCV = 3.0; _par.YCV = 1.0; break; // Dispersed phase in static box
	case 103: _par.XCV = 2.0; _par.YCV = 1.0; break; // Dispersed phase in static box
	case 199: _par.XCV = 1.0; _par.YCV = 1.0; break; // Periodic box with wavy pattern
	case 200: _par.XCV = 8.0; _par.YCV = 1.0; break; // Experiment with wave maker
	case 201: _par.XCV = 8.0; _par.YCV = 1.0; break; // Experiment with wave maker and send
	case 300: _par.XCV = 1.0; _par.YCV = 2.0; break;
	default:
		std::cerr << "Undefined case no. " << CASE << std::endl;
		exit(EXIT_FAILURE);
	}
}

void Domain::InitDefaultParameters(int CASE, double HDR)
{
	_par.CASE = CASE;
	_par.HDR = HDR;
	_par.T_MODEL = 2;
	_par.V_N = 0.0; _par.V_E = 0.0; _par.V_S = 0.0; _par.V_W = 0.0;
	_par.G_X = 0.0; _par.G_Y = 0.0;
	_par.T_BOUNDARY_PERIODICITY = 0;
	_par.DT = 0.0005;
	_par.END_TIME = 10.0;
	_par.INTERVAL_TIME = 0.1;
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
	_par.T_SURFACTANTS = 0;
	_par.T_TURBULENCE = 0;
	_par.T_SOIL = 0;
	_par.SOIL_COHESION = 0.0;
	_par.SOIL_INTERNAL_ANGLE = 0.51;
	_par.SOIL_MINIMAL_VISCOSITY = 0.01;
	_par.SOIL_MAXIMAL_VISCOSITY = 2000.0;
	_par.T_HYDROSTATIC_PRESSURE = 0;
	_par.T_STRAIN_TENSOR = 0;
	_par.T_SMOOTHING_DENSITY = 0;
	_par.T_VARIABLE_H = 0;
	_par.T_DISPERSED_PHASE_FLUID = 0;
	_par.N_DISPERSED_PHASE_FLUID = 0;
	_par.H = 0.5 * _par.XCV / (double)_par.NXC;
	_par.I_H = 1.0 / _par.H;
	_par.DH = 0.01*_par.H;
	_par.DR = _par.H / HDR;
	_par.NX = (int)(_par.XCV / _par.DR);
	_par.NY = (int)(_par.YCV / _par.DR);
	_par.N = _par.NX * _par.NY;
}

void Domain::InitDefaultParameters(double HDR)
{
	this->InitDefaultParameters(-1, HDR);
}

void Domain::InitCase(int CASE)
{
	switch (CASE) {
	case 0: // Static box
		_par.T_BOUNDARY_PERIODICITY = 1;
		SetParticles(_p);
		break;
	case 1: // Poisseuille flow
		_par.T_BOUNDARY_PERIODICITY = 2;
		_par.G_X = 1.0;
		_par.DT = 0.000025;
		_par.INTERVAL_TIME = 0.1;
		SetParticles(_p);
		for (auto& p : _p)
		{
			p.nu = 1.0;
			p.mi = 1.0;
			p.h = _par.H;
		}
		break;
	case 2: //Couette flow
		_par.V_N = 1.0;
		_par.G_X = 0.0;
		_par.T_BOUNDARY_PERIODICITY = 2;
		_par.DT = 0.00005;
		_par.INTERVAL_TIME = 0.1;
		_par.END_TIME = 5.0;
		SetParticles(_p);
		for (auto& p : _p)
		{
			p.nu = 0.1;
			p.mi = 0.1;
			p.s = 20.0;
			p.gamma = 7.0;
			p.b = pow2(p.s) * p.di / p.gamma;
			p.h = _par.H;
		}
		break;
	case 3: // Taylor-Green Vortex
		_par.T_BOUNDARY_PERIODICITY = 1;
		_par.DT = 0.00001;
		_par.INTERVAL_TIME = 0.1;
		SetParticles(_p);
		for (auto& p : _p) {
			p.nu = 0.01;
			p.mi = 0.01;
			p.vel.x = -cos(2.0*M_PI*p.pos.x)*sin(2.0*M_PI*p.pos.y);
			p.vel.y = sin(2.0*M_PI*p.pos.x)*cos(2.0*M_PI*p.pos.y);
		}
		break;
	case 4: // Lid-driven cavity Re=1000
		_par.T_TIME_STEP = 0;
		_par.T_BOUNDARY_PERIODICITY = 0;
		_par.T_RENORMALIZE_PRESSURE = 0;
		_par.V_N = 1.0;
		_par.END_TIME = 50.0;
		_par.DT = 0.00005;
		_par.INTERVAL_TIME = 0.1;
		_par.T_TURBULENCE = 0;
		_par.T_VARIABLE_H = 1;
		SetParticles(_p);
		for (auto& p : _p)
		{
			p.nu = 0.001;
			p.mi = 0.001;
			p.h = _par.H;
			p.s = 10.0;
			p.gamma = 7.0;
			p.b = pow2(p.s) * p.di / p.gamma;
		}
		break;
	case 5: // Rayleigh-Taylor instability
		_par.END_TIME = 5.0;
		_par.DT = 0.000025;
		_par.INTERVAL_TIME = 0.1;
		_par.G_Y = -1.0;
		SetParticles(_p);
		for (auto& p : _p)
		{
			if (p.pos.y > (1.0 - 0.15*sin(2.0*M_PI*p.pos.x))) {
				p.phaseId = 1;
				p.d = 1.8;
				p.di = 1.8;
				p.c = 1.0;
			}
			else
			{
				p.phaseId = 0;
				p.d = 1.0;
				p.di = 1.0;
				p.c = 0.0;
			}
			p.nu = 1.0 / 420.0;
			p.mi = p.nu * p.d;
			p.m = _par.XCV * _par.YCV * p.d / _par.N;
			p.s = 15.0;
			p.gamma = 7.0;
			p.b = pow2(p.s) * p.di / p.gamma;
			p.m = p.m * pow(1.0 + 1.8*(2.0 - p.pos.y) / p.b, 1.0 / 7.0);
			p.d = p.d * pow(1.0 + 1.8*(2.0 - p.pos.y) / p.b, 1.0 / 7.0);
		}
		break;
	case 6: // Square-droplet deformation
		_par.T_BOUNDARY_PERIODICITY = 0;
		_par.END_TIME = 1.0;
		_par.DT = 0.00001;
		_par.INTERVAL_TIME = 0.02;
		_par.T_SURFACE_TENSION = 1;
		_par.SURFACE_TENSION = 1.0;
		_par.T_NORMAL_VECTOR = 1;
		_par.T_NORMAL_VECTOR_TRESHOLD = 1;
		SetParticles(_p);
		for (auto& p : _p)
		{
			if ((p.pos.x > 0.2) && (p.pos.x < 0.8) && (p.pos.y > 0.2) && (p.pos.y < 0.8))
			{
				p.phaseId = 1;
				p.c = 1.0;
				p.d = 0.1;
				p.di = 0.1;
			}
			else
			{
				p.phaseId = 0;
				p.c = 0.0;
				p.d = 1.0;
				p.di = 1.0;
			}
			p.m = _par.XCV * _par.YCV * p.d / _par.N;
			p.s = 15.0;
			p.b = pow2(p.s) *p.di / p.gamma;
			p.mi = 0.2;
			p.nu = 0.2;
		}
		for (auto& p : _p)
		{
			p.b = _p[0].b;
		}
		break;
	case 7: // Square-droplet deformation no outer phase
		_par.END_TIME = 200.0;
		_par.T_BOUNDARY_PERIODICITY = 1;
		_par.DT = 0.0001;
		_par.INTERVAL_TIME = 0.5;
		_par.T_SURFACE_TENSION = 3;
		_par.SURFACE_TENSION = 1.0;
		SetParticles(_p);
		for (auto& p : _p)
		{
			p.phaseId = 0;
			p.c = 1.0;
			p.d = 1.95;
			//_p[i].di = 1.99;
			p.m = _par.XCV * _par.YCV * p.d / _par.N;
			//_p[i].m = 1.15;
			p.s = 15.0;
			p.b = pow2(p.s) * p.di / p.gamma;
			p.mi = 2.5; p.nu = 2.5;
		}
		for (int i = 0; i < _par.N; i++)
		{
			if ((_p[i].pos.x < 50.0*0.3) || (_p[i].pos.x > 50.0*0.7)
				|| (_p[i].pos.y < 50.0*0.3) || (_p[i].pos.y > 50.0*0.7))
			{
				_p.erase(_p.begin() + i);
				i--;
			}
		}
		break;
	case 8: // Square-droplet deformation with outer phase
		_par.END_TIME = 25.0;
		_par.T_BOUNDARY_PERIODICITY = 1;
		_par.DT = 0.0001;
		_par.INTERVAL_TIME = 0.5;
		_par.T_SURFACE_TENSION = 3;
		_par.SURFACE_TENSION = 1.0;
		SetParticles(_p);
		for (auto& p : _p)
		{
			p.phaseId = 0;
			p.c = 0.0;
			p.d = 1.95;
			//_p[i].di = 1.99;
			p.m = _par.XCV * _par.YCV * p.d / _par.N;
			//_p[i].m = 1.15;
			p.s = 15.0;
			p.b = pow2(p.s) * p.di / p.gamma;
			p.mi = 2.5; p.nu = 2.5;
			if ((p.pos.x < 50.0*0.3) || (p.pos.x > 50.0*0.7)
				|| (p.pos.y < 50.0*0.3) || (p.pos.y > 50.0*0.7))
			{
				p.phaseId = 1;
				p.c = 1.0;
			}
		}
		break;
	case 9: //Square-droplet deformation high density ratio
		_par.END_TIME = 1.0;
		_par.DT = 0.0000005;
		_par.INTERVAL_TIME = 0.005;
		_par.T_SURFACE_TENSION = 1;
		_par.SURFACE_TENSION = 0.1;
		_par.T_INTERFACE_CORRECTION = 0;
		_par.INTERFACE_CORRECTION = 0.0;
		SetParticles(_p);
		for (auto& p : _p) {
			if (((p.pos.x > 0.2) && (p.pos.x < 0.8) && (p.pos.y > 0.2) && (p.pos.y < 0.8))) {
				p.phaseId = 1;
				p.d = 0.001;
				p.di = 0.001;
				p.c = 1.0;
				p.s = 400.0;
				p.nu = 128.0*sqrt(8.0) / 1000.0;
				p.gamma = 1.4;
			}
			else
			{
				p.phaseId = 0;
				p.d = 1.0;
				p.di = 1.0;
				p.c = 0.0;
				p.s = 28.28;
				p.nu = sqrt(8.0) / 1000.0;
				p.gamma = 7.0;
			}
			p.m = _par.XCV * _par.YCV * p.d / _par.N;
			p.mi = p.nu * p.d;
			p.b = _p[0].s*_p[0].s*_p[0].di / _p[0].gamma;
		}
		break;
	case 10: // Bubble rising in water
		_par.END_TIME = 6.0;
		_par.DT = 0.000025;
		_par.INTERVAL_TIME = 0.1;
		_par.T_SURFACE_TENSION = 1;
		_par.SURFACE_TENSION = 0.02;
		_par.T_INTERFACE_CORRECTION = 0;
		_par.INTERFACE_CORRECTION = 0.0;
		_par.G_Y = -1.0;
		_par.T_RENORMALIZE_PRESSURE = 0;
		_par.T_STRAIN_TENSOR = 1;
		_par.T_XSPH = 0;
		_par.XSPH = 0.0;
		SetParticles(_p);
		for (auto& p : _p) {
			if (pow2(p.pos.x - 3.0) + pow2(p.pos.y - 2.0) < 1.0) {
				p.phaseId = 1;
				p.d = 0.001;
				p.di = 0.001;
				p.c = 1.0;
				p.s = 400.0;
				//p.nu = 100.0*sqrt(8.0) / 5.0;
				p.nu = 128.0*sqrt(8.0) / 1000.0;
				//p.nu = 128.0*sqrt(8.0) / 100.0;
				//p.nu = 128.0*sqrt(8.0) / 10.0;
				p.gamma = 1.4;
			}
			else
			{
				p.phaseId = 0;
				p.d = 1.0;
				p.di = 1.0;
				p.c = 0.0;
				p.s = 28.28;
				//p.nu = sqrt(8.0) / 5.0;
				p.nu = sqrt(8.0) / 1000.0;
				//p.nu = sqrt(8.0) / 100.0;
				//p.nu = sqrt(8.0) / 10.0;
				p.gamma = 7.0;
			}
			p.m = _par.XCV * _par.YCV * p.d / _par.N;
			p.mi = p.nu * p.d;
			p.b = _p[0].s*_p[0].s*_p[0].di / _p[0].gamma;

			if (pow2(p.pos.x - 3.0) + pow2(p.pos.y - 2.0) < 1.0)
			{
				p.m = p.m * pow(1.0 + 10.0 / p.b, 1.0 / 1.4);
			}
			else
			{
				p.m = p.m * pow(1.0 + (12.0 - p.pos.y) / p.b, 1.0 / 7.0);
			}
		}
		break;
	case 11: // Bubble-bubble coalescence (no gravity, only CSF)
		_par.T_BOUNDARY_PERIODICITY = 1;
		_par.END_TIME = 25.0;
		_par.DT = 0.00005;
		_par.INTERVAL_TIME = 0.05;
		_par.T_SURFACE_TENSION = 1;
		_par.SURFACE_TENSION = 0.5;
		_par.T_INTERFACE_CORRECTION = 1;
		_par.G_Y = 0.0;
		_par.T_RENORMALIZE_PRESSURE = 0;
		_par.T_XSPH = 0;
		_par.XSPH = 0.0;
		SetParticles(_p);
		for (auto& p : _p) {
			if (pow2(p.pos.x - 2.5) + pow2(p.pos.y - 4.0) < 1.0)
			{
				p.phaseId = 1;
				p.d = 0.001;
				p.di = 0.001;
				p.c = 1.0;
				p.s = 400.0;
				p.nu = 100.0*sqrt(8.0) / 5.0;
				p.gamma = 1.4;
			}
			else if (pow2(p.pos.x - 5.5) + pow2(p.pos.y - 4.0) < 1.0)
			{
				p.phaseId = 2;
				p.d = 0.001;
				p.di = 0.001;
				p.c = 1.0;
				p.s = 400.0;
				p.nu = 100.0*sqrt(8.0) / 5.0;
				p.gamma = 1.4;
			}
			else
			{
				p.phaseId = 0;
				p.d = 1.0;
				p.di = 1.0;
				p.c = 0.0;
				p.s = 28.28;
				p.nu = sqrt(8.0) / 5.0;
				p.gamma = 7.0;
			}
			p.m = _par.XCV * _par.YCV * p.d / _par.N;
			p.mi = p.nu * p.d;
			p.b = _p[0].s*_p[0].s*_p[0].di / _p[0].gamma;
		}
		break;
	case 12: // Drop deformation in shear flow
		_par.V_N = 1.0;
		_par.V_S = -1.0;
		_par.T_BOUNDARY_PERIODICITY = 2;
		_par.T_SURFACE_TENSION = 1;
		_par.SURFACE_TENSION = 0.1666666;
		_par.DT = 0.00005;
		_par.INTERVAL_TIME = 0.2;
		_par.END_TIME = 5.0;
		SetParticles(_p);
		for (auto& p : _p)
		{
			p.d = 1.0;
			p.di = 1.0;
			if (pow2(p.pos.x - 0.5) + pow2(p.pos.y - 0.5) > pow2(0.25))
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
	case 13: // Drop deformation in shear flow - high density ratio
		_par.V_N = 1.0;
		_par.V_S = -1.0;
		_par.T_BOUNDARY_PERIODICITY = 2;
		_par.T_SURFACE_TENSION = 1;
		_par.SURFACE_TENSION = 0.0625;
		_par.DT = 0.0000025;
		_par.INTERVAL_TIME = 0.05;
		_par.END_TIME = 5.0;
		SetParticles(_p);
		for (auto& p : _p)
		{
			if (pow2(p.pos.x - 0.5) + pow2(p.pos.y - 0.5) < pow2(0.25))
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
			p.m = _par.XCV * _par.YCV * p.d / _par.N;
			p.mi = p.nu * p.d;
			p.b = _p[0].s*_p[0].s*_p[0].di / _p[0].gamma;
		}
		break;
	case 14: // Bubble deformation in shear flow - high density ratio
		_par.V_N = 1.0;
		_par.V_S = -1.0;
		_par.T_BOUNDARY_PERIODICITY = 2;
		_par.T_SURFACE_TENSION = 1;
		_par.SURFACE_TENSION = 0.0625;
		_par.DT = 0.0000025;
		_par.INTERVAL_TIME = 0.05;
		_par.END_TIME = 5.0;
		SetParticles(_p);
		for (auto& p : _p)
		{
			if (pow2(p.pos.x - 0.5) + pow2(p.pos.y - 0.5) < pow2(0.25))
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
			p.m = _par.XCV * _par.YCV * p.d / _par.N;
			p.mi = p.nu * p.d;
			p.b = _p[0].s*_p[0].s*_p[0].di / _p[0].gamma;
		}
		break;
	case 16: // Dispersed phase in static box
		_par.END_TIME = 0.5;
		_par.DT = 0.00001;
		_par.INTERVAL_TIME = 0.002;
		_par.G_Y = -1.0;
		SetParticles(_p);
		_par.T_DISPERSED_PHASE = 1;
		_par.N_DISPERSED_PHASE = 1;
		_pDispersedPhase = std::vector<ParticleDispersedPhase>(_par.N_DISPERSED_PHASE);
		for (auto& p : _p)
		{
			p.m = p.m * pow(1.0 + fabs(_par.G_Y)*(1.0 - p.pos.y) / p.b, 1.0 / 7.0);
		}
		for (auto& pd : _pDispersedPhase)
		{
			pd.pos.x = 0.5;
			pd.pos.y = 0.8;
			pd.vel.x = 0.0;
			pd.vel.y = 0.0;
			pd.d = 2.0;
			pd.dia = 0.04;
			pd.dFl = 0.0;
			pd.miFl = 0.0;
			pd.velFl.x = 0.0;
			pd.velFl.y = 0.0;
		}
		break;
	case 17: // Taylor bubble
		_par.T_BOUNDARY_PERIODICITY = 2;
		_par.END_TIME = 1.0;
		_par.DT = 0.0000001;
		_par.INTERVAL_TIME = 0.0001;
		_par.T_SURFACE_TENSION = 1;
		_par.SURFACE_TENSION = 0.0005;
		_par.T_INTERFACE_CORRECTION = 0;
		_par.INTERFACE_CORRECTION = 0.0;
		_par.G_X = 9.82;
		_par.T_RENORMALIZE_PRESSURE = 0;
		_par.T_STRAIN_TENSOR = 1;
		_par.T_XSPH = 0;
		_par.XSPH = 0.0;
		SetParticles(_p);

		for (auto& p : _p) {
			real dh = 0.5*_par.YCV*0.417*(1.0 - exp(-1.69*pow(0.02, 0.5025)));
			real r = 0.5*_par.YCV - dh;
			real c1 = r;
			real c2 = c1 + 0.5*_par.XCV - 2.0*r;

			if ((pow2(p.pos.x - c1) + pow2(p.pos.y - 0.5*_par.YCV) < pow2(r))
				|| (pow2(p.pos.x - c2) + pow2(p.pos.y - 0.5*_par.YCV) < pow2(r))
				|| ((p.pos.x > c1) && (p.pos.x < c2) && (p.pos.y > dh) && (p.pos.y < _par.YCV - dh)))
			{
				p.phaseId = 1;
				p.d = 1.0;
				p.di = 1.0;
				p.c = 1.0;
				p.s = 1.0;
				p.mi = 0.01 / 1000.0;
				p.gamma = 1.4;
			}
			else
			{
				p.phaseId = 0;
				p.d = 1000.0;
				p.di = 1000.0;
				p.c = 0.0;
				p.s = 1.0;
				p.mi = 0.01;
				p.gamma = 7.0;
			}
			p.m = _par.XCV * _par.YCV * p.d / _par.N;
			p.nu = p.mi / p.d;
			p.b = _p[0].s*_p[0].s*_p[0].di / _p[0].gamma;
		}
		break;
	case 20: // Surfactants - bulk diffusion
		_par.T_BOUNDARY_PERIODICITY = 1;
		_par.DT = 0.000025;
		_par.INTERVAL_TIME = 0.5;
		_par.T_SURFACTANTS = 1;
		SetParticles(_p);
		for (auto& p : _p)
		{
			p.dBulk = 4.0e-6;
			p.cBulk = exp(-0.25*pow2(p.pos.x - 0.05f) / p.dBulk);
			p.mBulk = p.cBulk * p.m / p.d;
		}
		break;
	case 21: // Surfactants - bulk diffusion
		_par.T_BOUNDARY_PERIODICITY = 0;
		//_par.DT = 0.000025;
		_par.DT = 0.000025;
		_par.INTERVAL_TIME = 0.1;
		_par.T_SURFACTANTS = 1;
		SetParticles(_p);
		for (auto& p : _p)
		{
			if (p.pos.x > 0.05)
			{
				p.dBulk = 4.0e-6;
				p.cBulk = 1.0;
				p.mBulk = p.cBulk * p.m / p.d;
			}
			else
			{
				p.dBulk = 4.0e-6;
				p.cBulk = 0.0;
				p.mBulk = 0.0;
			}
		}
		break;
	case 22: // Surfactants - surface diffusion straight
		_par.T_BOUNDARY_PERIODICITY = 2;
		_par.DT = 0.000025;
		_par.INTERVAL_TIME = 0.000025;
		_par.END_TIME = 0.00003;
		_par.T_SURFACTANTS = 1;
		SetParticles(_p);
		for (auto& p : _p)
		{
			if (p.pos.y > 0.05)
			{
				p.phaseId = 1;
				p.c = 1.0;
			}
			p.dSurf = 4.0e-6;
			p.cSurf = sin(M_PI*p.pos.x / _par.XCV);
			p.mSurf = p.cBulk * p.m / p.d;
			p.dBulk = 4.0e-6;
			p.cBulk = 1.0;
			p.mBulk = p.mBulk * p.m / p.d;
		}
		break;
	case 23: // Surfactants - surface diffusion droplet
		_par.T_BOUNDARY_PERIODICITY = 1;
		_par.DT = 0.0000025;
		_par.INTERVAL_TIME = 0.01;
		_par.END_TIME = 0.1;
		_par.T_SURFACTANTS = 1;
		SetParticles(_p);
		for (auto& p : _p)
		{
			if (pow2(p.pos.x - 0.005) + pow2(p.pos.y - 0.005) > pow2(0.0025))
			{
				p.phaseId = 1;
				p.c = 1.0;
			}
			p.dSurf = 1.0e-4;
			double phi = acos((p.pos.x - 0.005) / hypot(p.pos.x - 0.005, p.pos.y - 0.005)) * (p.pos.y / fabs(p.pos.y));
			p.cSurf = 3.0e-6 * 0.5 * (cos(phi) + 1.0);
			p.dBulk = 4.0e-6;
			p.cBulk = 1.0;
			p.mBulk = p.mBulk * p.m / p.d;
		}
		break;
	case 24: // Surfactants - surface diffusion droplet with surfae tension
		_par.T_BOUNDARY_PERIODICITY = 1;
		_par.DT = 0.0000025;
		_par.INTERVAL_TIME = 0.002;
		_par.END_TIME = 0.3;
		_par.T_SURFACTANTS = 1;
		_par.T_SURFACE_TENSION = 1;
		_par.SURFACE_TENSION = 0.0004;
		SetParticles(_p);
		for (auto& p : _p)
		{
			if (pow2(p.pos.x - 0.005) + pow2(p.pos.y - 0.005) > pow2(0.0025))
			{
				p.phaseId = 1;
				p.c = 1.0;
			}
			p.dSurf = 5.0e-4;
			double phi = acos((p.pos.x - 0.005) / hypot(p.pos.x - 0.005, p.pos.y - 0.005)) * (p.pos.y / fabs(p.pos.y));
			p.cSurf = 3.0e-6 * 0.5 * (cos(phi) + 1.0);
			p.dBulk = 4.0e-6;
			p.cBulk = 1.0;
			p.mBulk = p.mBulk * p.m / p.d;

			//p.m = _par.XCV * _par.YCV * p.d / _par.N;
			//p.s = 15.0f;
			//p.b = pow2(p.s) *p.di / p.gamma;
			//p.mi = 0.2;
			//p.nu = 0.2;
		}
		break;
	case 25: // Surfactants - surface diffusion droplet with surfae tension
		_par.T_BOUNDARY_PERIODICITY = 1;
		_par.DT = 0.0000025;
		_par.INTERVAL_TIME = 0.002;
		_par.END_TIME = 2.0;
		_par.T_SURFACTANTS = 1;
		_par.T_SURFACE_TENSION = 1;
		_par.SURFACE_TENSION = 0.0004;
		SetParticles(_p);
		for (auto& p : _p)
		{
			//if (pow2(p.pos.x - 0.005) + pow2(p.pos.y - 0.005) > pow2(0.0025))
			if ((p.pos.x > 0.002) && (p.pos.x < 0.008) && (p.pos.y > 0.002) && (p.pos.y < 0.008))
			{
				p.phaseId = 1;
				p.c = 1.0;
			}
			p.dSurf = 5.0e-4;
			double phi = acos((p.pos.x - 0.005) / hypot(p.pos.x - 0.005, p.pos.y - 0.005)) * (p.pos.y / fabs(p.pos.y));
			if ((phi > -M_PI / 4.0) && (phi < M_PI / 4.0))
			{
				p.cSurf = 3.0e-6;
			}
			else
			{
				p.cSurf = 0.0;
			}
			p.dBulk = 4.0e-6;
			p.cBulk = 1.0;
			p.mBulk = p.mBulk * p.m / p.d;

			//p.m = _par.XCV * _par.YCV * p.d / _par.N;
			//p.s = 15.0f;
			//p.b = pow2(p.s) *p.di / p.gamma;
			//p.mi = 0.2;
			//p.nu = 0.2;
		}
		break;
	case 30: // Single solid particle
		_par.G_Y = -1.0;
		_par.DT = 0.000005;
		_par.INTERVAL_TIME = 0.05;
		_par.END_TIME = 6.0;
		SetParticles(_p);
		{
			std::vector<Particle> p;
			for (int i = 0; i < _par.N; i++)
			{
				if (_p[i].pos.y < 3.5)
				{
					p.push_back(_p[i]);
				}
			}
			_p = p;
			_par.N = p.size();

			for (auto& p : _p)
			{
				p.d = 1.0;
				p.di = 1.0;
				p.c = 0.0;
				p.phaseId = 0;
				p.nu = 1.0 / 10.0;
				p.mi = p.nu * p.d;
				p.m = _par.XCV * _par.YCV * p.d / _par.N;
				p.s = 15.0;
				p.gamma = 7.0;
				p.b = pow2(p.s) * p.di / p.gamma;

				if (pow2(p.pos.x - 2.0) + pow2(p.pos.y - 2.5) < pow2(0.2))
				{
					p.phaseId = 1;
					p.c = 1.0;
					//p.phaseType = 2;
					p.d = 4.0;
					p.di = 4.0;
					p.nu = 10.0;
					p.mi = p.nu * p.d;
					p.m = _par.XCV * _par.YCV * p.d / _par.N;
				}
				else
				{
					p.d = p.d * pow(1.0 + (3.0 - p.pos.y) / p.b, 1.0 / 7.0);
				}
			}
		}
		break;
	case 31: // Square in fluid
		_par.G_Y = -5.0;
		_par.DT = 0.00002;
		_par.INTERVAL_TIME = 0.05;
		_par.END_TIME = 6.0;
		SetParticles(_p);
		{
			real x = 2.0;
			real y = 6.0;
			real a = 1.0;
			real o = 6.28;
			std::vector<Particle> p;
			for (int i = 0; i < _par.N; i++)
			{
				if ( (_p[i].pos.y < 4.0) || ( (_p[i].pos.x > x-0.5*a) && (_p[i].pos.x < x+0.5*a) && (_p[i].pos.y < y+0.5*a) && (_p[i].pos.y > y-0.5*a) ) )
				{
					p.push_back(_p[i]);
				}
			}
			_p = p;
			_par.N = p.size();

			for (auto& p : _p)
			{
				p.d = 1.0;
				p.di = 1.0;
				p.c = 0.0;
				p.phaseId = 0;
				p.nu = 1.0 / 10.0;
				p.mi = p.nu * p.d;
				p.m = _par.XCV * _par.YCV * p.d / _par.N;
				p.s = 15.0;
				p.gamma = 7.0;
				p.b = pow2(p.s) * p.di / p.gamma;

				if (p.pos.y > 4.0)
				{
					p.phaseId = 1;
					p.c = 1.0;
					//p.phaseType = 2;
					p.d = 4.0;
					p.di = 4.0;
					p.nu = 10.0;
					p.mi = p.nu * p.d;
					p.m = _par.XCV * _par.YCV * p.d / _par.N;
					p.vel.x = o * (y - p.pos.y);
					p.vel.y = -o * (x - p.pos.x);
				}
				else
				{
					p.d = p.d * pow(1.0 + fabs(_par.G_Y)*(4.0 - p.pos.y) / p.b, 1.0 / 7.0);
				}
			}
		}
		break;
	case 40: // Square in fluid
		_par.T_BOUNDARY_PERIODICITY = 1;
		_par.DT = 0.000025;
		_par.INTERVAL_TIME = 0.05;
		_par.END_TIME = 6.0;
		SetParticles(_p);
		{
			real x = 0.5;
			real y = 0.5;
			real r = 0.25;
			real o = 6.28;
			std::vector<Particle> p;
			for (int i = 0; i < _par.N; i++)
			{
				if (pow2(_p[i].pos.x - x) + pow2(_p[i].pos.y - y) < pow2(r))
				{
					p.push_back(_p[i]);
				}
			}
			_p = p;
			_par.N = p.size();

			for (auto& p : _p)
			{
				p.d = 1.0;
				p.di = 1.0;
				p.c = 0.0;
				p.phaseId = 0;
				p.nu = 0.1;
				p.mi = p.nu * p.d;
				p.m = _par.XCV * _par.YCV * p.d / _par.N;
				p.s = 15.0;
				p.gamma = 7.0;
				p.b = pow2(p.s) * p.di / p.gamma;
				p.vel.x = o * (y - p.pos.y);
				p.vel.y = -o * (x - p.pos.x);
			}
		}
		break;
	case 50:
		_par.G_Y = -1.0;
		_par.DT = 0.000025;
		_par.INTERVAL_TIME = 0.1;
		_par.END_TIME = 25.0;
		SetParticles(_p);
		{
			std::vector<Particle> p;
			for (int i = 0; i < _par.N; i++)
			{
				if (((_p[i].pos.y < 1.0) && (_p[i].pos.x < 0.5))
					|| ((_p[i].pos.y < 0.5) && (_p[i].pos.x > 0.5)))
				{
					p.push_back(_p[i]);
				}
			}
			_p = p;
			_par.N = p.size();

			for (int i = 0; i < _par.N; i++)
			{
				if ((_p[i].pos.y < 1.0) && (_p[i].pos.x < 0.5))
				{
					_p[i].phaseId = 0;
					_p[i].d = 2.0;
					_p[i].di = 2.0;
					_p[i].mi = 2.0*0.01;
				}
				else
				{
					_p[i].phaseId = 1;
					_p[i].d = 1.0;
					_p[i].di = 1.0;
					_p[i].mi = 1.0*0.01;
				}
				_p[i].nu = _p[i].mi / _p[i].d;
				_p[i].m = _par.XCV * _par.YCV * _p[i].d / (_par.NX * _par.NY);
				_p[i].s = 15.0;
				_p[i].gamma = 7.0;
				if (i == 0) _p[i].b = pow2(_p[i].s) * _p[i].di / _p[i].gamma;
				else _p[i].b = _p[0].b;
			}
		}
		break;
	case 80: // Axisymmetric collapse of granular column
		_par.G_Y = -981.0;
		_par.DT = 0.0000005;
		_par.INTERVAL_TIME = 0.01;
		_par.END_TIME = 0.6;
		_par.T_SOIL = 1;
		_par.T_HYDROSTATIC_PRESSURE = 0;
		_par.SOIL_COHESION = 0.0;
		_par.SOIL_INTERNAL_ANGLE = 0.51;
		_par.SOIL_MINIMAL_VISCOSITY = 0.01;
		_par.SOIL_MAXIMAL_VISCOSITY = 2000.0;
		SetParticles(_p);
		{
			real a = 0.55;
			real r = 9.7;

			std::vector<Particle> p;
			for (int i = 0; i < _par.N; i++)
			{
				//if (((_p[i].pos.x < _par.XCV*0.5 + r) && (_p[i].pos.x > _par.XCV*0.5 - r) && (_p[i].pos.y < a*r + 1.0))
				//	|| (_p[i].pos.y < 1.0))
				if ((_p[i].pos.x < _par.XCV*0.5 + r) && (_p[i].pos.x > _par.XCV*0.5 - r) && (_p[i].pos.y < a*r))
				{
					p.push_back(_p[i]);
				}
			}
			_p = p;
			_par.N = p.size();

			for (int i = 0; i < _par.N; i++)
			{
				_p[i].phaseId = 0;
				_p[i].phaseType = 1;
				//if (_p[i].pos.y > 1.0) _p[i].phaseType = 1;
				//else _p[i].phaseType = -1;
				_p[i].d = 2.6;
				_p[i].di = 2.6;
				_p[i].mi = 0.0;
				_p[i].nu = _p[i].mi / _p[i].d;
				_p[i].m = _par.XCV * _par.YCV * _p[i].d / (_par.NX * _par.NY);
				_p[i].s = 1000.0;
				_p[i].gamma = 7.0;
				_p[i].b = pow2(_p[i].s) * _p[i].di / _p[i].gamma;

				/*if ((_p[i].pos.x < _par.XCV*0.5 + r) && (_p[i].pos.x > _par.XCV*0.5 - r))
				{
					_p[i].m = _p[i].m * pow(1.0 + _p[i].d*fabs(_par.G_Y)*(a*r - _p[i].pos.y) / _p[i].b, 1.0 / 7.0);
					_p[i].d = _p[i].d * pow(1.0 + _p[i].d*fabs(_par.G_Y)*(a*r - _p[i].pos.y) / _p[i].b, 1.0 / 7.0);
				}
				else
				{
					_p[i].m = _p[i].m * pow(1.0 + _p[i].d*fabs(_par.G_Y)*(1.0 - _p[i].pos.y) / _p[i].b, 1.0 / 7.0);
					_p[i].d = _p[i].d * pow(1.0 + _p[i].d*fabs(_par.G_Y)*(1.0 - _p[i].pos.y) / _p[i].b, 1.0 / 7.0);
				}*/

			}
		}
		break;
	case 81: // Axisymmetric collapse of granular column
		_par.G_Y = -981.0;
		_par.DT = 0.000002;
		_par.INTERVAL_TIME = 0.01;
		_par.END_TIME = 0.40;
		_par.T_SOIL = 1;
		_par.T_HYDROSTATIC_PRESSURE = 0;
		SetParticles(_p);
		{
			real a = 0.55;
			real r = 9.7;

			std::vector<Particle> p;
			for (int i = 0; i < _par.N; i++)
			{
				if ((_p[i].pos.x < _par.XCV*0.5 + r) && (_p[i].pos.x > _par.XCV*0.5 - r) && (_p[i].pos.y < a*r))
				{
					p.push_back(_p[i]);
				}
			}
			_p = p;
			_par.N = p.size();

			for (int i = 0; i < _par.N; i++)
			{
				_p[i].phaseId = 0;
				_p[i].phaseType = 1;
				_p[i].d = 2.6;
				_p[i].di = 2.6;
				_p[i].mi = 0.0;
				_p[i].nu = _p[i].mi / _p[i].d;
				_p[i].m = _par.XCV * _par.YCV * _p[i].d / (_par.NX * _par.NY);
				_p[i].s = 500.0;
				_p[i].gamma = 7.0;
				_p[i].b = pow2(_p[i].s) * _p[i].di / _p[i].gamma;

				//_p[i].m = _p[i].m * pow(1.0 + _p[i].d*fabs(_par.G_Y)*(a*r - _p[i].pos.y) / _p[i].b, 1.0 / 7.0);
			}
		}
		break;
	case 82:  // Soil in water 
		_par.G_Y = -981.0;
		_par.DT = 0.000001;
		_par.INTERVAL_TIME = 0.01;
		_par.END_TIME = 25.0;
		_par.T_SOIL = 1;
		SetParticles(_p);
		{
			std::vector<Particle> p;
			for (int i = 0; i < _par.N; i++)
			{
				if (((_p[i].pos.y < 30.0)) || ((_p[i].pos.y < 70.0) && (_p[i].pos.x > 87.5) && (_p[i].pos.x < 112.5)))
					//((_p[i].pos.y < 1.5) && (_p[i].pos.x < 0.625))
					//|| ( && (_p[i].pos.x > 0.25)))
				{
					p.push_back(_p[i]);
				}
			}
			_p = p;
			_par.N = p.size();

			for (int i = 0; i < _par.N; i++)
			{
				if (_p[i].pos.y > 15.0)
				{
					_p[i].phaseId = 0;
					_p[i].phaseType = 0;
					_p[i].d = 1.0;
					_p[i].di = 1.0;
					_p[i].mi = 0.01;
				}
				else
				{
					_p[i].phaseId = 1;
					_p[i].phaseType = 1;
					_p[i].d = 2.0;
					_p[i].di = 2.0;
					_p[i].mi = 0.0;
				}

				_p[i].nu = _p[i].mi / _p[i].d;
				_p[i].m = _par.XCV * _par.YCV * _p[i].d / (_par.NX * _par.NY);
				_p[i].s = 500.0;
				_p[i].gamma = 7.0;
				if (i == 0) _p[i].b = pow2(_p[i].s) * _p[i].di / _p[i].gamma;
				else _p[i].b = _p[0].b;
			}
		}
		break;
	case 83:  // Soil in water 
		//_par.T_BOUNDARY_PERIODICITY = 2;
		_par.G_Y = -981.0;
		_par.DT = 0.000001;
		_par.INTERVAL_TIME = 0.005;
		_par.END_TIME = 50.0;
		_par.T_SOIL = 1;
		_par.T_TURBULENCE = 1;
		SetParticles(_p);
		{
			std::vector<Particle> p;
			for (int i = 0; i < _par.N; i++)
			{
				//if (((_p[i].pos.y < 30.0)) || ((_p[i].pos.y < 70.0) && (_p[i].pos.x > 87.5) && (_p[i].pos.x < 112.5)))
					//((_p[i].pos.y < 1.5) && (_p[i].pos.x < 0.625))
					//|| ( && (_p[i].pos.x > 0.25)))
				if (10.0*sin( (2.0*M_PI*_p[i].pos.x/_par.XCV) + 1.5*M_PI ) + 35.0 > _p[i].pos.y)
				{
					p.push_back(_p[i]);
				}
			}
			_p = p;
			_par.N = p.size();

			for (int i = 0; i < _par.N; i++)
			{
				//if (_p[i].pos.y > 15.0)
				if (2.0*sin((8.0*M_PI*_p[i].pos.x / _par.XCV) + 1.5*M_PI) + 12.0 < _p[i].pos.y)
				{
					_p[i].phaseId = 0;
					_p[i].phaseType = 0;
					_p[i].d = 1.0;
					_p[i].di = 1.0;
					_p[i].mi = 0.0025;
				}
				else
				{
					_p[i].phaseId = 1;
					_p[i].phaseType = 1;
					_p[i].d = 2.0;
					_p[i].di = 2.0;
					_p[i].mi = 0.0;
				}

				_p[i].nu = _p[i].mi / _p[i].d;
				_p[i].m = _par.XCV * _par.YCV * _p[i].d / (_par.NX * _par.NY);
				_p[i].s = 500.0;
				_p[i].gamma = 7.0;
				if (i == 0) _p[i].b = pow2(_p[i].s) * _p[i].di / _p[i].gamma;
				else _p[i].b = _p[0].b;
			}
		}
		break;
	
	case 84: // Fraccarollo and Capart (2002) 
		_par.G_Y = -9.81;
		_par.DT = 0.0000025;
		_par.INTERVAL_TIME = 0.05;
		_par.END_TIME = 1.0;
		_par.T_SOIL = 2;
		_par.SOIL_COHESION = 0.0;
		_par.SOIL_INTERNAL_ANGLE = 0.51;
		_par.SOIL_MINIMAL_VISCOSITY = 0.001;
		_par.SOIL_MAXIMAL_VISCOSITY = 3000.0;
		_par.T_TURBULENCE = 0;
		_par.T_SMOOTHING_DENSITY = 100;
		SetParticles(_p);
		{
			std::vector<Particle> p;
			for (int i = 0; i < _par.N; i++)
			{
				if (((_p[i].pos.y < 0.16) && (_p[i].pos.x < 1.0)) || ((_p[i].pos.y < 0.06) && (_p[i].pos.x < 2.0)))
				{
				p.push_back(_p[i]);
				}
			}
			_p = p;
			_par.N = p.size();
		}
		for (int i = 0; i < _par.N; i++)
		{
			if (_p[i].pos.y > 0.06)
			{
				_p[i].phaseId = 0;
				_p[i].phaseType = 0;
				_p[i].c = 0.0;
				_p[i].d = 1000.0;
				_p[i].di = 1000.0;
				_p[i].mi = 0.001;
				_p[i].gamma = 7.0;
			} 
			else
			{
				_p[i].phaseId = 1;
				_p[i].phaseType = 1;
				_p[i].c = 1.0;
				_p[i].d = 1378.0;//1540.0;  // Saturated 1378.0;
				_p[i].di = 1378.0;//1540.0;
				_p[i].mi = 0.0;
				_p[i].gamma = 7.0;
			}

			_p[i].nu = _p[i].mi / _p[i].d;
			_p[i].m = _par.XCV * _par.YCV * _p[i].d / (_par.NX * _par.NY);
			_p[i].s = 10.0;
			if (i == 0) _p[i].b = pow2(_p[i].s) * _p[i].di / _p[i].gamma;
			else _p[i].b = _p[0].b;
		}
		break;
	case 85: // Falling sand
		_par.G_Y = -981.0;
		_par.DT = 0.00000025;
		_par.INTERVAL_TIME = 0.005;
		_par.END_TIME = 0.2;
		_par.T_SOIL = 1;
		_par.SOIL_COHESION = 0.0;
		_par.SOIL_INTERNAL_ANGLE = 0.51;
		_par.SOIL_MINIMAL_VISCOSITY = 0.001;
		_par.SOIL_MAXIMAL_VISCOSITY = 5000.0;
		_par.T_SMOOTHING_DENSITY = 50;
		SetParticles(_p);
		{
			real r = 0.3;

			std::vector<Particle> p;
			bool remove = false;
			for (int i = 0; i < _par.N; i++)
			{
				if (_p[i].pos.y < 2.0)
				{
					p.push_back(_p[i]);
				}
				else
				{
					if ((_p[i].pos.x < _par.XCV*0.5 + r) && (_p[i].pos.x > _par.XCV*0.5 - r) && (_p[i].pos.y < 10.0))
					{
						//if ((_p[i].id % 2 == 0) && (remove == false)) p.push_back(_p[i]);
						/*if ((_p[i].id % 2 == 0))
						{
							if (remove == false) p.push_back(_p[i]);
							else p.push_back(_p[i - 1]);
						}*/
						p.push_back(_p[i]);
					}
				}

				if (i % (int)(_par.HDR * 2.0 * _par.NYC) == 0)
				{
					if (remove == true) remove = false;
					else remove = true;
				}
			}
			_p = p;
			_par.N = p.size();

			for (int i = 0; i < _par.N; i++)
			{
				_p[i].phaseId = 0;
				_p[i].phaseType = 1;
				_p[i].d = 2.6;
				_p[i].di = 2.6;
				_p[i].mi = 0.0;
				_p[i].nu = _p[i].mi / _p[i].d;
				_p[i].m = _par.XCV * _par.YCV * _p[i].d / (_par.NX * _par.NY);
				_p[i].s = 1000.0;
				_p[i].gamma = 7.0;
				_p[i].b = pow2(_p[i].s) * _p[i].di / _p[i].gamma;


				if (_p[i].pos.y < 1.0)
				{
					_p[i].m = _p[i].m * pow(1.0 + _p[i].d*fabs(_par.G_Y)*(1.0 - _p[i].pos.y) / _p[i].b, 1.0 / 7.0);
					_p[i].d = _p[i].d * pow(1.0 + _p[i].d*fabs(_par.G_Y)*(1.0 - _p[i].pos.y) / _p[i].b, 1.0 / 7.0);
				}

			}
		}
		break;
	case 86: // Axisymmetric collapse of granular column
		_par.G_Y = -981.0;
		_par.DT = 0.00000025;
		_par.INTERVAL_TIME = 0.01;
		_par.END_TIME = 0.3;
		_par.T_SOIL = 1;
		_par.T_HYDROSTATIC_PRESSURE = 0;
		_par.SOIL_COHESION = 0.0;
		_par.SOIL_INTERNAL_ANGLE = 0.51;
		_par.SOIL_MINIMAL_VISCOSITY = 0.01;
		_par.SOIL_MAXIMAL_VISCOSITY = 2000.0;
		SetParticles(_p);
		{
			real a = 4.0;
			real r = 2.5;

			std::vector<Particle> p;
			for (int i = 0; i < _par.N; i++)
			{
				if (((_p[i].pos.x < _par.XCV*0.5 + r) && (_p[i].pos.x > _par.XCV*0.5 - r) && (_p[i].pos.y < a*r + 1.0))
					|| (_p[i].pos.y < 1.0))
				{
					p.push_back(_p[i]);
				}
			}
			_p = p;
			_par.N = p.size();

			for (int i = 0; i < _par.N; i++)
			{
				_p[i].phaseId = 0;
				_p[i].phaseType = 1;
				_p[i].d = 2.6;
				_p[i].di = 2.6;
				_p[i].mi = 0.0;
				_p[i].nu = _p[i].mi / _p[i].d;
				_p[i].m = _par.XCV * _par.YCV * _p[i].d / (_par.NX * _par.NY);
				_p[i].s = 1000.0;
				_p[i].gamma = 7.0;
				_p[i].b = pow2(_p[i].s) * _p[i].di / _p[i].gamma;

				/*if ((_p[i].pos.x < _par.XCV*0.5 + r) && (_p[i].pos.x > _par.XCV*0.5 - r))
				{
				_p[i].m = _p[i].m * pow(1.0 + _p[i].d*fabs(_par.G_Y)*(a*r - _p[i].pos.y) / _p[i].b, 1.0 / 7.0);
				_p[i].d = _p[i].d * pow(1.0 + _p[i].d*fabs(_par.G_Y)*(a*r - _p[i].pos.y) / _p[i].b, 1.0 / 7.0);
				}
				else
				{
				_p[i].m = _p[i].m * pow(1.0 + _p[i].d*fabs(_par.G_Y)*(1.0 - _p[i].pos.y) / _p[i].b, 1.0 / 7.0);
				_p[i].d = _p[i].d * pow(1.0 + _p[i].d*fabs(_par.G_Y)*(1.0 - _p[i].pos.y) / _p[i].b, 1.0 / 7.0);
				}*/

			}
		}
		break;
	case 100: // Dispersed phase in static box
		_par.T_BOUNDARY_PERIODICITY = 2;
		_par.END_TIME = 10.0;
		_par.DT = 0.00005;
		_par.INTERVAL_TIME = 0.1;
		_par.G_Y = -1.0;
		_par.T_DISPERSED_PHASE_FLUID = 1;
		_par.N_DISPERSED_PHASE_FLUID = 1;

		SetParticles(_p);
		{
			std::vector<Particle> p;
			for (int i = 0; i < _par.N; i++)
			{
				if (_p[i].pos.y < 0.8)
				{
					p.push_back(_p[i]);
				}
			}
			_p = p;
			_par.N = p.size();
		}
		for (auto& p : _p)
		{
			p.mi = 0.1;
			p.nu = p.mi / p.d;
			//p.m = p.m * pow(1.0 + fabs(_par.G_Y)*(0.8 - p.pos.y) / p.b, 1.0 / 7.0);
			p.d = p.d * pow(1.0 + fabs(_par.G_Y)*(0.8 - p.pos.y) / p.b, 1.0 / 7.0);
		}

		_pDispersedPhaseFluid = std::vector<Particle>(_par.N_DISPERSED_PHASE_FLUID);

		for (auto& pd : _pDispersedPhaseFluid)
		{
			static int l = 0;
			SetParticleDefaultProperites(_pDispersedPhaseFluid, l);
			pd.pos.x = 0.5;
			pd.pos.y = 0.5;
			pd.vel.x = 0.0;
			pd.vel.y = 0.0;
			pd.d = 1.05;
			pd.di = 2.0;
			pd.o = pd.d / pd.di;
			pd.m = _par.XCV * _par.YCV * pd.di / (_par.NX * _par.NY);
			pd.p = 0.0;
			l++;
		}
		break;
	case 101: // Dispersed phase in static box
		_par.T_BOUNDARY_PERIODICITY = 2;
		_par.END_TIME = 10.0;
		_par.DT = 0.000025;
		_par.INTERVAL_TIME = 0.05;
		_par.G_Y = -1.0;
		_par.T_DISPERSED_PHASE_FLUID = 1;
		_par.N_DISPERSED_PHASE_FLUID = _par.N;

		SetParticles(_p);
		{
			std::vector<Particle> p;
			for (int i = 0; i < _par.N; i++)
			{
				if (_p[i].pos.y < 1.8)
				{
					p.push_back(_p[i]);
				}
			}
			_p = p;
			_par.N = p.size();
		}
		for (auto& p : _p)
		{
			p.mi = 0.1;
			p.nu = p.mi / p.di;
			p.m = _par.XCV * _par.YCV * p.d / (_par.NX * _par.NY);
			//p.d = p.d * pow(1.0 + fabs(_par.G_Y)*(0.8 - p.pos.y) / p.b, 1.0 / 7.0);
			p.o = 1.0;
		}

		_pDispersedPhaseFluid = std::vector<Particle>(_par.N_DISPERSED_PHASE_FLUID);
		SetParticles(_pDispersedPhaseFluid);
		for (auto& pd : _pDispersedPhaseFluid)
		{
			pd.o = 0.01;
			pd.di = 1.0;
			pd.d = pd.o * pd.di;
			pd.m = _par.XCV * _par.YCV * pd.d / (_par.NX * _par.NY);
			pd.p = 0.0;
		}
		{
			std::vector<Particle> p;
			for (int i = 0; i < _par.N_DISPERSED_PHASE_FLUID; i++)
			{
				if ( (_pDispersedPhaseFluid[i].pos.y > 0.4) && (_pDispersedPhaseFluid[i].pos.y < 0.6) )
				{
					p.push_back(_pDispersedPhaseFluid[i]);
				}
			}
			_pDispersedPhaseFluid = p;
			_par.N_DISPERSED_PHASE_FLUID = p.size();
		}
		
		for (auto& p : _p)
		{
			if ((p.pos.y > 0.4) && (p.pos.y < 0.6))
			{
				p.o = 0.99;
				p.d = p.o * p.di;
				p.m = _par.XCV * _par.YCV * p.d / (_par.NX * _par.NY);
				//p.d = p.d * pow(1.0 + fabs(_par.G_Y)*(0.8 - p.pos.y) / p.b, 1.0 / 7.0);
			}
		}

		break;
	case 102: // Dispersed phase in static box
		_par.T_BOUNDARY_PERIODICITY = 0;
		_par.END_TIME = 5.0;
		_par.DT = 0.000001;
		_par.INTERVAL_TIME = 0.01;
		_par.G_Y = -9.81;
		_par.T_DISPERSED_PHASE_FLUID = 1;
		_par.N_DISPERSED_PHASE_FLUID = _par.N;

		SetParticles(_p);
		{
			std::vector<Particle> p;
			for (int i = 0; i < _par.N; i++)
			{
				if (_p[i].pos.y < 0.5)
				{
					p.push_back(_p[i]);
				}
			}
			_p = p;
			_par.N = p.size();
		}
		for (auto& p : _p)
		{
			p.d = 1000.0;
			p.di = 1000.0;
			p.mi = 0.001;
			p.nu = p.mi / p.di;
			p.m = _par.XCV * _par.YCV * p.di / (_par.NX * _par.NY);
			p.s = 20.0;
			p.b = pow2(p.s) * p.di / p.gamma;
			p.d = p.d * pow(1.0 + 1000.0*fabs(_par.G_Y)*(0.5 - p.pos.y) / p.b, 1.0 / 7.0);
			p.o = 1.0;
		}

		_pDispersedPhaseFluid = std::vector<Particle>(_par.N_DISPERSED_PHASE_FLUID);
		SetParticles(_pDispersedPhaseFluid);
		for (auto& pd : _pDispersedPhaseFluid)
		{
			pd.o = 0.05;
			pd.di = 2500.0;
			pd.d = pd.o * pd.di;
			pd.m = _par.XCV * _par.YCV * pd.d / (_par.NX * _par.NY);
			pd.p = 0.0;
		}
		{
			std::vector<Particle> p;
			for (int i = 0; i < _par.N_DISPERSED_PHASE_FLUID; i++)
			{
				if ((_pDispersedPhaseFluid[i].pos.y > 0.475) && (_pDispersedPhaseFluid[i].pos.y < 0.5)
					&& (_pDispersedPhaseFluid[i].pos.x > 1.0) && (_pDispersedPhaseFluid[i].pos.x < 2.0) )
				{
					p.push_back(_pDispersedPhaseFluid[i]);
				}
			}
			_pDispersedPhaseFluid = p;
			_par.N_DISPERSED_PHASE_FLUID = p.size();
		}

		for (auto& p : _p)
		{
			if ((p.pos.y > 0.475) && (p.pos.y < 0.5) && (p.pos.x > 1.0) && (p.pos.x < 2.0))
			{
				p.o = 0.95;
				p.d = p.o * p.di;
				p.m = _par.XCV * _par.YCV * p.d / (_par.NX * _par.NY);
				p.d = p.d * pow(1.0 + 1000.0*fabs(_par.G_Y)*(0.5 - p.pos.y) / p.b, 1.0 / 7.0);
			}
		}

		break;
	case 103: // Dispersed phase in static box
			_par.T_BOUNDARY_PERIODICITY = 0;
			_par.END_TIME = 10.0;
			_par.DT = 0.000001;
			_par.INTERVAL_TIME = 0.01;
			_par.G_Y = -9.81;
			_par.T_DISPERSED_PHASE_FLUID = 1;
			_par.N_DISPERSED_PHASE_FLUID = _par.N;
			_par.T_SOIL = 1;
			_par.SOIL_COHESION = 0.0;
			_par.SOIL_INTERNAL_ANGLE = 0.5;
			_par.SOIL_MINIMAL_VISCOSITY = 0.01;
			_par.SOIL_MAXIMAL_VISCOSITY = 2000.0;

			SetParticles(_p);
			{
				std::vector<Particle> p;
				for (int i = 0; i < _par.N; i++)
				{
					if (_p[i].pos.y < 0.8)
					{
						p.push_back(_p[i]);
					}
				}
				_p = p;
				_par.N = p.size();
			}
			for (auto& p : _p)
			{
				if (p.pos.x < 0.3) {
					p.phaseId = 1;
					p.phaseType = 1;
					p.d = 2000.0;
					p.di = 2000.0;
				}
				else {
					p.d = 1000.0;
					p.di = 1000.0;
				}
				p.mi = 0.001;
				p.nu = p.mi / p.di;
				p.m = _par.XCV * _par.YCV * p.di / (_par.NX * _par.NY);
				p.s = 20.0;
				p.b = pow2(p.s) * p.di / p.gamma;
				p.d = p.d * pow(1.0 + p.d*fabs(_par.G_Y)*(0.8 - p.pos.y) / p.b, 1.0 / 7.0);
				p.o = 1.0;
			}

			_pDispersedPhaseFluid = std::vector<Particle>(_par.N_DISPERSED_PHASE_FLUID);
			SetParticles(_pDispersedPhaseFluid);
			for (auto& pd : _pDispersedPhaseFluid)
			{
				pd.o = 0.06;
				pd.di = 2500.0;
				pd.d = pd.o * pd.di;
				pd.m = _par.XCV * _par.YCV * pd.d / (_par.NX * _par.NY);
				pd.p = 0.0;
			}
			{
				std::vector<Particle> p;
				for (int i = 0; i < _par.N_DISPERSED_PHASE_FLUID; i++)
				{
					if ((_pDispersedPhaseFluid[i].pos.y > 0.6) && (_pDispersedPhaseFluid[i].pos.y < 0.8)
						&& (_pDispersedPhaseFluid[i].pos.x > 0.8) && (_pDispersedPhaseFluid[i].pos.x < 1.8) )
					{
						p.push_back(_pDispersedPhaseFluid[i]);
					}
				}
				_pDispersedPhaseFluid = p;
				_par.N_DISPERSED_PHASE_FLUID = p.size();
			}

			for (auto& p : _p)
			{
				if ((p.pos.y > 0.6) && (p.pos.y < 0.8) && (p.pos.x > 0.8) && (p.pos.x < 1.8))
				{
					p.o = 0.94;
					p.d = p.o * p.di;
					p.m = _par.XCV * _par.YCV * p.d / (_par.NX * _par.NY);
					p.d = p.d * pow(1.0 + 1000.0*fabs(_par.G_Y)*(0.8 - p.pos.y) / p.b, 1.0 / 7.0);
				}
			}

			break;
	case 199: // Periodic box with wavy pattern
		_par.T_BOUNDARY_PERIODICITY = 2;
		_par.DT = 0.00005;
		_par.INTERVAL_TIME = 0.01;
		_par.END_TIME = 20.0;
		_par.T_SMOOTHING_DENSITY = 0;
		_par.V_N = 1.0;
		SetParticles(_p);

		for (int i = 0; i < _par.N; i++)
		{
			_p[i].phaseId = 0;
			_p[i].phaseType = 0;
			_p[i].c = 0.0;
			_p[i].d = 1.0;
			_p[i].di = 1.0;
			_p[i].mi = 0.01;
			_p[i].gamma = 7.0;
			_p[i].nu = _p[i].mi / _p[i].d;
			_p[i].m = _par.XCV * _par.YCV * _p[i].d / (_par.NX * _par.NY);
			_p[i].s = 10.0;
			if (i == 0) _p[i].b = pow2(_p[i].s) * _p[i].di / _p[i].gamma;
			else _p[i].b = _p[0].b;

			if (_p[i].pos.y < -0.05*sin(2.0*M_PI*_p[i].pos.x/0.25) + 0.2)
			{
				_p[i].phaseType = -1;
				_p[i].phaseId = 1;
			}

		}
		break;
	case 200: // Experiment with wave maker
			_par.G_Y = -9.81;
			_par.DT = 0.000002;
			_par.INTERVAL_TIME = 0.01;
			_par.END_TIME = 20.0;
			_par.T_SMOOTHING_DENSITY = 0;
			SetParticles(_p);
			{
				std::vector<Particle> p;
				for (int i = 0; i < _par.N; i++)
				{
					if ( (_p[i].pos.x < 1.0) && (_p[i].pos.y < -0.5*_p[i].pos.x + 1.0)) p.push_back(_p[i]);
					if (_p[i].pos.x > 7.8) p.push_back(_p[i]);
					if ( (_p[i].pos.x > 1.0) && (_p[i].pos.x < 7.8) && (_p[i].pos.y < 0.5) )
					{
						p.push_back(_p[i]);
					}
				}
				_p = p;
				_par.N = p.size();
			}
			for (int i = 0; i < _par.N; i++)
			{
				_p[i].phaseId = 0;
				_p[i].phaseType = 0;
				_p[i].c = 0.0;
				_p[i].d = 1000.0;
				_p[i].di = 1000.0;
				_p[i].mi = 0.001;
				_p[i].gamma = 7.0;
				_p[i].nu = _p[i].mi / _p[i].d;
				_p[i].m = _par.XCV * _par.YCV * _p[i].d / (_par.NX * _par.NY);
				_p[i].s = 10.0;
				if (i == 0) _p[i].b = pow2(_p[i].s) * _p[i].di / _p[i].gamma;
				else _p[i].b = _p[0].b;
				if (_p[i].pos.y < 0.5)
				{
					_p[i].d = _p[i].d * pow(1.0 + 1000.0*fabs(_par.G_Y)*(0.5 - _p[i].pos.y) / _p[i].b, 1.0 / 7.0);
					_p[i].m = _p[i].m * pow(1.0 + 1000.0*fabs(_par.G_Y)*(0.5 - _p[i].pos.y) / _p[i].b, 1.0 / 7.0);
				}
				

				if (_p[i].pos.y < -0.5*_p[i].pos.x + 1.0)
				{
					_p[i].phaseType = -1;
					_p[i].phaseId = 1;
				}
				if (_p[i].pos.x > 7.8)
				{
					_p[i].phaseType = -1;
					_p[i].phaseId = 1;
				}
				if (_p[i].pos.y < 0.2)
				{
					_p[i].phaseType = -1;
					_p[i].phaseId = 1;
				}
				if ((_p[i].pos.x > 7.8) && (_p[i].pos.y > 0.2))
				{
					_p[i].phaseType = -100;
					_p[i].phaseId = 1;
				}

			}
			break;
	case 201: // Experiment with wave maker
		_par.G_Y = -9.81;
		_par.DT = 0.000005;
		_par.INTERVAL_TIME = 0.01;
		_par.END_TIME = 20.0;
		_par.T_SMOOTHING_DENSITY = 0;
		_par.T_DISPERSED_PHASE_FLUID = 1;
		_par.N_DISPERSED_PHASE_FLUID = _par.N;
		SetParticles(_p);
		{
			std::vector<Particle> p;
			for (int i = 0; i < _par.N; i++)
			{
				if ((_p[i].pos.x < 1.0) && (_p[i].pos.y < -0.5*_p[i].pos.x + 1.0)) p.push_back(_p[i]);
				if (_p[i].pos.x > 7.8) p.push_back(_p[i]);
				if ((_p[i].pos.x > 1.0) && (_p[i].pos.x < 7.8) && (_p[i].pos.y < 0.5))
				{
					p.push_back(_p[i]);
				}
			}
			_p = p;
			_par.N = p.size();
		}

		_pDispersedPhaseFluid = std::vector<Particle>(_par.N_DISPERSED_PHASE_FLUID);
		SetParticles(_pDispersedPhaseFluid);
		for (auto& pd : _pDispersedPhaseFluid)
		{
			pd.o = 0.02;
			pd.di = 2000.0;
			pd.d = pd.o * pd.di;
			pd.m = _par.XCV * _par.YCV * pd.d / (_par.NX * _par.NY);
			pd.p = 0.0;
		}
		{
			std::vector<Particle> p;
			for (int i = 0; i < _par.N_DISPERSED_PHASE_FLUID; i++)
			{
				if ((_pDispersedPhaseFluid[i].pos.x > 5.0) && (_pDispersedPhaseFluid[i].pos.x < 7.0) && (_pDispersedPhaseFluid[i].pos.y > 0.35) && (_pDispersedPhaseFluid[i].pos.y < 0.5))
				{
					p.push_back(_pDispersedPhaseFluid[i]);
				}
			}
			_pDispersedPhaseFluid = p;
			_par.N_DISPERSED_PHASE_FLUID = p.size();
		}

		for (int i = 0; i < _par.N; i++)
		{
			_p[i].phaseId = 0;
			_p[i].phaseType = 0;
			_p[i].c = 0.0;
			_p[i].di = 1000.0;
			if ((_p[i].pos.x > 5.0) && (_p[i].pos.x < 7.0) && (_p[i].pos.y > 0.35) && (_p[i].pos.y < 0.5))
			{
				_p[i].o = 0.98;
			}
			_p[i].d = _p[i].o * _p[i].di;
			_p[i].mi = 0.002;
			_p[i].gamma = 7.0;
			_p[i].nu = _p[i].mi / _p[i].d;
			_p[i].m = _par.XCV * _par.YCV * _p[i].d / (_par.NX * _par.NY);
			_p[i].s = 10.0;
			if (i == 0) _p[i].b = pow2(_p[i].s) * _p[i].di / _p[i].gamma;
			else _p[i].b = _p[0].b;
			if (_p[i].pos.y < 0.5)
			{
				_p[i].d = _p[i].d * pow(1.0 + 1000.0*fabs(_par.G_Y)*(0.5 - _p[i].pos.y) / _p[i].b, 1.0 / 7.0);
				_p[i].m = _p[i].m * pow(1.0 + 1000.0*fabs(_par.G_Y)*(0.5 - _p[i].pos.y) / _p[i].b, 1.0 / 7.0);
			}


			if (_p[i].pos.y < -0.5*_p[i].pos.x + 1.0)
			{
				_p[i].phaseType = -1;
				_p[i].phaseId = 1;
			}
			if (_p[i].pos.x > 7.8)
			{
				_p[i].phaseType = -1;
				_p[i].phaseId = 1;
			}
			if (_p[i].pos.y < 0.2)
			{
				_p[i].phaseType = -1;
				_p[i].phaseId = 1;
			}
			if ((_p[i].pos.x > 7.8) && (_p[i].pos.y > 0.2))
			{
				_p[i].phaseType = -100;
				_p[i].phaseId = 1;
			}


		}
		break;
	case 300:
		_par.T_BOUNDARY_PERIODICITY = 2;
		_par.G_Y = -1.0;
		_par.DT = 0.00005;
		_par.INTERVAL_TIME = 0.5;
		SetParticles(_p);
		{
			bool isDispersedParticleSet = false;
			for (auto& p : _p)
			{
				p.nu = 1.0 / 100.0;
				p.mi = p.nu * p.d;
				p.m = _par.XCV * _par.YCV * p.d / _par.N;
				p.s = 15.0;
				p.gamma = 7.0;
				p.b = pow2(p.s) * p.di / p.gamma;
				//p.m = p.m * pow(1.0 + p.d*(2.0 - p.pos.y) / p.b, 1.0 / 7.0);
				p.d = p.d * pow(1.0 + p.d*(2.0 - p.pos.y) / p.b, 1.0 / 7.0);
				if ((isDispersedParticleSet == false) && (p.pos.x >= 0.5) && (p.pos.y >= 1.5))
				{
					p.d = 5.0;
					p.di = 5.0;
					p.m = _par.XCV * _par.YCV * p.d / _par.N;
					p.b = _p[0].b;
					isDispersedParticleSet = true;
				}
			}
		}
		break;
	default:
		break;
	}

}

void Domain::SetParticleDefaultProperites(std::vector<Particle>& p, int i)
{
	p[i].id = i; p[i].phaseId = 0;
	p[i].pos.x = 0.0; p[i].rh_pos.x = 0.0;
	p[i].pos.y = 0.0; p[i].rh_pos.y = 0.0;
	p[i].vel.x = 0.0; p[i].rh_vel.x = 0.0;
	p[i].vel.y = 0.0; p[i].rh_vel.y = 0.0;
	p[i].vel_s.x = 0.0; p[i].vel_s.y = 0.0;
	p[i].h = _par.H; p[i].rh_h = 0.0;
	p[i].d = 1.0; p[i].di = 1.0; p[i].rh_d = 0.0; p[i].d_s = 0.0;
	p[i].m = _par.XCV * _par.YCV * p[i].d / _par.N;
	p[i].rh_m = 0.0;
	p[i].p = 0.0; p[i].ph = 0.0; p[i].phs = 0.0;
	p[i].nu = 0.01; p[i].mi = 0.01; p[i].nut = 0.0;
	p[i].str.x = 0.0; p[i].str.y = 0.0; p[i].str.z = 0.0; p[i].str.w = 0.0;
	p[i].tau.x = 0.0; p[i].tau.y = 0.0; p[i].tau.z = 0.0; p[i].tau.w = 0.0;
	p[i].phaseType = 0;
	p[i].gamma = 7.0; p[i].s = 10.0;
	p[i].b = p[i].s * p[i].s * p[i].di / p[i].gamma;	
	p[i].o = 1.0;// pow2(p[i].m / p[i].d);
	p[i].c = 0.0;
	p[i].n.x = 0.0; p[i].n.y = 0.0; p[i].n.z = 0.0;
	p[i].na = 0;
	p[i].cu = 0.0;
	p[i].st.x = 0.0; p[i].st.y = 0.0;
	p[i].cs = 0.0;
	p[i].cw = 0.0;
	p[i].ct.x = 0.0; p[i].ct.y = 0.0; p[i].ct.z = 0.0; p[i].ct.w = 0.0;

	// Surfactants
	p[i].mBulk = 0.0;
	p[i].cBulk = 0.0;
	p[i].dBulk = 0.0;
	p[i].mSurf = 0.0;
	p[i].cSurf = 0.0;
	p[i].dSurf = 0.0;
	p[i].cSurfGrad.x = 0.0;
	p[i].cSurfGrad.y = 0.0;
	p[i].a = 0.0;
}

void Domain::SetParticles(std::vector<Particle>& p)
{
	double dx = _par.XCV / _par.NX;
	double dy = _par.YCV / _par.NY;
	double x0 = 0.5 * dx;
	double y0 = 0.5 * dy;
	int l = 0;
	for (int i = 0; i < _par.NX; i++) {
		for (int j = 0; j < _par.NY; j++) {
			SetParticleDefaultProperites(p, l);
			p[l].pos.x = x0 + dx * static_cast<double>(i);
			p[l].pos.y = y0 + dx * static_cast<double>(j);
			p[l].rh_pos.x = 0.0;
			p[l].rh_pos.y = 0.0;
			l++;
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

std::vector<Particle>* Domain::GetParticlesDispersedPhaseFluid()
{
	return &_pDispersedPhaseFluid;
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
	if (!IsConsistentWithGeometry()) {
		std::cerr << "Domain is geometrically not consistent." << std::endl;
		std::cerr << "XCV=" << _par.XCV << " YCV=" << _par.YCV << std::endl;
		std::cerr << "NXC=" << _par.NXC << " NYC=" << _par.NYC << std::endl;
		
		exit(EXIT_FAILURE);
	}
}

bool Domain::IsConsistentWithGeometry()
{
	if ( (_par.NXC/_par.NYC == static_cast<int>( (_par.XCV + 0.25*_par.YCV) / _par.YCV) )
	  && (_par.NYC / _par.NXC == static_cast<int>((_par.YCV + 0.25*_par.XCV) / _par.XCV) ) )
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
	write_to_file(filename, _p, _pDispersedPhase, _pDispersedPhaseFluid, &_par, fileFormat);
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

double Domain::GetAndWriteKinetic(const char *filename)
{
	double kinetic = 0.0;
	for (int i = 0; i < _par.N; i++) {
		kinetic += 0.5 * _p[i].m * ( pow2(_p[i].vel.x) + pow2(_p[i].vel.y) );
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

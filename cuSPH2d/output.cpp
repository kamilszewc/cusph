/*
*  output.cpp
*
*  Author: Kamil Szewc (kamil.szewc@gmail.com)
*  Modified on: 26-09-2014
*
*/

#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <ctime>
#include "output.h"
#include "sph.h"
#include "hlp.h"
#include "license.h"

template<class T> std::string parse_parameter(std::string name, std::string type, T value, FileFormat fileFormat)
{
	std::string str = "";
	std::ostringstream stream;

	switch (fileFormat)
	{
	case XML:
		str += "      <parameter name=\"" + name + "\" type=\"" + type + "\">";
		stream << value << "</parameter>";
		break;
	case SPH:
		str += name + " " + type + " ";
		stream << value;
		break;
	default:
		break;
	}

	return str + stream.str();
}


std::string parse_field(std::string name, std::string type)
{
	return "    <field name=\"" + name + "\" type=\"" + type + "\">";
}


std::string parse_field_start(std::string name, std::string type, FileFormat fileFormat)
{
	std::ostringstream stream;
	switch (fileFormat)
	{
	case XML:
		stream << "    <field name=\"" << name << "\" type=\"" << type << "\">";
		break;
	case SPH:
		stream << "*" << name << " " << type << " ";
		break;
	default:
		break;
	}
	return stream.str();
}

std::string parse_field_end(FileFormat fileFormat)
{
	std::ostringstream stream;
	switch (fileFormat)
	{
	case XML:
		stream << "</field>" << std::endl;
		break;
	case SPH:
		stream << std::endl;
		break;
	default:
		break;
	}
	return stream.str();
}

void write_to_file(const char* filename, std::vector<Particle> p, std::vector<ParticleDispersedPhase> pDispersedPhase, std::vector<Particle> pDispersedPhaseFluid, Parameters *par, FileFormat fileFormat)
{
	std::ofstream file;
	file.open(filename);

	if (file.is_open())
	{
		switch (fileFormat)
		{
		case XML:
			file << "<?xml version=\"1.0\" encoding=\"utf-8\" ?>" << std::endl;
			file << "<sph>" << std::endl;
			file << "  <domain>" << std::endl;
			file << "     <parameters>" << std::endl;
			break;
		case SPH:
			file << "# " << License::GetShortInfo() << std::endl;
			file << "# Version: " << License::Version << std::endl;
			file << "# Compilation date: " << __DATE__ << std::endl;
			{
				time_t t = time(0);
				file << "# Output date: " << asctime(localtime(&t));
			}
			file << "@parameters" << std::endl;
			break;
		default:
			break;
		}
		file << parse_parameter<double>("HDR", "double", par->HDR, fileFormat) << std::endl;
		file << parse_parameter<int>("NXC", "int", par->NXC, fileFormat) << std::endl;
		file << parse_parameter<int>("NYC", "int", par->NYC, fileFormat) << std::endl;
		file << parse_parameter<double>("XCV", "double", par->XCV, fileFormat) << std::endl;
		file << parse_parameter<double>("YCV", "double", par->YCV, fileFormat) << std::endl;
		file << parse_parameter<double>("DT", "double", par->DT, fileFormat) << std::endl;
		file << parse_parameter<double>("INTERVAL_TIME", "double", par->INTERVAL_TIME, fileFormat) << std::endl;
		file << parse_parameter<int>("N", "int", par->N, fileFormat) << std::endl;
		file << parse_parameter<int>("T_BOUNDARY_PERIODICITY", "int", par->T_BOUNDARY_PERIODICITY, fileFormat) << std::endl;
		file << parse_parameter<int>("T_MODEL", "int", par->T_MODEL, fileFormat) << std::endl;
		file << parse_parameter<int>("T_TIME_STEP", "int", par->T_TIME_STEP, fileFormat) << std::endl;
		file << parse_parameter<double>("END_TIME", "double", par->END_TIME, fileFormat) << std::endl;
		file << parse_parameter<double>("G_X", "double", par->G_X, fileFormat) << std::endl;
		file << parse_parameter<double>("G_Y", "double", par->G_Y, fileFormat) << std::endl;
		file << parse_parameter<double>("V_N", "double", par->V_N, fileFormat) << std::endl;
		file << parse_parameter<double>("V_E", "double", par->V_E, fileFormat) << std::endl;
		file << parse_parameter<double>("V_S", "double", par->V_S, fileFormat) << std::endl;
		file << parse_parameter<double>("V_W", "double", par->V_W, fileFormat) << std::endl;
		file << parse_parameter<int>("T_INTERFACE_CORRECTION", "int", par->T_INTERFACE_CORRECTION, fileFormat) << std::endl;
		file << parse_parameter<double>("INTERFACE_CORRECTION", "double", par->INTERFACE_CORRECTION, fileFormat) << std::endl;
		file << parse_parameter<int>("T_SURFACE_TENSION", "int", par->T_SURFACE_TENSION, fileFormat) << std::endl;
		file << parse_parameter<double>("SURFACE_TENSION", "double", par->SURFACE_TENSION, fileFormat) << std::endl;
		file << parse_parameter<int>("T_NORMAL_VECTOR", "int", par->T_NORMAL_VECTOR, fileFormat) << std::endl;
		file << parse_parameter<int>("T_NORMAL_VECTOR_TRESHOLD", "int", par->T_NORMAL_VECTOR_TRESHOLD, fileFormat) << std::endl;
		file << parse_parameter<int>("T_XSPH", "int", par->T_XSPH, fileFormat) << std::endl;
		file << parse_parameter<double>("XSPH", "double", par->XSPH, fileFormat) << std::endl;
		file << parse_parameter<int>("T_TURBULENCE", "int", par->T_TURBULENCE, fileFormat) << std::endl;
		file << parse_parameter<int>("T_SOIL", "int", par->T_SOIL, fileFormat) << std::endl;
		file << parse_parameter<real>("SOIL_COHESION", "real", par->SOIL_COHESION, fileFormat) << std::endl;
		file << parse_parameter<real>("SOIL_INTERNAL_ANGLE", "real", par->SOIL_INTERNAL_ANGLE, fileFormat) << std::endl;
		file << parse_parameter<real>("SOIL_MINIMAL_VISCOSITY", "real", par->SOIL_MINIMAL_VISCOSITY, fileFormat) << std::endl;
		file << parse_parameter<real>("SOIL_MAXIMAL_VISCOSITY", "real", par->SOIL_MAXIMAL_VISCOSITY, fileFormat) << std::endl;
		file << parse_parameter<int>("T_HYDROSTATIC_PRESSURE", "int", par->T_HYDROSTATIC_PRESSURE, fileFormat) << std::endl;
		file << parse_parameter<int>("T_SMOOTHING_DENSITY", "int", par->T_SMOOTHING_DENSITY, fileFormat) << std::endl;
		file << parse_parameter<int>("T_STRAIN_TENSOR", "int", par->T_STRAIN_TENSOR, fileFormat) << std::endl;
		file << parse_parameter<int>("T_SURFACTANTS", "int", par->T_SURFACTANTS, fileFormat) << std::endl;
		file << parse_parameter<int>("T_VARIABLE_H", "int", par->T_VARIABLE_H, fileFormat) << std::endl;
		file << parse_parameter<int>("T_DISPERSED_PHASE_FLUID", "int", par->T_DISPERSED_PHASE_FLUID, fileFormat) << std::endl;
		file << parse_parameter<int>("N_DISPERSED_PHASE_FLUID", "int", par->N_DISPERSED_PHASE_FLUID, fileFormat) << std::endl;

		switch (fileFormat)
		{
		case XML:
			file << "    </parameters>" << std::endl;

			file << "    <interphase-parameters>" << std::endl;
			file << "    </interphase-parameters>" << std::endl;

			file << "  </domain>" << std::endl;

			file << "  <sph-particles>" << std::endl;
			break;
		case SPH:
			file << "@sph-particles" << std::endl;
			break;
		default:
			break;
		}

		file << parse_field_start("id", "int", fileFormat);
		for (int i = 0; i < par->N; i++) { file << p[i].id << " "; }
		file << parse_field_end(fileFormat);

		file << parse_field_start("phase-id", "int", fileFormat);
		for (int i = 0; i < par->N; i++) { file << p[i].phaseId << " "; }
		file << parse_field_end(fileFormat);

		file << parse_field_start("phase-type", "int", fileFormat);
		for (int i = 0; i < par->N; i++) { file << p[i].phaseType << " "; }
		file << parse_field_end(fileFormat);

		file << parse_field_start("x-position", "double", fileFormat);
		for (int i = 0; i < par->N; i++) { file << p[i].pos.x << " "; }
		file << parse_field_end(fileFormat);

		file << parse_field_start("y-position", "double", fileFormat);
		for (int i = 0; i < par->N; i++) { file << p[i].pos.y << " "; }
		file << parse_field_end(fileFormat);

		file << parse_field_start("x-velocity", "double", fileFormat);
		for (int i = 0; i < par->N; i++) { file << p[i].vel.x << " "; }
		file << parse_field_end(fileFormat);

		file << parse_field_start("y-velocity", "double", fileFormat);
		for (int i = 0; i < par->N; i++) { file << p[i].vel.y << " "; }
		file << parse_field_end(fileFormat);

		file << parse_field_start("mass", "double", fileFormat);
		for (int i = 0; i < par->N; i++) { file << p[i].m << " "; }
		file << parse_field_end(fileFormat);

		file << parse_field_start("pressure", "double", fileFormat);
		for (int i = 0; i < par->N; i++) { file << p[i].p << " "; }
		file << parse_field_end(fileFormat);

		file << parse_field_start("density", "double", fileFormat);
		for (int i = 0; i < par->N; i++) { file << p[i].d << " "; }
		file << parse_field_end(fileFormat);

		file << parse_field_start("initial-density", "double", fileFormat);
		for (int i = 0; i < par->N; i++) { file << p[i].di << " "; }
		file << parse_field_end(fileFormat);

		file << parse_field_start("volume", "double", fileFormat);
		for (int i = 0; i < par->N; i++) { file << p[i].o << " "; }
		file << parse_field_end(fileFormat);

		file << parse_field_start("dynamic-viscosity", "double", fileFormat);
		for (int i = 0; i < par->N; i++) { file << p[i].mi << " "; }
		file << parse_field_end(fileFormat);

		file << parse_field_start("gamma", "double", fileFormat);
		for (int i = 0; i < par->N; i++) { file << p[i].gamma << " "; }
		file << parse_field_end(fileFormat);

		file << parse_field_start("sound-speed", "double", fileFormat);
		for (int i = 0; i < par->N; i++) { file << p[i].s << " "; }
		file << parse_field_end(fileFormat);

		file << parse_field_start("equation-of-state-coefficient", "double", fileFormat);
		for (int i = 0; i < par->N; i++) { file << p[i].b << " "; }
		file << parse_field_end(fileFormat);

		file << parse_field_start("color-function", "double", fileFormat);
		for (int i = 0; i < par->N; i++) { file << p[i].c << " "; }
		file << parse_field_end(fileFormat);

		file << parse_field_start("smoothing-length", "double", fileFormat);
		for (int i = 0; i < par->N; i++) { file << p[i].h << " "; }
		file << parse_field_end(fileFormat);

		if (par->T_HYDROSTATIC_PRESSURE != 0)
		{
			file << parse_field_start("hydrostatic-pressure", "double", fileFormat);
			for (int i = 0; i < par->N; i++) { file << p[i].ph << " "; }
			file << parse_field_end(fileFormat);
		}
		if ((par->T_STRAIN_TENSOR != 0) || (par->T_TURBULENCE != 0) || (par->T_SOIL != 0))
		{
			file << parse_field_start("strain-rate", "double", fileFormat);
			for (int i = 0; i < par->N; i++) { file << sqrt(2.0) * sqrt(pow2(p[i].str.x) + pow2(p[i].str.y) + pow2(p[i].str.z) + pow2(p[i].str.w)) << " "; }
			file << parse_field_end(fileFormat);
		}
		if ((par->T_TURBULENCE != 0) || (par->T_SOIL != 0))
		{
			file << parse_field_start("turbulent-viscosity/soil-viscosity", "double", fileFormat);
			for (int i = 0; i < par->N; i++) { file << p[i].nut << " "; }
			file << parse_field_end(fileFormat);
		}
		if (par->T_SOIL == 2)
		{
			file << parse_field_start("smoothed-color-function", "double", fileFormat);
			for (int i = 0; i < par->N; i++) { file << p[i].cs << " "; }
			file << parse_field_end(fileFormat);
		}
		if (par->T_SURFACE_TENSION != 0)
		{
			file << parse_field_start("x-normal-vector", "double", fileFormat);
			for (int i = 0; i < par->N; i++) { file << p[i].n.x << " "; }
			file << parse_field_end(fileFormat);
			 
			file << parse_field_start("y-normal-vector", "double", fileFormat);
			for (int i = 0; i < par->N; i++) { file << p[i].n.y << " "; }
			file << parse_field_end(fileFormat);

			file << parse_field_start("normal-vector-norm", "double", fileFormat);
			for (int i = 0; i < par->N; i++) { file << p[i].n.z << " "; }
			file << parse_field_end(fileFormat);

			file << parse_field_start("curvature-influence-indicator", "int", fileFormat);
			for (int i = 0; i < par->N; i++) { file << p[i].na << " "; }
			file << parse_field_end(fileFormat);

			file << parse_field_start("curvature", "double", fileFormat);
			for (int i = 0; i < par->N; i++) { file << p[i].cu << " "; }
			file << parse_field_end(fileFormat);

			file << parse_field_start("x-surface-tension", "double", fileFormat);
			for (int i = 0; i < par->N; i++) { file << p[i].st.x << " "; }
			file << parse_field_end(fileFormat);

			file << parse_field_start("y-surface-tension", "double", fileFormat);
			for (int i = 0; i < par->N; i++) { file << p[i].st.y << " "; }
			file << parse_field_end(fileFormat);

		}
		if (par->T_SURFACE_TENSION == 2)
		{
			file << parse_field_start("xx-capillary-tensor", "double", fileFormat);
			for (int i = 0; i < par->N; i++) { file << p[i].ct.x << " "; }
			file << parse_field_end(fileFormat);

			file << parse_field_start("xy-capillary-tensor", "double", fileFormat);
			for (int i = 0; i < par->N; i++) { file << p[i].ct.y << " "; }
			file << parse_field_end(fileFormat);

			file << parse_field_start("yx-capillary-tensor", "double", fileFormat);
			for (int i = 0; i < par->N; i++) { file << p[i].ct.z << " "; }
			file << parse_field_end(fileFormat);

			file << parse_field_start("yy-capillary-tensor", "double", fileFormat);
			for (int i = 0; i < par->N; i++) { file << p[i].ct.w << " "; }
			file << parse_field_end(fileFormat);
		}

		if (par->T_SURFACTANTS != 0)
		{
			file << parse_field_start("bulk-surfactant-mass", "double", fileFormat);
			for (int i = 0; i < par->N; i++) { file << p[i].mBulk << " "; }
			file << parse_field_end(fileFormat);

			file << parse_field_start("bulk-surfactant-concentration", "double", fileFormat);
			for (int i = 0; i < par->N; i++) { file << p[i].cBulk<< " "; }
			file << parse_field_end(fileFormat);
			
			file << parse_field_start("bulk-surfactant-diffusion-coefficient", "double", fileFormat);
			for (int i = 0; i < par->N; i++) { file << p[i].dBulk << " "; }
			file << parse_field_end(fileFormat);

			file << parse_field_start("surfactant-mass", "double", fileFormat);
			for (int i = 0; i < par->N; i++) { file << p[i].mSurf << " "; }
			file << parse_field_end(fileFormat);
			
			file << parse_field_start("surfactant-concentration", "double", fileFormat);
			for (int i = 0; i < par->N; i++) { file << p[i].cSurf << " "; }
			file << parse_field_end(fileFormat);
			
			file << parse_field_start("surfactant-diffusion-coefficient", "double", fileFormat);
			for (int i = 0; i < par->N; i++) { file << p[i].dSurf << " "; }
			file << parse_field_end(fileFormat);
			
			file << parse_field_start("x-surfactant-concentration-gradient", "double", fileFormat);
			for (int i = 0; i < par->N; i++) { file << p[i].cSurfGrad.x << " "; }
			file << parse_field_end(fileFormat);

			file << parse_field_start("y-surfactant-concentration-gradient", "double", fileFormat);
			for (int i = 0; i < par->N; i++) { file << p[i].cSurfGrad.y << " "; }
			file << parse_field_end(fileFormat);
			
			file << parse_field_start("interface-area", "double", fileFormat);
			for (int i = 0; i < par->N; i++) { file << p[i].a << " "; }
			file << parse_field_end(fileFormat);
		}

		switch (fileFormat)
		{
		case XML:
			file << "  </sph-particles>" << std::endl;
			break;
		case SPH:
			break;
		default:
			break;
		}

		if (par->T_DISPERSED_PHASE != 0)
		{
			switch (fileFormat)
			{
			case XML:
				file << "  <dispersed-phase>" << std::endl;
				break;
			case SPH:
				file << "@dispersed-phase-particles" << std::endl;
				break;
			default:
				break;
			}

			file << parse_field_start("x-position-dispersed-phase", "double", fileFormat);
			for (int i = 0; i < par->N_DISPERSED_PHASE; i++) { file << pDispersedPhase[i].pos.x << " "; }
			file << parse_field_end(fileFormat);

			file << parse_field_start("y-position-dispersed-phase", "double", fileFormat);
			for (int i = 0; i < par->N_DISPERSED_PHASE; i++) { file << pDispersedPhase[i].pos.y << " "; }
			file << parse_field_end(fileFormat);

			file << parse_field_start("x-velocity-dispersed-phase", "double", fileFormat);
			for (int i = 0; i < par->N_DISPERSED_PHASE; i++) { file << pDispersedPhase[i].vel.x << " "; }
			file << parse_field_end(fileFormat);

			file << parse_field_start("y-velocity-dispersed-phase", "double", fileFormat);
			for (int i = 0; i < par->N_DISPERSED_PHASE; i++) { file << pDispersedPhase[i].vel.y << " "; }
			file << parse_field_end(fileFormat);

			file << parse_field_start("density-dispersed-phase", "double", fileFormat);
			for (int i = 0; i < par->N_DISPERSED_PHASE; i++) { file << pDispersedPhase[i].d << " "; }
			file << parse_field_end(fileFormat);

			file << parse_field_start("diameter-dispersed-phase", "double", fileFormat);
			for (int i = 0; i < par->N_DISPERSED_PHASE; i++) { file << pDispersedPhase[i].dia << " "; }
			file << parse_field_end(fileFormat);

			file << parse_field_start("fluid-density-dispersed-phase", "double", fileFormat);
			for (int i = 0; i < par->N_DISPERSED_PHASE; i++) { file << pDispersedPhase[i].dFl << " "; }
			file << parse_field_end(fileFormat);

			switch (fileFormat)
			{
			case XML:
				file << "  </dispersed-phase>" << std::endl;
				break;
			default:
				break;
			}
		}

		if (par->T_DISPERSED_PHASE_FLUID != 0)
		{
			switch (fileFormat)
			{
			case XML:
				file << "  <dispersed-phase-fluid>" << std::endl;
				break;
			case SPH:
				file << "@dispersed-phase-fluid-particles" << std::endl;
				break;
			default:
				break;
			}

			file << parse_field_start("x-position-dispersed-phase-fluid", "double", fileFormat);
			for (int i = 0; i < par->N_DISPERSED_PHASE_FLUID; i++) { file << pDispersedPhaseFluid[i].pos.x << " "; }
			file << parse_field_end(fileFormat);

			file << parse_field_start("y-position-dispersed-phase-fluid", "double", fileFormat);
			for (int i = 0; i < par->N_DISPERSED_PHASE_FLUID; i++) { file << pDispersedPhaseFluid[i].pos.y << " "; }
			file << parse_field_end(fileFormat);

			file << parse_field_start("x-velocity-dispersed-phase-fluid", "double", fileFormat);
			for (int i = 0; i < par->N_DISPERSED_PHASE_FLUID; i++) { file << pDispersedPhaseFluid[i].vel.x << " "; }
			file << parse_field_end(fileFormat);

			file << parse_field_start("y-velocity-dispersed-phase-fluid", "double", fileFormat);
			for (int i = 0; i < par->N_DISPERSED_PHASE_FLUID; i++) { file << pDispersedPhaseFluid[i].vel.y << " "; }
			file << parse_field_end(fileFormat);

			file << parse_field_start("density-dispersed-phase-fluid", "double", fileFormat);
			for (int i = 0; i < par->N_DISPERSED_PHASE_FLUID; i++) { file << pDispersedPhaseFluid[i].d << " "; }
			file << parse_field_end(fileFormat);

			file << parse_field_start("initial-density-dispersed-phase-fluid", "double", fileFormat);
			for (int i = 0; i < par->N_DISPERSED_PHASE_FLUID; i++) { file << pDispersedPhaseFluid[i].di << " "; }
			file << parse_field_end(fileFormat);

			file << parse_field_start("volume-fraction-dispersed-phase-fluid", "double", fileFormat);
			for (int i = 0; i < par->N_DISPERSED_PHASE_FLUID; i++) { file << pDispersedPhaseFluid[i].o << " "; }
			file << parse_field_end(fileFormat);

			file << parse_field_start("mass-dispersed-phase-fluid", "double", fileFormat);
			for (int i = 0; i < par->N_DISPERSED_PHASE_FLUID; i++) { file << pDispersedPhaseFluid[i].m << " "; }
			file << parse_field_end(fileFormat);

			file << parse_field_start("pressure-dispersed-phase-fluid", "double", fileFormat);
			for (int i = 0; i < par->N_DISPERSED_PHASE_FLUID; i++) { file << pDispersedPhaseFluid[i].p << " "; }
			file << parse_field_end(fileFormat);

			switch (fileFormat)
			{
			case XML:
				file << "  </dispersed-phase-fluid>" << std::endl;
				break;
			default:
				break;
			}
		}


		switch (fileFormat)
		{
		case XML:
			file << "</sph>" << std::endl;
			break;
		case SPH:
			file << std::endl;
			break;
		default:
			break;
		}
		

		file.close();
	}
}

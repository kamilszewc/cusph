/*
*  output.cpp
*
*  Author: Kamil Szewc (kamil.szewc@gmail.com)
*  Modified on: 27-09-2014
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

void write_to_file(const char* filename, std::vector<Particle> p, std::vector<ParticleDispersedPhase> pDispersedPhase, Parameters *par, FileFormat fileFormat)
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
				file << "# Output time: " << asctime(localtime(&t));
			}
			
			file << "@parameters" << std::endl;
			break;
		default:
			break;
		}
		file << parse_parameter<double>("HDR", "double", par->HDR, fileFormat) << std::endl;
		file << parse_parameter<int>("NXC", "int", par->NXC, fileFormat) << std::endl;
		file << parse_parameter<int>("NYC", "int", par->NYC, fileFormat) << std::endl;
		file << parse_parameter<int>("NZC", "int", par->NZC, fileFormat) << std::endl;
		file << parse_parameter<double>("XCV", "double", par->XCV, fileFormat) << std::endl;
		file << parse_parameter<double>("YCV", "double", par->YCV, fileFormat) << std::endl;
		file << parse_parameter<double>("ZCV", "double", par->ZCV, fileFormat) << std::endl;
		file << parse_parameter<double>("DT", "double", par->DT, fileFormat) << std::endl;
		file << parse_parameter<double>("INTERVAL_TIME", "double", par->INTERVAL_TIME, fileFormat) << std::endl;
		file << parse_parameter<int>("N", "int", par->N, fileFormat) << std::endl;
		file << parse_parameter<int>("T_BOUNDARY_PERIODICITY", "int", par->T_BOUNDARY_PERIODICITY, fileFormat) << std::endl;
		file << parse_parameter<int>("T_MODEL", "int", par->T_MODEL, fileFormat) << std::endl;
		file << parse_parameter<int>("T_TIME_STEP", "int", par->T_TIME_STEP, fileFormat) << std::endl;
		file << parse_parameter<double>("END_TIME", "double", par->END_TIME, fileFormat) << std::endl;
		file << parse_parameter<double>("G_X", "double", par->G_X, fileFormat) << std::endl;
		file << parse_parameter<double>("G_Y", "double", par->G_Y, fileFormat) << std::endl;
		file << parse_parameter<double>("G_Z", "double", par->G_Z, fileFormat) << std::endl;
		file << parse_parameter<double>("V_N", "double", par->V_N, fileFormat) << std::endl;
		file << parse_parameter<double>("V_E", "double", par->V_E, fileFormat) << std::endl;
		file << parse_parameter<double>("V_S", "double", par->V_S, fileFormat) << std::endl;
		file << parse_parameter<double>("V_W", "double", par->V_W, fileFormat) << std::endl;
		file << parse_parameter<double>("V_B", "double", par->V_B, fileFormat) << std::endl;
		file << parse_parameter<double>("V_T", "double", par->V_T, fileFormat) << std::endl;
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
		file << parse_parameter<int>("T_STRAIN_RATE", "int", par->T_STRAIN_RATE, fileFormat) << std::endl;
		file << parse_parameter<int>("T_HYDROSTATIC_PRESSURE", "int", par->T_HYDROSTATIC_PRESSURE, fileFormat) << std::endl;
		file << parse_parameter<int>("T_SOLID_PARTICLE", "int", par->T_SOLID_PARTICLE, fileFormat) << std::endl;
		file << parse_parameter<int>("T_SMOOTHING_DENSITY", "int", par->T_SMOOTHING_DENSITY, fileFormat) << std::endl;

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

		file << parse_field_start("z-position", "double", fileFormat);
		for (int i = 0; i < par->N; i++) { file << p[i].pos.z << " "; }
		file << parse_field_end(fileFormat);

		file << parse_field_start("x-velocity", "double", fileFormat);
		for (int i = 0; i < par->N; i++) { file << p[i].vel.x << " "; }
		file << parse_field_end(fileFormat);

		file << parse_field_start("y-velocity", "double", fileFormat);
		for (int i = 0; i < par->N; i++) { file << p[i].vel.y << " "; }
		file << parse_field_end(fileFormat);

		file << parse_field_start("z-velocity", "double", fileFormat);
		for (int i = 0; i < par->N; i++) { file << p[i].vel.z << " "; }
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

		if (par->T_HYDROSTATIC_PRESSURE != 0)
		{
			file << parse_field_start("hydrostatic-pressure", "double", fileFormat);
			for (int i = 0; i < par->N; i++) { file << p[i].ph << " "; }
			file << parse_field_end(fileFormat);
		}

		if ((par->T_TURBULENCE != 0) || (par->T_SOIL != 0))
		{
			file << parse_field_start("strain-rate", "double", fileFormat);
			for (int i = 0; i < par->N; i++) { file << p[i].str << " "; }
			file << parse_field_end(fileFormat);

			file << parse_field_start("turbulent-viscosity/soil-viscosity", "double", fileFormat);
			for (int i = 0; i < par->N; i++) { file << p[i].nut << " "; }
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

			file << parse_field_start("z-normal-vector", "double", fileFormat);
			for (int i = 0; i < par->N; i++) { file << p[i].n.z << " "; }
			file << parse_field_end(fileFormat);

			file << parse_field_start("normal-vector-norm", "double", fileFormat);
			for (int i = 0; i < par->N; i++) { file << p[i].n.w << " "; }
			file << parse_field_end(fileFormat);

			file << parse_field_start("curvature-influence-indicator", "int", fileFormat);
			for (int i = 0; i < par->N; i++) { file << p[i].na << " "; }
			file << parse_field_end(fileFormat);

			file << parse_field_start("curvature", "double", fileFormat);
			for (int i = 0; i < par->N; i++) { file << p[i].cu << " "; }
			file << parse_field_end(fileFormat);

			file << parse_field_start("x-surface tension", "double", fileFormat);
			for (int i = 0; i < par->N; i++) { file << p[i].st.x << " "; }
			file << parse_field_end(fileFormat);

			file << parse_field_start("y-surface tension", "double", fileFormat);
			for (int i = 0; i < par->N; i++) { file << p[i].st.y << " "; }
			file << parse_field_end(fileFormat);

			file << parse_field_start("z-surface tension", "double", fileFormat);
			for (int i = 0; i < par->N; i++) { file << p[i].st.z << " "; }
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
		}

		switch (fileFormat)
		{
		case XML:
			file << "  </dispersed-phase>" << std::endl;
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


void write_raw_phase_data(const char* filename, int phaseId, std::vector<Particle> p, Parameters *par)
{
	std::ofstream file;
	file.open(filename);
	if (file.is_open())
	{
		for (int i = 0; i < par->N; i++)
		{
			if (p[i].phaseId == phaseId)
			{
				file << p[i].pos.x << " " << p[i].pos.y << " " << p[i].pos.z << " ";
				file << p[i].vel.x << " " << p[i].vel.y << " " << p[i].vel.z << " ";
				file << p[i].m << " " << p[i].p << " " << p[i].d;
				if ((par->T_TURBULENCE != 0) || (par->T_SOIL != 0))
				{
					file << " " << p[i].str << " " << p[i].nut;
				}
				if ((par->T_HYDROSTATIC_PRESSURE != 0))
				{
					file << " " << p[i].ph;
				}
				file << std::endl;
			}
		}

		file.close();
	}
}

void write_raw_phases_data(const char* filename, std::vector<Particle> p, Parameters* par)
{
	std::ofstream file;
	file.open(filename);
	if (file.is_open())
	{
		for (int i = 0; i < par->N; i++)
		{
			if (p[i].phaseId != 0)
			{
				file << p[i].pos.x << " " << p[i].pos.y << " " << p[i].pos.z << " ";
				file << p[i].vel.x << " " << p[i].vel.y << " " << p[i].vel.z << " ";
				file << p[i].m << " " << p[i].p << " " << p[i].d;
				if ((par->T_TURBULENCE != 0) || (par->T_SOIL != 0))
				{
					file << " " << p[i].str << " " << p[i].nut;
				}
				if ((par->T_HYDROSTATIC_PRESSURE != 0))
				{
					file << " " << p[i].ph;
				}
				file << std::endl;
			}
		}
	}
}

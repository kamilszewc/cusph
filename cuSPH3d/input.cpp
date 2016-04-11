/*
*  input.cpp
*
*  Author: Kamil Szewc (kamil.szewc@gmail.com)
*  Modified on: 27-09-2014
*
*/

#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include "sph.h"
#include "hlp.h"
#include "tinyxml2/tinyxml2.h"


void read_parameters_from_xml(const char *filename, Parameters *par)
{
	using namespace tinyxml2;

	XMLDocument xmlDocument;
	xmlDocument.LoadFile(filename);

	XMLElement* xmlElementRoot = xmlDocument.FirstChildElement("sph");

	XMLElement* xmlElementParameters = xmlElementRoot->FirstChildElement("domain")->FirstChildElement("parameters");

	for (XMLElement* xmlElement = xmlElementParameters->FirstChildElement("parameter"); xmlElement != NULL; xmlElement = xmlElement->NextSiblingElement("parameter"))
	{
		const char* name = xmlElement->Attribute("name");
		if (!strcmp(name, "HDR"))
		{
			par->HDR = atof(xmlElement->GetText());
		}
		if (!strcmp(name, "N"))
		{
			par->N = atoi(xmlElement->GetText());
		}
		if (!strcmp(name, "NXC"))
		{
			par->NXC = atoi(xmlElement->GetText());
		}
		else if (!strcmp(name, "NYC"))
		{
			par->NYC = atoi(xmlElement->GetText());
		}
		else if (!strcmp(name, "NZC"))
		{
			par->NZC = atoi(xmlElement->GetText());
		}
		else if (!strcmp(name, "XCV"))
		{
			par->XCV = atof(xmlElement->GetText());
		}
		else if (!strcmp(name, "YCV"))
		{
			par->YCV = atof(xmlElement->GetText());
		}
		else if (!strcmp(name, "ZCV"))
		{
			par->ZCV = atof(xmlElement->GetText());
		}
		else if (!strcmp(name, "DT"))
		{
			par->DT = atof(xmlElement->GetText());
		}
		else if (!strcmp(name, "END_TIME"))
		{
			par->END_TIME = atof(xmlElement->GetText());
		}
		else if (!strcmp(name, "G_X"))
		{
			par->G_X = atof(xmlElement->GetText());
		}
		else if (!strcmp(name, "G_Y"))
		{
			par->G_Y = atof(xmlElement->GetText());
		}
		else if (!strcmp(name, "G_Z"))
		{
			par->G_Z = atof(xmlElement->GetText());
		}
		else if (!strcmp(name, "INTERVAL_TIME"))
		{
			par->INTERVAL_TIME = atof(xmlElement->GetText());
		}
		else if (!strcmp(name, "T_BOUNDARY_PERIODICITY"))
		{
			par->T_BOUNDARY_PERIODICITY = atoi(xmlElement->GetText());
		}
		else if (!strcmp(name, "T_INTERFACE_CORRECTION"))
		{
			par->T_INTERFACE_CORRECTION = atoi(xmlElement->GetText());
		}
		else if (!strcmp(name, "INTERFACE_CORRECTION"))
		{
			par->INTERFACE_CORRECTION = atof(xmlElement->GetText());
		}
		else if (!strcmp(name, "T_MODEL"))
		{
			par->T_MODEL = atoi(xmlElement->GetText());
		}
		else if (!strcmp(name, "T_SURFACE_TENSION"))
		{
			par->T_SURFACE_TENSION = atoi(xmlElement->GetText());
		}
		else if (!strcmp(name, "SURFACE_TENSION"))
		{
			par->SURFACE_TENSION = atof(xmlElement->GetText());
		}
		else if (!strcmp(name, "T_NORMAL_VECTOR"))
		{
			par->T_NORMAL_VECTOR = atoi(xmlElement->GetText());
		}
		else if (!strcmp(name, "T_NORMAL_VECTOR_TRESHOLD"))
		{
			par->T_NORMAL_VECTOR_TRESHOLD = atoi(xmlElement->GetText());
		}
		else if (!strcmp(name, "T_TURBULENCE"))
		{
			par->T_TURBULENCE = atoi(xmlElement->GetText());
		}
		else if (!strcmp(name, "T_SOIL"))
		{
			par->T_SOIL = atoi(xmlElement->GetText());
		}
		else if (!strcmp(name, "SOIL_COHESION"))
		{
			par->SOIL_COHESION = atof(xmlElement->GetText());
		}
		else if (!strcmp(name, "SOIL_INTERNAL_ANGLE"))
		{
			par->SOIL_INTERNAL_ANGLE = atof(xmlElement->GetText());
		}
		else if (!strcmp(name, "SOIL_MINIMAL_VISCOSITY"))
		{
			par->SOIL_MINIMAL_VISCOSITY = atof(xmlElement->GetText());
		}
		else if (!strcmp(name, "SOIL_MAXIMAL_VISCOSITY"))
		{
			par->SOIL_MAXIMAL_VISCOSITY = atof(xmlElement->GetText());
		}
		else if (!strcmp(name, "T_STRAIN_RATE"))
		{
			par->T_STRAIN_RATE = atoi(xmlElement->GetText());
		}
		else if (!strcmp(name, "T_HYDROSTATIC PRESSURE"))
		{
			par->T_HYDROSTATIC_PRESSURE = atoi(xmlElement->GetText());
		}
		else if (!strcmp(name, "T_TIME_STEP"))
		{
			par->T_TIME_STEP = atoi(xmlElement->GetText());
		}
		else if (!strcmp(name, "V_E"))
		{
			par->V_E = atof(xmlElement->GetText());
		}
		else if (!strcmp(name, "V_N"))
		{
			par->V_N = atof(xmlElement->GetText());
		}
		else if (!strcmp(name, "V_S"))
		{
			par->V_S = atof(xmlElement->GetText());
		}
		else if (!strcmp(name, "V_W"))
		{
			par->V_W = atof(xmlElement->GetText());
		}
		else if (!strcmp(name, "V_T"))
		{
			par->V_T = atof(xmlElement->GetText());
		}
		else if (!strcmp(name, "V_B"))
		{
			par->V_B = atof(xmlElement->GetText());
		}
		else if (!strcmp(name, "T_DISPERSED_PHASE"))
		{
			par->T_DISPERSED_PHASE = atoi(xmlElement->GetText());
		}
		else if (!strcmp(name, "N_DISPERSED_PHASE"))
		{
			par->N_DISPERSED_PHASE = atoi(xmlElement->GetText());
		}
		else if (!strcmp(name, "T_SOLID_PARTICLE"))
		{
			par->T_SOLID_PARTICLE = atoi(xmlElement->GetText());
		}
		else if (!strcmp(name, "T_SMOOTHING_DENSITY"))
		{
			par->T_SMOOTHING_DENSITY = atoi(xmlElement->GetText());
		}
	}
	par->H = (real)(0.5 * par->XCV / (double)par->NXC);
	par->NC = par->NXC * par->NYC * par->NZC;
	par->I_H = 1.0 / par->H;
	par->DH = 0.01f * par->H;
	par->KNORM = (real)(M_1_PI * pow3(par->I_H));
	par->GKNORM = (real)(M_1_PI * pow5(par->I_H));
	par->DR = par->H / par->HDR;

}

std::string read_phases_from_xml(const char *filename, Parameters* par)
{
	using namespace tinyxml2;

	XMLDocument xmlDocument;
	xmlDocument.LoadFile(filename);

	XMLElement* xmlElementRoot = xmlDocument.FirstChildElement("sph");

	XMLElement* xmlElementPhases = xmlElementRoot->FirstChildElement("domain")->FirstChildElement("phases");


	XMLPrinter printer;
	xmlElementPhases->Accept(&printer);

	return printer.CStr();
}

std::vector<char*> split(char *str, const char* sep)
{
	std::vector<char*> vec;

	char* tokens;
	tokens = strtok(str, sep);
	while (tokens != NULL)
	{
		vec.push_back(tokens);
		tokens = strtok(NULL, sep);
	}

	return vec;
}

void read_particles_from_xml_file(const char* filename, std::vector<Particle>& p, Parameters* par)
{
	using namespace tinyxml2;

	XMLDocument xmlDocument;
	xmlDocument.LoadFile(filename);

	XMLElement* xmlElementRoot = xmlDocument.FirstChildElement("sph");

	XMLElement* xmlElementParticles = xmlElementRoot->FirstChildElement("sph-particles");

	for (XMLElement* xmlElement = xmlElementParticles->FirstChildElement("field"); xmlElement != NULL; xmlElement = xmlElement->NextSiblingElement("field"))
	{
		const char* name = xmlElement->Attribute("name");
		char* stringValues = (char*)xmlElement->GetText();
		std::vector<char*> splitedStringValues = split(stringValues, " ");

		if (!strcmp(name, "id"))
		{
			for (unsigned int i = 0; i < splitedStringValues.size(); i++) p[i].id = atoi(splitedStringValues[i]);
		}
		if (!strcmp(name, "phase-id"))
		{
			for (unsigned int i = 0; i < splitedStringValues.size(); i++) p[i].phaseId = atoi(splitedStringValues[i]);
		}
		if (!strcmp(name, "phase-type"))
		{
			for (unsigned int i = 0; i < splitedStringValues.size(); i++) p[i].phaseType = atoi(splitedStringValues[i]);
		}
		if (!strcmp(name, "x-position"))
		{
			for (unsigned int i = 0; i < splitedStringValues.size(); i++) p[i].pos.x = atof(splitedStringValues[i]);
		}
		if (!strcmp(name, "y-position"))
		{
			for (unsigned int i = 0; i < splitedStringValues.size(); i++) p[i].pos.y = atof(splitedStringValues[i]);
		}
		if (!strcmp(name, "z-position"))
		{
			for (unsigned int i = 0; i < splitedStringValues.size(); i++) p[i].pos.z = atof(splitedStringValues[i]);
		}
		if (!strcmp(name, "x-velocity"))
		{
			for (unsigned int i = 0; i < splitedStringValues.size(); i++) p[i].vel.x = atof(splitedStringValues[i]);
		}
		if (!strcmp(name, "y-velocity"))
		{
			for (unsigned int i = 0; i < splitedStringValues.size(); i++) p[i].vel.y = atof(splitedStringValues[i]);
		}
		if (!strcmp(name, "z-velocity"))
		{
			for (unsigned int i = 0; i < splitedStringValues.size(); i++) p[i].vel.z = atof(splitedStringValues[i]);
		}
		if (!strcmp(name, "mass"))
		{
			for (unsigned int i = 0; i < splitedStringValues.size(); i++) p[i].m = atof(splitedStringValues[i]);
		}
		if (!strcmp(name, "density"))
		{
			for (unsigned int i = 0; i < splitedStringValues.size(); i++) p[i].d = atof(splitedStringValues[i]);
		}
		if (!strcmp(name, "initial density"))
		{
			for (unsigned int i = 0; i < splitedStringValues.size(); i++) p[i].di = atof(splitedStringValues[i]);
		}
		if (!strcmp(name, "dynamic viscosity"))
		{
			for (unsigned int i = 0; i < splitedStringValues.size(); i++) p[i].mi = atof(splitedStringValues[i]);
		}
		if (!strcmp(name, "sound speed"))
		{
			for (unsigned int i = 0; i < splitedStringValues.size(); i++) p[i].s = atof(splitedStringValues[i]);
		}
		if (!strcmp(name, "gamma"))
		{
			for (unsigned int i = 0; i < splitedStringValues.size(); i++) p[i].gamma = atof(splitedStringValues[i]);
		}
		if (!strcmp(name, "equation of state coefficient"))
		{
			for (unsigned int i = 0; i < splitedStringValues.size(); i++) p[i].b = atof(splitedStringValues[i]);
		}

		if (!strcmp(name, "color function"))
		{
			for (unsigned int i = 0; i < splitedStringValues.size(); i++) p[i].c = atof(splitedStringValues[i]);
		}

	}
	for (int i = 0; i < par->N; i++)
	{
		p[i].nu = p[i].mi / p[i].di;
	}
}


void read_particles_dispersed_phase_from_xml_file(const char* filename, std::vector<ParticleDispersedPhase>& pDispersedPhase, Parameters* par)
{
	using namespace tinyxml2;

	XMLDocument xmlDocument;
	xmlDocument.LoadFile(filename);

	XMLElement* xmlElementRoot = xmlDocument.FirstChildElement("sph");

	XMLElement* xmlElementParticles = xmlElementRoot->FirstChildElement("particles-dispersed-phase");

	for (XMLElement* xmlElement = xmlElementParticles->FirstChildElement("field"); xmlElement != NULL; xmlElement = xmlElement->NextSiblingElement("field"))
	{
		const char* name = xmlElement->Attribute("name");
		char* stringValues = (char*)xmlElement->GetText();
		std::vector<char*> splitedStringValues = split(stringValues, " ");

		if (!strcmp(name, "x-position"))
		{
			for (unsigned int i = 0; i < splitedStringValues.size(); i++) pDispersedPhase[i].pos.x = atof(splitedStringValues[i]);
		}
		if (!strcmp(name, "y-position"))
		{
			for (unsigned int i = 0; i < splitedStringValues.size(); i++) pDispersedPhase[i].pos.y = atof(splitedStringValues[i]);
		}
		if (!strcmp(name, "z-position"))
		{
			for (unsigned int i = 0; i < splitedStringValues.size(); i++) pDispersedPhase[i].pos.z = atof(splitedStringValues[i]);
		}
		if (!strcmp(name, "x-velocity"))
		{
			for (unsigned int i = 0; i < splitedStringValues.size(); i++) pDispersedPhase[i].vel.x = atof(splitedStringValues[i]);
		}
		if (!strcmp(name, "y-velocity"))
		{
			for (unsigned int i = 0; i < splitedStringValues.size(); i++) pDispersedPhase[i].vel.y = atof(splitedStringValues[i]);
		}
		if (!strcmp(name, "z-velocity"))
		{
			for (unsigned int i = 0; i < splitedStringValues.size(); i++) pDispersedPhase[i].vel.z = atof(splitedStringValues[i]);
		}
		if (!strcmp(name, "density"))
		{
			for (unsigned int i = 0; i < splitedStringValues.size(); i++) pDispersedPhase[i].d = atof(splitedStringValues[i]);
		}
		if (!strcmp(name, "diameter"))
		{
			for (unsigned int i = 0; i < splitedStringValues.size(); i++) pDispersedPhase[i].dia = atof(splitedStringValues[i]);
		}
		if (!strcmp(name, "x-fluid-velocity"))
		{
			for (unsigned int i = 0; i < splitedStringValues.size(); i++) pDispersedPhase[i].velFl.x = atof(splitedStringValues[i]);
		}
		if (!strcmp(name, "y-fluid-velocity"))
		{
			for (unsigned int i = 0; i < splitedStringValues.size(); i++) pDispersedPhase[i].velFl.y = atof(splitedStringValues[i]);
		}
		if (!strcmp(name, "z-fluid-velocity"))
		{
			for (unsigned int i = 0; i < splitedStringValues.size(); i++) pDispersedPhase[i].velFl.z = atof(splitedStringValues[i]);
		}
		if (!strcmp(name, "fluid-density"))
		{
			for (unsigned int i = 0; i < splitedStringValues.size(); i++) pDispersedPhase[i].dFl = atof(splitedStringValues[i]);
		}
		
	}

}




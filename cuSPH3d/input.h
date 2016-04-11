/*
*  input.h
*
*  Author: Kamil Szewc (kamil.szewc@gmail.com)
*  Modified on: 23-01-2015
*
*/

#if !defined(__INPUT_H__)
#define __INPUT_H__

#include <string>
#include <vector>
#include "sph.h"

void read_parameters_from_xml(const char *filename, Parameters *_par);
std::string read_phases_from_xml(const char *filename, Parameters *_par);
void read_particles_from_xml_file(const char*, std::vector<Particle>&, Parameters*);
void read_particles_dispersed_phase_from_xml_file(const char*, std::vector<ParticleDispersedPhase>&, Parameters*);

#endif
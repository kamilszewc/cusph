/*
*  output.h
*
*  Author: Kamil Szewc (kamil.szewc@gmail.com)
*  Modified on: 23-01-2015
*
*/

#if !defined(__OUTPUT_H__)
#define __OUTPUT_H__

#include <string>
#include <vector>
#include "sph.h"

enum FileFormat {XML, SPH};

void write_raw_phase_data(const char* filename, int phaseId, std::vector<Particle>, Parameters *par);
void write_raw_phases_data(const char* filename, std::vector<Particle>, Parameters* par);
void write_to_file(const char* filename, std::vector<Particle> p, std::vector<ParticleDispersedPhase> pDispersedPhase, Parameters *par, FileFormat fileFormat);

#endif
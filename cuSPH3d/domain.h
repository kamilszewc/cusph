/*
*  domain.h
*
*  Author: Kamil Szewc (kamil.szewc@gmail.com)
*  Modified on: 27-09-2014
*
*/

#if !defined(__DOMAIN_H__)
#define __DOMAIN_H__

#include <iostream>
#include <string>
#include <sstream>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <vector>
#include "output.h"
#include "sph.h"
#include "hlp.h"

class Domain 
{
private:
	std::vector<Particle> _p;
	std::vector<ParticleDispersedPhase> _pDispersedPhase;
	Parameters _par;

	double _time;
	const char* _outputDirectory;

	void SetupDomain(int CASE, double HDR);
	void InitDefaultGeometry(int CASE);
	void InitDefaultParameters(double HDR);
	void InitCase(int CASE);
	void SetParticles();
	void CheckConsistency();
	bool IsConsistentWithGeometry();
	bool IsConsistentWithSearchAlgorithm();
public:
	Domain(const char*);
	Domain(int, double, double);
	Domain(int, double, int);
	Domain(int, double, int, int, int);
	~Domain();

	std::vector<Particle>* GetParticles();
	std::vector<ParticleDispersedPhase>* GetParticlesDispersedPhase();
	Parameters* GetParameters();
	void SetModel(int);
	double* GetTime();
	const char* GetOutputDirectory();
	void SetOutputDirectory(const char*);
	void WriteToFile(const char*, FileFormat);
	void WriteToFile(FileFormat);
	void WriteToFile();
	void WritePhaseDataToRawFile(const char*, int);
	void WritePhaseDataToRawFile(int); 
	void WritePhasesToRawFile(const char*);
	void WritePhasesToRawFile();
	void WritePhasesToRawFiles();
	double GetAndWriteKinetic(const char*);
	double GetAndWriteKinetic();
	int GetSizeOfParticles();
	int GetSizeOfParameters();
};

#endif

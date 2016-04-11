/*
*  domain.h
*
*  Author: Kamil Szewc (kamil.szewc@gmail.com)
*  Modified on: 14-12-2014
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

/**
 * Class of the initial domain configuration.
 */
class Domain {
private:
	std::vector<Particle> _p;
	std::vector<ParticleDispersedPhase> _pDispersedPhase;
	std::vector<Particle> _pDispersedPhaseFluid;
	Parameters _par;

	double _time;
	std::string _phasesXmlInformation;
	const char* _outputDirectory;

	void SetupDomain(int CASE, double HDR);
	void InitDefaultGeometry(int CASE);
	void InitDefaultParameters(int CASE, double HDR);
	void InitDefaultParameters(double HDR);
	void InitCase(int CASE);
	void SetParticles(std::vector<Particle>& p);
	void SetParticleDefaultProperites(std::vector<Particle>& p, int id);
	void CheckConsistency();
	bool IsConsistentWithGeometry();
public:
	/**
	 * @brief Domain constructor
	 * @param[in] filename Filename
	 */
	Domain(char* filename);

	/**
	 * @brief Domain constructor
	 * @param[in] CASE Test case number
	 * @param[in] HDR \f$ h/\Delta r \f$
	 * @param[in] H Smoothing length, \f$ h \f$
	 */
	Domain(int CASE, double HDR, double H);


	/**
	 * @brief Domain constructor
	 * @param[in] CASE Test case number
	 * @param[in] HDR \f$ h/\Delta r \f$
	 * @param[in] MIN_NC Minimal number of cells in the shortest domain direction
	 */
	Domain(int CASE, double HDR, int MIN_NC);     ///< Constructor

	/**
	 * @brief Domain constructor
	 * @param[in] CASE Test case number
	 * @param[in] HDR \f$ h/\Delta r \f$
	 * @param[in] NXC Number of cells in x-direction
	 * @param[in] NYC Number of cells in y-direction
	 */
	Domain(int CASE, double HDR, int NXC, int NYC);

	/**
	 * @brief Destructor
	 */
	~Domain();

	/**
	 * @brief Returns particle array (host)
	 * @return STL vector of Particle
	 */
	std::vector<Particle>* GetParticles();

	/**
	 * @brief Returns dispersed phase particle array (host)
	 * @return STL vector of ParticleDispersedPhase
	 */
	std::vector<ParticleDispersedPhase>* GetParticlesDispersedPhase();

	/**
	 * @brief Returns dispersed phase particle array in Monaghan & Kocharyan (1995) model (host)
	 * @return STL vector of Particle
	 */
	std::vector<Particle>* GetParticlesDispersedPhaseFluid();

	/**
	 * @brief Returns parameters (host)
	 * @return Parameters
	 */
	Parameters* GetParameters();

	/**
	 * @brief Sets model
	 * @param[in] CASE Test case number
	 */
	void SetModel(int CASE);

	/**
	 * @brief Returns time
	 * @return Time
	 */
	double* GetTime();                         ///< Returns time

	/**
	 * @brief Returns name of the output directory
	 * @return Name of the output directory
	 */
	const char* GetOutputDirectory();

	/**
	 * @brief Sets the output directory
	 * @param[in] filename Filename
	 */
	void SetOutputDirectory(const char* filename);

	/**
	 * @brief Writes state to file
	 * @param[in] filename Filename
	 * @param[in] fileFormat Format of the output file (FileFormat)
	 */
	void WriteToFile(const char* filename, FileFormat fileFormat);

	/**
	 * @brief Writes state to the default file
	 * @param[in] fileFormat Format of the output file (FileFormat)
	 */
	void WriteToFile(FileFormat fileFormat);

	/**
	 * @brief Writes state to the default file in default format
	 */
	void WriteToFile();

	/**
	 * @brief Calculates, returns and saves the kinetic energy
	 * @param[in] filename Filename
	 * @return Kinetic energy
	 */
	double GetAndWriteKinetic(const char* filename);

	/**
	 * @brief Calculates, returns and saves (in the defuault file) the kinetic energy
	 * @return Kinetic energy
	 */
	double GetAndWriteKinetic();

	/**
	 * @brief Returns the size of Particle array (in bytes)
	 * @return Size of Particle array
	 */
	int GetSizeOfParticles();

	/**
	 * @brief Returns the size of parameters (in bytes)
	 * @return Size of Parameters
	 */
	int GetSizeOfParameters();
};

#endif

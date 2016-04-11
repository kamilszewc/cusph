/**
* @file output.h
* @author Kamil Szewc (kamil.szewc@gmail.com)
* @since 23-01-2015
*/

#if !defined(__OUTPUT_H__)
#define __OUTPUT_H__

#include <string>
#include <vector>
#include "sph.h"

/**
 * @brief Defines data format
 */
enum FileFormat { XML,  ///< Standard XML data file, sometimes requires hu
				  SPH ///< New format, specific to cuSPH
};

/**
 * @brief Writes data to file
 * @param[in] filename Filename
 * @param[in] p Vector of Particle
 * @param[in] pDispersedPhase Vector of dispersed phase particles
 * @param[in] pDispersedPhaseFluid Vector of dispersed phase particles in 2-fluid formulation
 * @param[in] par Parameters
 * @param[in] fileFormat Format of data: "XML" or "SPH" (FileFormat)
 */
void write_to_file(const char* filename, std::vector<Particle> p, std::vector<ParticleDispersedPhase> pDispersedPhase, std::vector<Particle> pDispersedPhaseFluid, Parameters *par, FileFormat fileFormat);

#endif

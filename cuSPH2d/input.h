/**
* @file input.h
* @author Kamil Szewc (kamil.szewc@gmail.com)
* @since 23-01-2015
*/

#if !defined(__INPUT_H__)
#define __INPUT_H__

#include <string>
#include <vector>
#include "sph.h"

/**
 * @brief Reads parameters from xml file.
 * @param[in] filename XML file's name
 * @param[out] par Parameters
 */
void read_parameters_from_xml(const char* filename, Parameters* par);

/**
 * @brief Reads `phases' string from the XML file
 * @param[in] filename XML file's name
 * @param[in] par Parameters
 * @return String of `phases'
 */
std::string read_phases_from_xml(const char* filename, Parameters* par);

/**
 * @brief Reads particle array from the XML file
 * @param[in] filename XML file's name
 * @param[out] p Particle array
 * @param[in] par Parameters
 */
void read_particles_from_xml_file(const char* filename, std::vector<Particle>& p, Parameters* par);

/**
 * @brief Reads dispersed phase particle (ParticleDispersedPhase) array from the XML file
 * @param[in] filename XML file's name
 * @param[out] p ParticleDispersedPhase array
 * @param[in] par Parameters
 */
void read_particles_dispersed_phase_from_xml_file(const char* filename, std::vector<ParticleDispersedPhase> p, Parameters* par);

#endif

/**
 * @file dispersedParticleManager.h
 *
 *  Created on: 4 mar 2016
 *      Author: kamil
 */
#include <thrust/device_vector.h>
#include <thrust/remove.h>
#include "../../../sph.h"
#include "../../../hlp.h"

#ifndef DISPERSEDPARTICLEMANAGER_H_
#define DISPERSEDPARTICLEMANAGER_H_

class DispersedPhaseFluidParticleManager {
private:
	thrust::device_vector<Particle>* _pDispersedPhaseFluidVector;
	Parameters* _parHost;
	Parameters* _par;
	/**
	 * @brief Sets particle id properties to default
	 * @param[in] id Particle id
	 */
	void SetParticleProperties(Particle& p, Parameters* par, int id, real x, real y, real di, real o);

	void SetNumberOfParticlesOnGPU();
public:
	/**
	 * @brief Constructor for dispersed phase particle manager
	 */
	DispersedPhaseFluidParticleManager(thrust::device_vector<Particle>* pDispersedPhaseFluidVector, Parameters* par, Parameters* parHost);

	/**
	 * @brief Adds particle
	 * @param[in] x X-position
	 * @param[in] y Y-position
	 * @param[in] di Density
	 * @param[in] o Volume fraction
	 */
	void AddParticle(real x, real y, real di, real o);

	/**
	 * @brief Adds a row of particles
	 * @param[in] x0 X-position of row begin
	 * @param[in] y0 Y-position of row begin
	 * @param[in] x1 X-position of row end
	 * @param[in] y1 Y-position of row end
	 * @param[in] di Density
	 * @param[in] o Volume fraction
	 */
	void AddParticlesRow(real x0, real y0, real x1, real y1, real di, real o);

	void DelParticle(int id);


	/**
	 * @brief Deletes particles in given area
	 * @param[in] condition The condition describing the area
	 */
	void DelParticles();

};

#endif /* DISPERSEDPARTICLEMANAGER_H_ */

#if !defined(__CALC_CHEZY_VISCOSITY_CUH__)
#define __CALC_CHEZY_VISCOSITY_CUH__

/**
 * @brief Calculates Chezy viscosity model
 * @param[in,out] p Particle array
 * @param[in] gridParticleIndex Particle-in-cell list of indexes (sorted)
 * @param[in] cellStart Array of beginnings of cell in gridParticleIndex
 * @param[in] cellEnd Array of ends of cell in gridParticleIndex
 * @param[in] par Parameters
 */
__global__ void calcChezyViscosity(Particle *p,
	uint *gridParticleIndex,
	uint *cellStart,
	uint *cellEnd,
	Parameters *par);

#endif

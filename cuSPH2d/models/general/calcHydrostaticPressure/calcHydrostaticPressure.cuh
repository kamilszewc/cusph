#if !defined(__CALC_HYDROSTATIC_PRESSURE_CUH__)
#define __CALC_HYDROSTATIC_PRESSURE_CUH__

/**
 * @brief Calculates hydrostatic pressure
 * @param[in,out] p Particle array
 * @param[in] gridParticleIndex Particle-in-cell list of indexes (sorted)
 * @param[in] cellStart Array of beginnings of cell in gridParticleIndex
 * @param[in] cellEnd Array of ends of cell in gridParticleIndex
 * @param[in] par Parameters
 */
__global__ void calcHydrostaticPressure(Particle *p,
	uint *gridParticleIndex,
	uint *cellStart,
	uint *cellEnd,
	Parameters *par);

#endif

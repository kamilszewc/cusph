#if !defined(__SMOOTHING_DENSITY_CUH__)
#define __SMOOTHING_DENSITY_CUH__

/**
 * @brief Smoothes density
 * @param[in,out] p Particle array
 * @param[in] gridParticleIndex Particle-in-cell list of indexes (sorted)
 * @param[in] cellStart Array of beginnings of cell in gridParticleIndex
 * @param[in] cellEnd Array of ends of cell in gridParticleIndex
 * @param[in] par Parameters
 */
__global__ void smoothingDensity(Particle *p,
	uint *gridParticleIndex,
	uint *cellStart,
	uint *cellEnd,
	Parameters *par);

#endif

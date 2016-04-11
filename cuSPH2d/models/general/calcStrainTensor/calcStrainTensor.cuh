#if !defined(__CALC_STRAIN_TENSOR_CUH__)
#define __CALC_STRAIN_TENSOR_CUH__

/**
 * @brief Calculates strain tensor
 * @param[in,out] p Particle array
 * @param[in] gridParticleIndex Particle-in-cell list of indexes (sorted)
 * @param[in] cellStart Array of beginnings of cell in gridParticleIndex
 * @param[in] cellEnd Array of ends of cell in gridParticleIndex
 * @param[in] par Parameters
 */
__global__ void calcStrainTensor(Particle *p,
	uint *gridParticleIndex,
	uint *cellStart,
	uint *cellEnd,
	Parameters *par);

#endif

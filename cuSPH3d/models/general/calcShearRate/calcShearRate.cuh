#if !defined(__CALC_SHEAR_RATE__)
#define __CALC_SHEAR_RATE__

__global__ void calcShearRate(Particle *pSort,
	uint *gridParticleIndex,
	uint *cellStart,
	uint *cellEnd,
	Parameters *par);

#endif

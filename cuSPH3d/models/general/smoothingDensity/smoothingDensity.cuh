#if !defined(__SMOOTHING_DENSITY_CUH__)
#define __SMOOTHING_DENSITY_CUH__

__global__ void smoothingDensity(Particle *p,
	uint *gridParticleIndex,
	uint *cellStart,
	uint *cellEnd,
	Parameters *par);

#endif
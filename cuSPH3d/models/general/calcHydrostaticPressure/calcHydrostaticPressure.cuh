#if !defined(__HYDROSTATIC_PRESSURE_CUH__)
#define __HYDROSTATIC_PRESSURE_CUH__

__global__ void calcHydrostaticPressure(Particle *pSort,
	uint *gridParticleIndex,
	uint *cellStart,
	uint *cellEnd,
	Parameters *par);

#endif
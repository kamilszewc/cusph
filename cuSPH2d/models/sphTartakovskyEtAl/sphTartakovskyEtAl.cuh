#if !defined(__SPH_TARTAKOVSKY_ET_AL_CUH__)
#define __SPH_TARTAKOVSKY_ET_AL_CUH__

__global__ void calcPressureSTEA(Particle *p, Parameters *par);

__global__ void calcInteractionSTEA(Particle *p,
	uint *gridParticleIndex,
	uint *cellStart,
	uint *cellEnd,
	Parameters *par);


__global__ void calcDensitySTEA(Particle *p,
	uint *gridParticleIndex,
	uint *cellStart,
	uint *cellEnd,
	Parameters *par);

__global__ void calcAdvectionSTEA(Particle *p, Parameters *par);

__global__ void calcInitialDensitySTEA(Particle *p, Parameters *par);

__global__ void calcDeformationSTEA(Particle *p, Parameters *par);

#endif

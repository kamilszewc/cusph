#if !defined(__WCSPH_COLAGROSSI_LANDRINI_CUH__)
#define __WCSPH_COLAGROSSI_LANDRINI_CUH__

__global__ void calcPressureWS(Particle *p, Parameters *par);

__global__ void calcInteractionWS(Particle *p,
	uint *gridParticleIndex,
	uint *cellStart,
	uint *cellEnd,
	Parameters *par);


__global__ void calcXsphWS(Particle *p,
	uint *gridParticleIndex,
	uint *cellStart,
	uint *cellEnd,
	Parameters *par);


__global__ void calcAdvectionWS(Particle *p, Parameters *par, real time);


#endif

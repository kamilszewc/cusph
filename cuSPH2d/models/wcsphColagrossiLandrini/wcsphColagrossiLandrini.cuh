#if !defined(__WCSPH_COLAGROSSI_LANDRINI_CUH__)
#define __WCSPH_COLAGROSSI_LANDRINI_CUH__

__global__ void calcPressureWCL(Particle *p, Parameters *par);

__global__ void calcInteractionWCL(Particle *p,
	uint *gridParticleIndex,
	uint *cellStart,
	uint *cellEnd,
	Parameters *par);


__global__ void calcSmoothedColorWCL(Particle *p,
	uint *gridParticleIndex,
	uint *cellStart,
	uint *cellEnd,
	Parameters *par);

__global__ void calcNormalFromSmoothedColorWCL(Particle *p,
	uint *gridParticleIndex,
	uint *cellStart,
	uint *cellEnd,
	Parameters *par);

__global__ void calcNormalThresholdWCL(Particle *p, Parameters *par);

__global__ void calcCurvatureWCL(Particle *p,
	uint *gridParticleIndex,
	uint *cellStart,
	uint *cellEnd,
	Parameters *par);

__global__ void calcXsphWCL(Particle *p,
	uint *gridParticleIndex,
	uint *cellStart,
	uint *cellEnd,
	Parameters *par);


__global__ void calcAdvectionWCL(Particle *p, Parameters *par, real time);


#endif

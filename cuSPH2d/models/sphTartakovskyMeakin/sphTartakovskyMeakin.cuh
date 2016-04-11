#if !defined(__SPH_TARTAKOVSKY_MEAKIN_CUH__)
#define __SPH_TARTAKOVSKY_MEAKIN_CUH__

__global__ void calcPressureSTM(Particle *p, Parameters *par);

__global__ void calcInteractionSTM(Particle *p,
	uint *gridParticleIndex,
	uint *cellStart,
	uint *cellEnd,
	Parameters *par);

//__global__ void calcDensitySTM(Particle *p,
//		Particle *pSort,
//		uint *gridParticleIndex,
//		uint *cellStart,
//		uint *cellEnd,
//		Parameters *par);

__global__ void calcDensitySTM(Particle *p, Parameters *par);

__global__ void calcAdvectionSTM(Particle *p, Parameters *par);

__global__ void calcDeformationSTM(Particle *p, Parameters *par);

#endif

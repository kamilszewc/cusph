#if !defined(__DISPERSED_PHASE_CUH__)
#define __DISPERSED_PHASE_CUH__

__global__ void calcDispersedPhaseField(Particle *p,
	uint *gridParticleIndex,
	uint *cellStart,
	uint *cellEnd,
	ParticleDispersedPhase *pDispersedPhase,
	Parameters *par);

__global__ void calcDispersedPhaseAdvection(ParticleDispersedPhase *p, Parameters *par);

#endif
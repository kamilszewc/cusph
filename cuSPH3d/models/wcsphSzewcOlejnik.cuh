
__global__ void calcPressureSO(Particle *p, Parameters *par);

__global__ void calcInteractionSO(Particle *p,
	uint *gridParticleIndex,
	uint *cellStart,
	uint *cellEnd,
	Parameters *par);

__global__ void calcAdvectionSO(Particle *p, Parameters *par);


__global__ void calcSoilViscositySO(Particle *p, Parameters *par);


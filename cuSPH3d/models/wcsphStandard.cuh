__global__ void calcPressureWS(Particle *p, Parameters *par);

__global__ void gridHash(uint *gridParticleHash, uint *gridParticleIndex, Particle *p, Parameters *par);

__global__ void reorderDataAndFindCellStart(uint *cellStart,      
	uint   *cellEnd,          
	Particle *pSort,
	uint   *gridParticleHash, 
	uint   *gridParticleIndex,
	Particle *p,   
	uint    numParticles);

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


__global__ void calcAdvectionWS(Particle *p, Parameters *par);


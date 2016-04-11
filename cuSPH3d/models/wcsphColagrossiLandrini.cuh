
__global__ void calcPressureWCL(Particle *p, Parameters *par);

__global__ void gridHash(uint *gridParticleHash, uint *gridParticleIndex, Particle *p, Parameters *par);

__global__ void reorderDataAndFindCellStart(uint *cellStart, 
	uint   *cellEnd,    
	Particle *pSort,
	uint   *gridParticleHash,
	uint   *gridParticleIndex,
	Particle *p,      
	uint    numParticles);

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


__global__ void calcAdvectionWCL(Particle *p, Parameters *par);



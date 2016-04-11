#if !defined(__WCSPH_HU_ADAMS_H__)
#define __WCSPH_HU_ADAMS_H__

__global__ void calcPressureWHA(Particle *p, Parameters *par);

__global__ void gridHash(uint *gridParticleHash, uint *gridParticleIndex, Particle *p, Parameters *par);

__global__ void reorderDataAndFindCellStart(uint *cellStart,  
	uint   *cellEnd,        
	Particle *pSort,
	uint   *gridParticleHash, 
	uint   *gridParticleIndex,
	Particle *p,          
	uint    numParticles);

__global__ void calcInteractionWHA(Particle *p,
	uint *gridParticleIndex,
	uint *cellStart,
	uint *cellEnd,
	Parameters *par);

__global__ void calcDensityWHA(Particle *p,
	uint *gridParticleIndex,
	uint *cellStart,
	uint *cellEnd,
	Parameters *par);

__global__ void calcSmoothedColorWHA(Particle *p,
	uint *gridParticleIndex,
	uint *cellStart,
	uint *cellEnd,
	Parameters *par);

__global__ void calcNormalFromSmoothedColorWHA(Particle *p,
	uint *gridParticleIndex,
	uint *cellStart,
	uint *cellEnd,
	Parameters *par);

__global__ void calcNormalThresholdWHA(Particle *p, Parameters *par);

__global__ void calcSurfaceTensionFromCurvatureWHA(Particle *p,
	uint *gridParticleIndex,
	uint *cellStart,
	uint *cellEnd,
	Parameters *par);

__global__ void calcXsphWHA(Particle *p,
	uint *gridParticleIndex,
	uint *cellStart,
	uint *cellEnd,
	Parameters *par);

__global__ void calcDispersedPhaseField(Particle *p,
										uint *gridParticleIndex,
										uint *cellStart,
										uint *cellEnd,
										ParticleDispersedPhase *pDispersedPhase,
										Parameters *par);

__global__ void calcDispersedPhaseAdvection(ParticleDispersedPhase *p, Parameters *par);


__global__ void calcAdvectionWHA(Particle *p, Parameters *par);

__global__ void calcInitialDensityWHA(Particle *p, Parameters *par);

__global__ void copyPressure(real *p, Particle *particle, Parameters *par);

void sortPressure(real *p, uint numParticles);

void sortParticles(uint *dGridParticleHash, uint *dGridParticleIndex, uint numParticles);

//__global__ void renormalizePressure(real *p, Particle *particle, uint numParticles);

__global__ void calcNormalWHA(Particle *p,
	uint *gridParticleIndex,
	uint *cellStart,
	uint *cellEnd,
	Parameters *par);

#endif
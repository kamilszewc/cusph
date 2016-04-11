/**
 * @file
 */

#if !defined(__WCSPH_STANDARD_H__)
#define __WCSPH_STANDARD_H__

/**
 * Calculates pressure of fluid particles in WCSPH Standard 2-fluid Dispersed Phase model
 */
__global__ void calcPressureWSDP(Particle *p, Parameters *par);

/**
 * Calculates fluid particles interactions in WCSPH Standard 2-fluid Dispersed Phase model
 */
__global__ void calcInteractionWSDP(Particle *p,
	uint *gridParticleIndex,
	uint *cellStart,
	uint *cellEnd,
	Parameters *par);

/**
 * Calculates XSPH for fluid particles in WCSPH Standard 2-fluid Dispersed Phase model
 */
__global__ void calcXsphWSDP(Particle *p,
	uint *gridParticleIndex,
	uint *cellStart,
	uint *cellEnd,
	Parameters *par);

/**
 * Calculates advection of fluid particles in WCSPH Standard 2-fluid Dispersed Phase model
 */
__global__ void calcAdvectionWSDP(Particle *p, Parameters *par, real time);

/**
 * Calculates advection of dispersed phase particles in WCSPH Standard 2-fluid Dispersed Phase model
 */
__global__ void calcAdvectionParticlesWSDP(Particle *pPDPF, Parameters *par);

/**
 * Calculates influence of fluid particles on dispersed phase particles in WCSPH Standard 2-fluid Dispersed Phase model
 */
__global__ void calcInteractionFluidOnParticlesWSDP(Particle *p,
	uint *gridParticleIndex,
	uint *cellStart,
	uint *cellEnd,
	Particle *pPDPF,
	Parameters *par);

/**
 * Calculates influence of dispersed phase particles on fluid particles in WCSPH Standard 2-fluid Dispersed Phase model
 */
__global__ void calcInteractionParticlesOnFluidWSDP(Particle *p,
	uint *gridParticleIndexPDPF,
	uint *cellStartPDPF,
	uint *cellEndPDPF,
	Particle *pPDPF,
	Parameters *par);

/**
 * Calculates particles density and volume fraction in WCSPH Standard 2-fluid Dispersed Phase model
 */
__global__ void calcParticlesDensityAndVolumeFractionWSDP(Particle *p,
	uint *gridParticleIndex,
	uint *cellStart,
	uint *cellEnd,
	Parameters *par);

__global__ void setParticlesWSDP(Particle *p, Parameters *par);

/**
 * Calculates fluid volume fraction in WCSPH Standard 2-fluid Dispersed Phase model
 */
__global__ void calcFluidVolumeFractionWSDP(Particle *p,
	uint *gridParticleIndex,
	uint *cellStart,
	uint *cellEnd,
	Particle *pPDPF,
	Parameters *par);

__global__ void calcSoilViscosityWSDP(Particle *p, Parameters *par);

__global__ void calcTurbulentViscosityWSDP(Particle *p, Parameters *par);



#endif

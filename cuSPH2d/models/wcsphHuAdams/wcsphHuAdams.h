/**
 * @file
 */

#if !defined(__WCSPH_HU_ADAMS_H__)
#define __WCSPH_HU_ADAMS_H__

/**
 * Calculates pressure of fluid particles in the WCSPH Hu and Adams model
 */
__global__ void calcPressureWHA(Particle *p, Parameters *par);

/**
 * Calculates fluid particles interactions in the WCSPH Hu and Adams model
 */
__global__ void calcInteractionWHA(Particle *p,
	uint *gridParticleIndex,
	uint *cellStart,
	uint *cellEnd,
	Parameters *par);

/**
 * Calculates the density field in the WCSPH Hu and Adams model
 */
__global__ void calcDensityWHA(Particle *p,
	uint *gridParticleIndex,
	uint *cellStart,
	uint *cellEnd,
	Parameters *par);

/**
 * Calculates the smoothed color function in the WCSPH Hu and Adams model
 */
__global__ void calcSmoothedColorWHA(Particle *p,
	uint *gridParticleIndex,
	uint *cellStart,
	uint *cellEnd,
	Parameters *par);

/**
 * Calculates the normal vector from smoothed color function in the WCSPH Hu and Adams model
 */
__global__ void calcNormalFromSmoothedColorWHA(Particle *p,
	uint *gridParticleIndex,
	uint *cellStart,
	uint *cellEnd,
	Parameters *par);

/**
 * Calculates the normal vector threshold in the WCSPH Hu and Adams model
 */
__global__ void calcNormalThresholdWHA(Particle *p, Parameters *par);

/**
 * Calculates the surface tension from curvature in the WCSPH Hu and Adams model
 */
__global__ void calcSurfaceTensionFromCurvatureWHA(Particle *p,
	uint *gridParticleIndex,
	uint *cellStart,
	uint *cellEnd,
	Parameters *par);

/**
 * Calculates XSPH in the WCSPH Hu and Adams model
 */
__global__ void calcXsphWHA(Particle *p,
	uint *gridParticleIndex,
	uint *cellStart,
	uint *cellEnd,
	Parameters *par);

/**
 * Calculates advection of fluid particles in the WCSPH Hu and Adams model
 */
__global__ void calcAdvectionWHA(Particle *p, Parameters *par);

/**
 * Calculates mass in the WCSPH Hu and Adams model
 */
__global__ void calcMassWHA(Particle *p,
	uint *gridParticleIndex,
	uint *cellStart,
	uint *cellEnd,
	Parameters *par);

/**
 * Calculates surfactants bulk diffusion in the WCSPH Hu and Adams model
 */
__global__ void calcSurfactantsDiffusionBulkWHA(Particle *p,
	uint *gridParticleIndex,
	uint *cellStart,
	uint *cellEnd,
	Parameters *par);

/**
 * Calculates gradient of surfactant concentration in the WCSPH Hu and Adams model
 */
__global__ void calcSurfactantsGradientWHA(Particle *p,
	uint *gridParticleIndex,
	uint *cellStart,
	uint *cellEnd,
	Parameters *par);

__global__ void calcSurfactantsGradientNormWHA(Particle *p,
	uint *gridParticleIndex,
	uint *cellStart,
	uint *cellEnd,
	Parameters *par);

/**
 * Calculates the interface area (length of the interface) seen by particle in the WCSPH Hu and Adams model
 */
__global__ void calcAreaWHA(Particle *p,
	uint *gridParticleIndex,
	uint *cellStart,
	uint *cellEnd,
	Parameters *par);

/**
 * Calculates the concentration of surfactants in the WCSPH Hu and Adams model
 */
__global__ void calcSurfactantsConcentrationWHA(Particle *p,
	uint *gridParticleIndex,
	uint *cellStart,
	uint *cellEnd,
	Parameters *par);

/**
 * Calculates diffusion of surfactant on interface in the WCSPH Hu and Adams model
 */
__global__ void calcSurfactantsDiffusionWHA(Particle *p,
	uint *gridParticleIndex,
	uint *cellStart,
	uint *cellEnd,
	Parameters *par);

/**
 * Calculates surfactant mass in the WCSPH Hu and Adams model
 */
__global__ void calcSurfactantsMassWHA(Particle *p,
	uint *gridParticleIndex,
	uint *cellStart,
	uint *cellEnd,
	Parameters *par);

/**
 * Calculates the normal vectors in the WCSPH Hu and Adams model
 */
__global__ void calcNormalWHA(Particle *p,
	uint *gridParticleIndex,
	uint *cellStart,
	uint *cellEnd,
	Parameters *par);

/**
 * Calculates the capillary tensor in the WCSPH Hu and Adams model
 */
__global__ void calcCapillaryTensorWHA(Particle *p,
	uint *gridParticleIndex,
	uint *cellStart,
	uint *cellEnd,
	Parameters *par);

/**
 * Calculates surface tension from the capillary tensor in the WCSPH Hu and Adams model
 */
__global__ void calcSurfaceTensionFromCapillaryTensorWHA(Particle *p,
	uint *gridParticleIndex,
	uint *cellStart,
	uint *cellEnd,
	Parameters *par);


#endif

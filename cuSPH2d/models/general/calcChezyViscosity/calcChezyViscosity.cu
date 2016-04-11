/*
 * calcChezyViscosity.cu
 *
 *  Created on: 28-08-2015
 *      Author: Kamil Szewc (kamil.szewc@gmail.com)
 *              Michal Olejnik
 */

#include "../../../sph.h"
#include "../../../hlp.h"
#include "../../../methods/kernels.cuh"
#include "../../../methods/interactions.cuh"

__device__ static real2 interaction(uint i, uint j, real2 dpos, real2 dvel, Particle *p, Parameters *par)
{
	real r = sqrt(pow2(dpos.x) + pow2(dpos.y));
	real q = r * par->I_H;

	if (q < 2.0)
	{
		return MAKE_REAL2( p[j].m * p[j].c / p[j].d , p[j].m / p[j].d);
	}
	else 
	{
		return MAKE_REAL2(0.0, 0.0);
	}
}


__global__ void calcChezyViscosity(Particle *p,
	uint *gridParticleIndex,
	uint *cellStart,
	uint *cellEnd,
	Parameters *par)
{
	uint index = threadIdx.x + blockIdx.x*blockDim.x;
	if (index < par->N) {
		register real2 result = MAKE_REAL2(0.0,0.0);
		#include "../../../methods/interactions/interactionsPositiveOnWallNoSlip.cuh"

		real smoothedColor = result.x / result.y;
		p[index].cs = smoothedColor;

		//real fluidDensity = 1000.0;
		real soilDensity = 1378.0;

		real strainRate = sqrt(4.0) * sqrt( pow2(p[index].str.x) + pow2(p[index].str.y + p[index].str.z) + pow2(p[index].str.w) );
		
		if (strainRate > 0.0)
		{
			real chezyViscosity = soilDensity * 0.01 * ( pow2(p[index].vel.x) + pow2(p[index].vel.y) ) / (strainRate * p[index].d);


			if (smoothedColor < 0.3)
			{
				p[index].nut = smoothedColor * (chezyViscosity - p[index].nu) / 0.3;
			}
			else if (smoothedColor < 0.6)
			{
				p[index].nut = - p[index].nu + chezyViscosity;
			}
			else if (smoothedColor < 0.99)
			{
				p[index].nut = - p[index].nu + chezyViscosity + (smoothedColor - 0.6) * (p[index].nut - chezyViscosity) * (1.0 / 0.4);
			}

			if ( p[index].nut > par->SOIL_MAXIMAL_VISCOSITY )
			{
				p[index].nut = par->SOIL_MAXIMAL_VISCOSITY / p[index].d;
			}
		
		}

	}
}

/*
* calcSurfactantsDiffusionBulkWHA.cu
*
*  Created on: 13-10-2014
*      Author: Kamil Szewc (kamil.szewc@gmail.com)
*/
#include <math.h>
#include "../../sph.h"
#include "../../hlp.h"
#include "../../methods/kernels.cuh"
#include "../../methods/interactions.cuh"
#include <stdio.h>

__device__ static real2 interaction(uint i, uint j, real2 dpos, real2 dvel, Particle *p, Parameters *par)
{
	real q = hypot(dpos.x, dpos.y) * par->I_H;
	if (q < 2.0)
	{
		real k = kern(q, par->I_H);
		return MAKE_REAL2( p[j].mSurf*k, p[j].a*k );
	}
	else
	{
		return MAKE_REAL2( 0.0, 0.0 );
	}
}


__global__ void calcSurfactantsConcentrationWHA(Particle *p,
	uint *gridParticleIndex,
	uint *cellStart,
	uint *cellEnd,
	Parameters *par)
{
	uint index = threadIdx.x + blockIdx.x*blockDim.x;
	if (index < par->N) {
		register real2 result = MAKE_REAL2(0.0,0.0);
		#include "../../methods/interactions/interactionsNegativeOnWallNoSlip.cuh"

		if (result.y > 0.00000001f)
		{
			p[index].cSurf = result.x / result.y;
		}
		else
		{
			p[index].cSurf = 0.0;
		}
	}
}



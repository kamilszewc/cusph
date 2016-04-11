/*
* calcSurfactantsGradientWHA.cu
*
*  Created on: 17-10-2014
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
	real r = hypot(dpos.x, dpos.y);
	real q = r * par->I_H;
	if ((p[j].a > 0.000001f) && (q < 2.0) && ((i != j) || ((i == j) && (q > 0.001f*par->I_H)))) {
		real gkx = grad_of_kern(dpos.x, q, par->I_H);
		real gky = grad_of_kern(dpos.y, q, par->I_H);

		real val = (p[i].cSurf - p[j].cSurf) * p[j].o * p[j].m / p[j].d;

		return MAKE_REAL2(val*gkx, val*gky);
	}
	else
	{
		return MAKE_REAL2(0.0, 0.0);
	}
}

__global__ void calcSurfactantsGradientWHA(Particle *p,
	uint *gridParticleIndex,
	uint *cellStart,
	uint *cellEnd,
	Parameters *par)
{
	uint index = threadIdx.x + blockIdx.x*blockDim.x;
	if (index < par->N) {
		register real2 result = MAKE_REAL2(0.0,0.0);
		#include "../../methods/interactions/interactionsNegativeOnWallNoSlip.cuh"

		p[index].cSurfGrad.x = result.x;
		p[index].cSurfGrad.y = result.y;
	}
}



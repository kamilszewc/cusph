/*
* calcSurfactantsDiffusionWHA.cu
*
*  Created on: 13-10-2014
*      Author: Kamil Szewc (kamil.szewc@gmail.com)
*/
#include <math.h>
#include "../../sph.h"
#include "../../hlp.h"
#include "../../methods/kernels.cuh"
#include "../../methods/interactions.cuh"

__device__ static real interaction(uint i, uint j, real2 dpos, real2 dvel, Particle *p, Parameters *par)
{
	real r = hypot(dpos.x, dpos.y);
	real q = r * par->I_H;
	if ((q < 2.0) && ((i != j) || ((i == j) && (q > 0.001f*par->I_H)))) {
		real gkx = grad_of_kern(dpos.x, q, par->I_H);
		real gky = grad_of_kern(dpos.y, q, par->I_H);

		real valX = (p[i].cSurfGrad.x * p[i].o + p[j].cSurfGrad.x * p[j].o);
		real valY = (p[i].cSurfGrad.y * p[i].o + p[j].cSurfGrad.y * p[j].o);

		return valX * gkx + valY * gky;
	}
	else
	{
		return 0.0;
	}
}


__global__ void calcSurfactantsDiffusionWHA(Particle *p,
	uint *gridParticleIndex,
	uint *cellStart,
	uint *cellEnd,
	Parameters *par)
{
	uint index = threadIdx.x + blockIdx.x*blockDim.x;
	if (index < par->N) {
		register real result = 0.0;
		#include "../../methods/interactions/interactionsNegativeOnWallNoSlip.cuh"

		p[index].mSurf += par->DT * result;
	}
}

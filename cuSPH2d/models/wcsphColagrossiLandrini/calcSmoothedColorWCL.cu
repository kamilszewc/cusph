/*
* calcSmoothedColorWCL.cu
*
*  Created on: 16-12-2013
*      Author: Kamil Szewc (kamil.szewc@gmail.com)
*/
#include "../../sph.h"
#include "../../hlp.h"
#include "../../methods/kernels.cuh"
#include "../../methods/interactions.cuh"

__device__ static real interaction(uint i, uint j, real2 dpos, real2 dvel, Particle *p, Parameters *par)
{
	real q = sqrt(pow2(dpos.x) + pow2(dpos.y)) * par->I_H;

	if (q < 2.0) 
	{
		return p[j].m * kern(q, par->I_H) * p[j].c / p[j].d;
	}
	else 
	{
		return 0.0;
	}
}

__global__ void calcSmoothedColorWCL(Particle *p,
	uint *gridParticleIndex,
	uint *cellStart,
	uint *cellEnd,
	Parameters *par)
{
	uint index = threadIdx.x + blockIdx.x*blockDim.x;
	if (index < par->N) 
	{
		register real result = 0.0;
		#include "../../methods/interactions/interactionsPositiveOnWallNoSlip.cuh"

		p[index].cs = result;
	}
}


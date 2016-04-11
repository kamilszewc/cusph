/*
* calcSmoothedColorWHA.cu
*
*  Created on: 02-09-2013
*      Author: Kamil Szewc (kamil.szewc@gmail.com)
*/
#include <math.h>
#include "../../sph.h"
#include "../../hlp.h"
#include "../../methods/kernels.cuh"
#include "../../methods/interactions.cuh"

__device__ static real interaction(uint i, uint j, real2 dpos, real2 dvel, Particle *p, Parameters *par)
{
	real q = hypot(dpos.x, dpos.y) * par->I_H;

	if (q < 2.0) 
	{
		return p[j].m * kern(q, par->I_H) * p[j].c / p[j].d;
	}
	else {
		return 0.0;
	}
}

__global__ void calcSmoothedColorWHA(Particle *p,
	uint *gridParticleIndex,
	uint *cellStart,
	uint *cellEnd,
	Parameters *par)
{
	uint index = threadIdx.x + blockIdx.x*blockDim.x;
	if (index < par->N) {
		if (par->T_NORMAL_VECTOR == 0)
		{
			p[index].cs = p[index].cs;
		}
		else if (par->T_NORMAL_VECTOR == 1)
		{
			register real result = 0.0;
			#include "../../methods/interactions/interactionsPositiveOnWallNoSlip.cuh"

			p[index].cs = result;
		}
	}
}


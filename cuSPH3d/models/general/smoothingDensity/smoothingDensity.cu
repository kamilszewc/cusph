/*
* smoothingDensity.cu
*
*  Created on: 20-10-2015
*      Author: Kamil Szewc (kamil.szewc@gmail.com)
*              Michal Olejnik
*/

#include "../../../sph.h"
#include "../../../hlp.h"
#include "../../../methods/kernels.cuh"
#include "../../../methods/interactions.cuh"

__device__ static real2 interaction(uint i, uint j, real3 dpos, real3 dvel, Particle *p, Parameters *par)
{
	real r = sqrt(pow2(dpos.x) + pow2(dpos.y) + pow2(dpos.z));
	real q = r * par->I_H;

	if (q < 2.0)
	{
		real k = kern(q, par->I_H);
		return MAKE_REAL2(k, k * p[j].m / p[j].d);
	}
	else
	{
		return MAKE_REAL2(0.0, 0.0);
	}
}


__global__ void smoothingDensity(Particle *p,
	uint *gridParticleIndex,
	uint *cellStart,
	uint *cellEnd,
	Parameters *par)
{
	uint index = threadIdx.x + blockIdx.x*blockDim.x;
	if (index < par->N) {
		register real2 result = MAKE_REAL2(0.0, 0.0);
#include "../../../methods/interactions/interactionsPositiveOnWallNoSlip.cuh"

		p[index].d = p[index].m * result.x / result.y;
	}
}
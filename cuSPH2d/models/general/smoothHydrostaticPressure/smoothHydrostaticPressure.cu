/*
* smoothHydrostaticPressure.cu
*
*  Created on: 10-06-2015
*      Author: Kamil Szewc (kamil.szewc@gmail.com)
*/

#include "../../../sph.h"
#include "../../../hlp.h"
#include "../../../methods/kernels.cuh"
#include "../../../methods/interactions.cuh"

__device__ static real interaction(uint i, uint j, real2 dpos, real2 dvel, Particle *p, Parameters *par)
{
	real r = sqrt(pow2(dpos.x) + pow2(dpos.y));
	real q = r * par->I_H;

	if (q < 2.0) return kern(q, par->I_H)*p[i].ph*p[i].m / p[i].d;
	else return 0.0;
}


__global__ void smoothHydrostaticPressure(Particle *p,
	uint *gridParticleIndex,
	uint *cellStart,
	uint *cellEnd,
	Parameters *par)
{
	uint index = threadIdx.x + blockIdx.x*blockDim.x;
	if (index < par->N) {
		register real result = 0.0;
		#include "../../../methods/interactions/interactionsPositiveOnWallNoSlip.cuh"

		p[index].phs = result;
	}
}

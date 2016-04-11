/*
* calcDensityWHA.cu
*
*  Created on: 02-09-2013
*      Author: Kamil Szewc (kamil.szewc@gmail.com)
*/
#include "../../sph.h"
#include "../../hlp.h"
#include "../../methods/kernels.cuh"
#include "../../methods/interactions.cuh"


__device__ static real interaction(uint i, uint j, real3 dpos, real3 dvel, Particle *p, Parameters *par)
{
	real q = sqrt(pow2(dpos.x) + pow2(dpos.y) + pow2(dpos.z)) * par->I_H;
	if (q < 2.0) return kern(q, par->KNORM);
	else return 0.0;
}


__global__ void calcDensityWHA(Particle *p,
							   uint *gridParticleIndex,
							   uint *cellStart,
							   uint *cellEnd,
							   Parameters *par)
{
	uint index = threadIdx.x + blockIdx.x*blockDim.x;
	if (index < par->N) {
		//real result = interactionsPositiveOnWallNoSlip(index, p, cellStart, cellEnd, par, density);

		register real result = 0.0;
		#include "../../methods/interactions/interactionsPositiveOnWallNoSlip.cuh"

		p[index].d = result * p[index].m;
		p[index].o = pow2(1.0 / result);
	}
}

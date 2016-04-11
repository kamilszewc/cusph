/*
 * calcStrainTensor.cu
 *
 *  Created on: 1-07-2015
 *      Author: Kamil Szewc (kamil.szewc@gmail.com)
 *             
 */

#include "../../../sph.h"
#include "../../../hlp.h"
#include "../../../methods/kernels.cuh"
#include "../../../methods/interactions.cuh"

__device__ static real4 interaction(uint i, uint j, real2 dpos, real2 dvel, Particle *p, Parameters *par)
{
	real r = sqrt(pow2(dpos.x) + pow2(dpos.y));
	real q = r * par->I_H;

	if (q < 2.0)
	{
		real gkx = grad_of_kern(dpos.x, q, par->I_H);
		real gky = grad_of_kern(dpos.y, q, par->I_H);

		return MAKE_REAL4( dvel.x*gkx, dvel.x*gky, dvel.y*gkx, dvel.y*gky );
	}
	else 
	{
		return MAKE_REAL4(0.0, 0.0, 0.0, 0.0);
	}
}


__global__ void calcStrainTensor(Particle *p,
	uint *gridParticleIndex,
	uint *cellStart,
	uint *cellEnd,
	Parameters *par)
{
	uint index = threadIdx.x + blockIdx.x*blockDim.x;
	if (index < par->N) {
		register real4 result = MAKE_REAL4(0.0,0.0,0.0,0.0);
		#include "../../../methods/interactions/interactionsNegativeOnWallNoSlip.cuh"
		//#include "../../../methods/interactions/interactionsNegativeOnWallFreeSlip.cuh"

		p[index].str.x = p[index].m * result.x / p[index].d;
		p[index].str.y = p[index].m * 0.5 * (result.y + result.z) / p[index].d;
		p[index].str.z = p[index].m * 0.5 * (result.z + result.y) / p[index].d;
		p[index].str.w = p[index].m * result.w / p[index].d;
	}
}



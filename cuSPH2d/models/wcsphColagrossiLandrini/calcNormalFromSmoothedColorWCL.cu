/*
* calcNormalFromSmoothedColorWCL.cu
*
*  Created on: 16-12-2013
*      Author: Kamil Szewc (kamil.szewc@gmail.com)
*/
#include "../../sph.h"
#include "../../hlp.h"
#include "../../methods/kernels.cuh"
#include "../../methods/interactions.cuh"

__device__ static real2 interaction(uint i, uint j, real2 dpos, real2 dvel, Particle *p, Parameters *par)
{
	real q = sqrt(pow2(dpos.x) + pow2(dpos.y)) * par->I_H;

	if ((q < 2.0) && ((i != j) || ((i == j) && (q > 0.001f*par->H)))) {
		real gkx = grad_of_kern(dpos.x, q, par->I_H);
		real gky = grad_of_kern(dpos.y, q, par->I_H);
		real put = p[i].cs - p[j].cs;

		return MAKE_REAL2(p[j].m*put*gkx / p[j].d, p[j].m*put*gky / p[j].d);
	}
	else {
		return MAKE_REAL2(0.0, 0.0);
	}
}


__global__ void calcNormalFromSmoothedColorWCL(Particle *p,
	uint *gridParticleIndex,
	uint *cellStart,
	uint *cellEnd,
	Parameters *par)
{
	uint index = threadIdx.x + blockIdx.x*blockDim.x;
	if (index < par->N) {
		register real2 result = MAKE_REAL2(0.0, 0.0);
		#include "../../methods/interactions/interactionsNegativeOnWallNoSlip.cuh"

		p[index].n.x = -result.x;
		p[index].n.y = -result.y;
		p[index].n.z = sqrt(pow2(result.x) + pow2(result.y));
	}
}

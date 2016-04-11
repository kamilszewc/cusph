/*
* calcNormalFromSmoothedColorWCL.cu
*
*  Created on: 02-09-2013
*      Author: Kamil Szewc (kamil.szewc@gmail.com)
*/
#include "../../sph.h"
#include "../../hlp.h"
#include "../../methods/kernels.cuh"
#include "../../methods/interactions.cuh"

__device__ static real3 interaction(uint i, uint j, real3 dpos, real3 dvel, Particle *p, Parameters *par)
{
	real q = sqrt(pow2(dpos.x) + pow2(dpos.y) + pow2(dpos.z)) * par->I_H;

	if ((q < 2.0) && ((i != j) || ((i == j) && (q > 0.001f*par->H)))) 
	{
		real gkx = grad_of_kern(dpos.x, q, par->GKNORM);
		real gky = grad_of_kern(dpos.y, q, par->GKNORM);
		real gkz = grad_of_kern(dpos.z, q, par->GKNORM);
		real put = p[i].cs - p[j].cs;

		return MAKE_REAL3(p[j].m*put*gkx/p[j].d, p[j].m*put*gky/p[j].d, p[j].m*put*gkz/p[j].d);
	}
	else 
	{
		return MAKE_REAL3(0.0, 0.0, 0.0);
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
		register real3 result = MAKE_REAL3(0.0, 0.0, 0.0);
		#include "../../methods/interactions/interactionsNegativeOnWallNoSlip.cuh"

		p[index].n.x = -result.x;
		p[index].n.y = -result.y;
		p[index].n.z = -result.z;
		p[index].n.w = sqrt(pow2(p[index].n.x) + pow2(p[index].n.y) + pow2(p[index].n.z));
	}
}

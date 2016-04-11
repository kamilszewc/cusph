/*
* calcNormalWHA.cu
*
*  Created on: 18-07-2015
*      Author: Kamil Szewc (kamil.szewc@gmail.com)
*/
#include <math.h>
#include "../../sph.h"
#include "../../hlp.h"
#include "../../methods/kernels.cuh"
#include "../../methods/interactions.cuh"

__device__ static real3 interaction(uint i, uint j, real3 dpos, real3 dvel, Particle *p, Parameters *par)
{
	real q = (pow2(dpos.x) + pow2(dpos.y) + pow2(dpos.z)) * par->I_H;

	if (q < 2.0) {
		real gkx = grad_of_kern(dpos.x, q, par->GKNORM);
		real gky = grad_of_kern(dpos.y, q, par->GKNORM);
		real gkz = grad_of_kern(dpos.z, q, par->GKNORM);

		real put = p[i].d / (p[i].d + p[j].d);
		if (p[i].c != p[j].c)
		{
			put += p[j].d / (p[i].d + p[j].d);
		}
		put *= (p[i].o + p[j].o);

		return MAKE_REAL3(put*gkx, put*gky, put*gkz);
	}
	else {
		return MAKE_REAL3(0.0, 0.0, 0.0);
	}
}


__global__ void calcNormalWHA(Particle *p,
	uint *gridParticleIndex,
	uint *cellStart,
	uint *cellEnd,
	Parameters *par)
{
	uint index = threadIdx.x + blockIdx.x*blockDim.x;
	if (index < par->N) 
	{
		register real3 result = MAKE_REAL3(0.0, 0.0, 0.0);
		#include "../../methods/interactions/interactionsNegativeOnWallNoSlip.cuh"

		if (p[index].c == 0)
		{
			p[index].n.x = -result.x * p[index].d / p[index].m;
			p[index].n.y = -result.y * p[index].d / p[index].m;
			p[index].n.z = -result.z * p[index].d / p[index].m;
			p[index].n.w = sqrt(pow2(p[index].n.x) + pow2(p[index].n.y) + pow2(p[index].n.z));
		}
		else
		{
			p[index].n.x = result.x * p[index].d / p[index].m;
			p[index].n.y = result.y * p[index].d / p[index].m;
			p[index].n.z = result.z * p[index].d / p[index].m;
			p[index].n.w = sqrt(pow2(p[index].n.x) + pow2(p[index].n.y) + pow2(p[index].n.z));
		}
	}
}

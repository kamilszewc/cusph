/*
* calcNormalWHA.cu
*
*  Created on: 02-07-2015
*      Author: Kamil Szewc (kamil.szewc@gmail.com)
*/
#include <math.h>
#include "../../sph.h"
#include "../../hlp.h"
#include "../../methods/kernels.cuh"
#include "../../methods/interactions.cuh"

__device__ static real2 interaction(uint i, uint j, real2 dpos, real2 dvel, Particle *p, Parameters *par)
{
	real q = hypot(dpos.x, dpos.y) * par->I_H;

	if (q < 2.0) {
		real gkx = grad_of_kern(dpos.x, q, par->I_H);
		real gky = grad_of_kern(dpos.y, q, par->I_H);
		
		real put = 0.0;
		if (p[i].c != p[j].c)
		{
			put = p[j].d/(p[i].d + p[j].d);
		}
		put *= (p[i].o + p[j].o);

		return MAKE_REAL2(put*gkx, put*gky);
	}
	else {
		return MAKE_REAL2(0.0, 0.0);
	}
}


__global__ void calcNormalWHA(Particle *p,
	uint *gridParticleIndex,
	uint *cellStart,
	uint *cellEnd,
	Parameters *par)
{
	uint index = threadIdx.x + blockIdx.x*blockDim.x;
	if (index < par->N) {
		register real2 result = MAKE_REAL2(0.0,0.0);
		#include "../../methods/interactions/interactionsNegativeOnWallNoSlip.cuh"

		p[index].n.x = result.x * p[index].d / p[index].m;
		p[index].n.y = result.y * p[index].d / p[index].m;
		p[index].n.z = sqrt(pow2(result.x) + pow2(result.y));

		/*uint originalIndex = gridParticleIndex[index];
		if (p[originalIndex].c == 0)
		{
			p[originalIndex].n.x = -result.x * p[originalIndex].d / p[originalIndex].m;
			p[originalIndex].n.y = -result.y * p[originalIndex].d / p[originalIndex].m;
			p[originalIndex].n.z = sqrt(pow2(p[originalIndex].n.x) + pow2(p[originalIndex].n.y));
		}
		else
		{
			p[originalIndex].n.x = result.x * p[originalIndex].d / p[originalIndex].m;
			p[originalIndex].n.y = result.y * p[originalIndex].d / p[originalIndex].m;
			p[originalIndex].n.z = sqrt(pow2(p[originalIndex].n.x) + pow2(p[originalIndex].n.y));
		}*/
		
	}
}

/*
* calcSurfaceTensionFromCapillaryTensorWHA.cu
*
*  Created on: 13-04-2013
*      Author: Kamil Szewc (kamil.szewc@gmail.com)
*/
#include <math.h>
#include "../../sph.h"
#include "../../hlp.h"
#include "../../methods/kernels.cuh"
#include "../../methods/interactions.cuh"

__device__ static real2 interaction(uint i, uint j, real2 dpos, real2 dvel, Particle *p, Parameters *par)
{
	real r = sqrt(pow2(dpos.x) + pow2(dpos.y));
	real q = r * par->I_H;
	if (q < 2.0) {
		real gkx = grad_of_kern(dpos.x, q, par->I_H);
		real gky = grad_of_kern(dpos.y, q, par->I_H);

		real sx = (p[i].ct.x * gkx + p[i].ct.y * gky) * p[i].o + (p[j].ct.x * gkx + p[j].ct.y * gky) * p[j].o;
		real sy = (p[i].ct.z * gkx + p[i].ct.w * gky) * p[i].o + (p[j].ct.z * gkx + p[j].ct.w * gky) * p[j].o;

		return MAKE_REAL2(sx, sy);
	}
	else {
		return MAKE_REAL2(0.0, 0.0);
	}
}


__global__ void calcSurfaceTensionFromCapillaryTensorWHA(Particle *p,
	uint *gridParticleIndex,
	uint *cellStart,
	uint *cellEnd,
	Parameters *par)
{
	uint index = threadIdx.x + blockIdx.x*blockDim.x;
	if (index < par->N) {
		register real2 result = MAKE_REAL2(0.0,0.0);
		#include "../../methods/interactions/interactionsNegativeOnWallNoSlip.cuh"

		p[index].st.x = -result.x / p[index].m;
		p[index].st.y = -result.y / p[index].m;
	}
}

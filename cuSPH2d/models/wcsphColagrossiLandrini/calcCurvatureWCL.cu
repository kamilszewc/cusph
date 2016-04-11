/*
* calcCurvatureWCL.cu
*
*  Created on: 16-12-2013
*      Author: Kamil Szewc (kamil.szewc@gmail.com)
*/

#include "../../sph.h"
#include "../../hlp.h"
#include "../../methods/kernels.cuh"
#include "../../methods/interactions.cuh"

__device__ static real3 interaction(uint i, uint j, real2 dpos, real2 dvel, Particle *p, Parameters *par)
{
	real q = sqrt(pow2(dpos.x) + pow2(dpos.y)) * par->I_H;

	if ((q < 2.0) && ((i != j) || ((i == j) && (q > 0.001f*par->H)))
		&& ((p[i].na == 1) && (p[j].na == 1))) {

		real diffx = ((p[i].n.x / p[i].n.z) - (p[j].n.x / p[j].n.z)) * grad_of_kern(dpos.x, q, par->I_H);
		real diffy = ((p[i].n.y / p[i].n.z) - (p[j].n.y / p[j].n.z)) * grad_of_kern(dpos.y, q, par->I_H);

		return MAKE_REAL3(p[j].m*diffx / p[j].d, p[j].m*diffy / p[j].d, p[j].m*kern(q, par->I_H) / p[j].d);
	}
	else {
		return MAKE_REAL3(0.0, 0.0, 0.0);
	}
}


__global__ void calcCurvatureWCL(Particle *p,
	uint *gridParticleIndex,
	uint *cellStart,
	uint *cellEnd,
	Parameters *par)
{
	uint index = threadIdx.x + blockIdx.x*blockDim.x;
	if (index < par->N) {
		register real3 result = MAKE_REAL3(0.0, 0.0, 0.0);
		#include "../../methods/interactions/interactions_2NegativeOnWallNoSlip_1PositiveOnWallNoSlip.cuh"

		if (result.z > 0.0) {
			p[index].cu = (result.x + result.y) / result.z;
			p[index].cw = result.z;
		}
		real help = par->SURFACE_TENSION * p[index].cu / p[index].d;
		p[index].st.x = help * p[index].n.x;
		p[index].st.y = help * p[index].n.y;
	}
}






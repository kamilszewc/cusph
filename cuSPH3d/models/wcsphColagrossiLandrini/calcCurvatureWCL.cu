/*
* calcCurvatureWCL.cu
*
*  Created on: 02-09-2013
*      Author: Kamil Szewc (kamil.szewc@gmail.com)
*/

#include "../../sph.h"
#include "../../hlp.h"
#include "../../methods/kernels.cuh"
#include "../../methods/interactions.cuh"

__device__ static real4 interaction(uint i, uint j, real3 dpos, real3 dvel, Particle *p, Parameters *par)
{
	real q = sqrt(pow2(dpos.x) + pow2(dpos.y) + pow2(dpos.z)) * par->I_H;

	if ((q < 2.0) && ((i != j) || ((i == j) && (q > 0.001f*par->H)))
		&& ((p[i].na == 1) && (p[j].na == 1))) 
	{
			real diffx = ((p[i].n.x / p[i].n.w) - (p[j].n.x / p[j].n.w)) * grad_of_kern(dpos.x, q, par->GKNORM);
			real diffy = ((p[i].n.y / p[i].n.w) - (p[j].n.y / p[j].n.w)) * grad_of_kern(dpos.y, q, par->GKNORM);
			real diffz = ((p[i].n.z / p[i].n.w) - (p[j].n.z / p[j].n.w)) * grad_of_kern(dpos.z, q, par->GKNORM);

			real vol = p[j].m / p[j].d;
			return MAKE_REAL4(diffx * vol, diffy * vol, diffz * vol, kern(q, par->KNORM) * vol);
	}
	else {
		return MAKE_REAL4(0.0, 0.0, 0.0, 0.0);
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
		register real4 result = MAKE_REAL4(0.0, 0.0, 0.0, 0.0);
		#include "../../methods/interactions/interactionsPositiveOnWallNoSlip.cuh"

		if (result.w > 0.0) {
			p[index].cu = (result.x + result.y + result.z) / result.w;
			p[index].cw = result.w;
		}
		real help = par->SURFACE_TENSION * p[index].cu / p[index].d;
		p[index].st.x = help * p[index].n.x;
		p[index].st.y = help * p[index].n.y;
		p[index].st.z = help * p[index].n.z;
	}
}

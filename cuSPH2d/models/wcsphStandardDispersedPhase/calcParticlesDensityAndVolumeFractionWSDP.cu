/*
 * calcParticleVolumeAndVolumeFractionWS.cu
 *
 *  Created on: 17-12-2013
 *      Author: Kamil Szewc (kamil.szewc@gmail.com)
 */

#include "../../sph.h"
#include "../../hlp.h"
#include "../../methods/kernels.cuh"
#include "../../methods/interactions.cuh"

__device__ static real interaction(uint i, uint j, real2 dpos, real2 dvel, Particle *p, Parameters *par)
{
	real r = sqrt(pow2(dpos.x) + pow2(dpos.y));
	real q = r * par->I_H;

	if (q < 2.0)
	{
		real gkx = grad_of_kern(dpos.x, q, par->I_H);
		real gky = grad_of_kern(dpos.y, q, par->I_H);

		real dens = dvel.x*gkx + dvel.y*gky;

		return p[i].m * p[j].m * dens;
	}
	else {
		return 0.0;
	}
}


__global__ void calcParticlesDensityAndVolumeFractionWSDP(Particle *p,
	uint *gridParticleIndex,
	uint *cellStart,
	uint *cellEnd,
	Parameters *par)
{
	uint index = threadIdx.x + blockIdx.x*blockDim.x;
	if (index < par->N_DISPERSED_PHASE_FLUID) {
		register real result = 0.0;
		#include "../../methods/interactions/interactionsPositiveOnWallNoSlip.cuh"

		p[index].rh_d = result / p[index].m;
		p[index].o = p[index].d / p[index].di;
	}
}

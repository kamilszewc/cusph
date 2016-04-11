/*
 * calcInteractionSO.cu
 *
 *  Created on: 16-02-2015
 *      Author: Kamil Szewc (kamil.szewc@gmail.com)
 *              Michal Olejnik
 */

#include "../../sph.h"
#include "../../hlp.h"
#include "../../methods/kernels.cuh"
#include "../../methods/interactions.cuh"

__device__ static real3 interaction(uint i, uint j, real2 dpos, real2 dvel, real2 dvelSlip, Particle *p, Parameters *par)
{
	real r = sqrt(pow2(dpos.x) + pow2(dpos.y));
	real q = r * par->I_H;

	if (q < 2.0)
	{
		real gkx = grad_of_kern(dpos.x, q, par->I_H);
		real gky = grad_of_kern(dpos.y, q, par->I_H);

		real pres = (p[i].p * pow2(p[i].m/p[i].d)) + (p[j].p * pow2(p[j].m/p[j].d));

		if (par->T_HYDROSTATIC_PRESSURE !=0 ) pres += (p[i].ph * pow2(p[i].m/p[i].d)) + (p[j].ph * pow2(p[j].m/p[j].d));

		//real visc = 8.0f * (p[i].nu + p[i].nut + p[j].nu + p[j].nut) * (dvel.x*dpos.x + dvel.y*dpos.y) / (pow2(r) * (p[i].d + p[j].d));
		real visc = p[i].m * p[j].m * (p[i].d*(p[i].nu+p[i].nut) + p[j].d*(p[j].nu+p[j].nut)) * (dpos.x*gkx + dpos.y*gky) / ((pow2(r) + 0.01*pow2(par->H)) * p[i].d * p[j].d);

		real dens = dvelSlip.x*gkx + dvelSlip.y*gky;

		return MAKE_REAL3( visc*dvel.x - pres*gkx, visc*dvel.y - pres*gky, dens);
	}
	else {
		return MAKE_REAL3(0.0, 0.0, 0.0);
	}
}


__global__ void calcInteractionSO(Particle *p,
	uint *gridParticleIndex,
	uint *cellStart,
	uint *cellEnd,
	Parameters *par)
{
	uint index = threadIdx.x + blockIdx.x*blockDim.x;
	if (index < par->N) {
		register real3 result = MAKE_REAL3(0.0,0.0,0.0);
		#include "../../methods/interactions/interactions_2NegativeOnWallNoSlip_1PositiveOnWallFreeSlip.cuh"
		//#include "../../methods/interactions/interactions_2NegativeOnWallFreeSlip_1PositiveOnWallFreeSlip.cuh"

		p[index].rh_vel.x = result.x / p[index].m;
		p[index].rh_vel.y = result.y / p[index].m;
		p[index].rh_d = result.z * p[index].m;
	}
}



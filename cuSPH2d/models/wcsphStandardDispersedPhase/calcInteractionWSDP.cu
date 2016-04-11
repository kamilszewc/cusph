/*
 * calcInteractionWS.cu
 *
 *  Created on: 17-12-2013
 *      Author: Kamil Szewc (kamil.szewc@gmail.com)
 */

#include "../../sph.h"
#include "../../hlp.h"
#include "../../methods/kernels.cuh"
#include "../../methods/interactions.cuh"

__device__ static real4 interaction(uint i, uint j, real2 dpos, real2 dvel, real2 dvelSlip, Particle *p, Parameters *par)
{
	real r = sqrt(pow2(dpos.x) + pow2(dpos.y));
	real q = r * par->I_H;

	if (q < 2.0)
	{
		real gkx = grad_of_kern(dpos.x, q, par->I_H);
		real gky = grad_of_kern(dpos.y, q, par->I_H);

		real pres = (p[i].p*p[i].o / pow2(p[i].d)) + (p[j].p*p[j].o / pow2(p[j].d));

		//real visc = 8.0f * (p[i].nu + p[j].nu) * (dvel.x*dpos.x + dvel.y*dpos.y) / (pow2(r) * (p[i].d + p[j].d));
		real visc = (p[i].d*(p[i].nu + p[i].nut) + p[j].d*(p[j].nu+p[j].nut)) * (dpos.x*gkx + dpos.y*gky) / ((pow2(r) + 0.01*pow2(par->H)) * p[i].d * p[j].d);

		real dens = dvelSlip.x*gkx + dvelSlip.y*gky;

		return MAKE_REAL4(p[i].m * p[j].m * (visc * dvel.x - pres * gkx), p[i].m * p[j].m * (visc * dvel.y - pres * gky), p[i].m * p[j].m * dens, p[i].m * pow2(p[j].m) * dens / p[j].d);
	}
	else {
		return MAKE_REAL4(0.0, 0.0, 0.0, 0.0);
	}
}


__global__ void calcInteractionWSDP(Particle *p,
	uint *gridParticleIndex,
	uint *cellStart,
	uint *cellEnd,
	Parameters *par)
{
	uint index = threadIdx.x + blockIdx.x*blockDim.x;
	if (index < par->N) {
		register real4 result = MAKE_REAL4(0.0, 0.0, 0.0, 0.0);
		#include "../../methods/interactions/interactions_2NegativeOnWallNoSlip_2PositiveOnWallFreeSlip.cuh"

		if (par->T_DISPERSED_PHASE_FLUID == 0)
		{
			p[index].rh_vel.x = 0.0;
			p[index].rh_vel.y = 0.0;
		}
		p[index].rh_vel.x += result.x / p[index].m;
		p[index].rh_vel.y += result.y / p[index].m;
		p[index].rh_d = result.z / p[index].m;
		p[index].a = result.w / p[index].m;
	}
}



/*
 * calcInteractionSO.cu
 *
 *  Created on: 17-02-2015
 *      Author: Kamil Szewc (kamil.szewc@gmail.com)
 *              Michal Olejnik
 */

#include "../../sph.h"
#include "../../hlp.h"
#include "../../methods/kernels.cuh"
#include "../../methods/interactions.cuh"

__device__ static real4 interaction(uint i, uint j, real3 dpos, real3 dvel, real3 dvelSlip, Particle *p, Parameters *par)
{
	real r = sqrt(pow2(dpos.x) + pow2(dpos.y) + pow2(dpos.z));
	real q = r * par->I_H;

	if (q < 2.0)
	{
		real gkx = grad_of_kern(dpos.x, q, par->GKNORM);
		real gky = grad_of_kern(dpos.y, q, par->GKNORM);
		real gkz = grad_of_kern(dpos.z, q, par->GKNORM);

		real pres = (p[i].p * pow2(p[i].m/p[i].d)) + (p[j].p * pow2(p[j].m/p[j].d));

		if (par->T_HYDROSTATIC_PRESSURE != 0) pres += (p[i].ph * pow2(p[i].m/p[i].d)) + (p[j].ph * pow2(p[j].m/p[j].d));

		real visc = p[i].m * p[j].m * (p[i].d*(p[i].nu+p[i].nut) + p[j].d*(p[j].nu+p[j].nut)) * (dpos.x*gkx + dpos.y*gky + dpos.z*gkz) / ((pow2(r) + 0.01*pow2(par->H)) * p[i].d * p[j].d);
		//real visc = p[i].m * p[j].m * 8.0*(p[i].nu + p[j].nu + p[i].nut + p[j].nut) * (dvelSlip.x*dpos.x + dvelSlip.y*dpos.y + dvelSlip.z*dpos.z) / ((r*r + 0.01*pow2(par->H)) * (p[i].d + p[j].d));

		real dens = dvelSlip.x*gkx + dvelSlip.y*gky + dvelSlip.z*gkz;

		return MAKE_REAL4(visc*dvel.x - pres*gkx, visc*dvel.y - pres*gky, visc*dvel.z - pres*gkz, dens);
		//return MAKE_REAL4(visc*dvelSlip.x - pres*gkx, visc*dvelSlip.y - pres*gky, visc*dvelSlip.z - pres*gkz, dens);
		//return MAKE_REAL4( (visc - pres)*gkx, (visc - pres)*gky, (visc - pres)*gkz, dens);
	}
	else {
		return MAKE_REAL4(0.0, 0.0, 0.0, 0.0);
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
		register real4 result = MAKE_REAL4(0.0,0.0,0.0,0.0);
		//#include "../../methods/interactions/interactions_3NegativeOnWallNoSlip_1PositiveOnWallFreeSlip.cuh"
		#include "../../methods/interactions/interactions_3NegativeOnWallNoSlip_1PositiveOnWallFreeSlip.cuh"

		p[index].rh_vel.x = result.x / p[index].m;
		p[index].rh_vel.y = result.y / p[index].m;
		p[index].rh_vel.z = result.z / p[index].m;
		p[index].rh_d = result.w * p[index].m;
	}
}



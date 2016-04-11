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

__device__ static real4 interaction(uint i, uint j, real3 dpos, real3 dvel, real3 dvelSlip, Particle *p, Parameters *par)
{
	real r = sqrt(pow2(dpos.x) + pow2(dpos.y) + pow2(dpos.z));
	real q = r * par->I_H;
	if (q < 2.0) {
		real gkx = grad_of_kern(dpos.x, q, par->GKNORM);
		real gky = grad_of_kern(dpos.y, q, par->GKNORM);
        real gkz = grad_of_kern(dpos.z, q, par->GKNORM);

		real pres = (p[i].p / pow2(p[i].d)) + (p[j].p / pow2(p[j].d));

		//real visc = 8.0 * (p[i].nu + p[j].nu) * (dvel.x*dpos.x + dvel.y*dpos.y + dvel.z*dpos.z) / ( (pow2(r) + 0.0001*pow2(par->H)) * (p[i].d + p[j].d));
		real visc = (p[i].d*p[i].nu + p[j].d*p[j].nu) * (dpos.x*gkx + dpos.y*gky + dpos.z*gkz) / ( (pow2(r)+0.01*pow2(par->H)) * p[i].d * p[j].d);

		real dens = dvelSlip.x*gkx + dvelSlip.y*gky + dvelSlip.z*gkz;

		return MAKE_REAL4(p[j].m*(visc * dvel.x - pres * gkx), p[j].m*(visc * dvel.y - pres * gky), p[j].m*(visc * dvel.z - pres * gkz), p[j].m*dens);
	}
	else {
		return MAKE_REAL4(0.0, 0.0, 0.0, 0.0);
	}
}


__global__ void calcInteractionWS(Particle *p,
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

		p[index].rh_vel.x = result.x;
		p[index].rh_vel.y = result.y;
        p[index].rh_vel.z = result.z;
		p[index].rh_d = result.w;
	}
}



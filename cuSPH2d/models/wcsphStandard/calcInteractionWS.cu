/*
 * calcInteractionWS.cu
 *
 *  Created on: 16-12-2013
 *      Author: Kamil Szewc (kamil.szewc@gmail.com)
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

		real pres = (p[i].p + p[j].p) / (p[i].d * p[j].d);
		if ((par->T_INTERFACE_CORRECTION == 1) && (p[i].c != p[j].c)) pres += par->INTERFACE_CORRECTION * fabs(p[i].p + p[j].p) / (p[i].d*p[j].d);

		real visc = 8.0f*(p[i].nu + p[j].nu) * (dvel.x*dpos.x + dvel.y*dpos.y) / ( (r*r + 0.01*par->H*par->H) *  (p[i].d + p[j].d));
		//real visc = (p[i].d*p[i].nu + p[j].d*p[j].nu) * (dpos.x*gkx + dpos.y*gky) / ((pow2(r) + 0.01*pow2(par->H)) * p[i].d * p[j].d);

		real dens = dvelSlip.x*gkx + dvelSlip.y*gky;

		//return MAKE_REAL3(p[i].m*p[j].m*(visc * dvel.x - pres * gkx), p[i].m*p[j].m*(visc * dvel.y - pres * gky), dens*p[i].m*p[j].m/(p[i].d*p[j].d) );

		return MAKE_REAL3(p[i].m*p[j].m*(visc - pres) * gkx, p[i].m*p[j].m*(visc - pres) * gky, dens*p[i].m*p[j].m/(p[i].d*p[j].d) );
	}
	else 
	{
		return MAKE_REAL3(0.0, 0.0, 0.0);
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
		register real3 result = MAKE_REAL3(0.0,0.0,0.0);
		#include "../../methods/interactions/interactions_2NegativeOnWallNoSlip_1PositiveOnWallFreeSlip.cuh"

		p[index].rh_vel.x = result.x / p[index].m;
		p[index].rh_vel.y = result.y / p[index].m;
		p[index].rh_d = pow2(p[index].d) * result.z / p[index].m;
	}
}



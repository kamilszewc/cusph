/*
* calcInteractionSTM.cu
*
*  Created on: 21-07-2014
*      Author: Kamil Szewc (kamil.szewc@gmail.com)
*/
#include "../../sph.h"
#include "../../hlp.h"
#include "../../methods/kernels.cuh"
#include "../../methods/interactions.cuh"
#include <stdio.h>


__device__ static real3 interaction(uint i, uint j, real2 dpos, real2 dvel, Particle *p, Parameters *par)
{
	real r = sqrt(pow2(dpos.x) + pow2(dpos.y));
	real q = r * par->I_H;
	if ((q < 2.0) && ((i != j) || ((i == j) && (q > 0.001f*par->H)))) {
		real gkx_long = grad_of_kern(dpos.x, q, par->I_H);
		real gky_long = grad_of_kern(dpos.y, q, par->I_H);
		real gkx = grad_of_kernel_half(dpos.x, q, par->I_H);
		real gky = grad_of_kernel_half(dpos.y, q, par->I_H);

		real pres = (p[i].p / pow2(p[i].d)) + (p[j].p / pow2(p[j].d));

		real visc = 8.0f * (p[i].nu + p[j].nu) * (dvel.x*dpos.x + dvel.y*dpos.y) / (pow2(r) * (p[i].d + p[j].d));

		real c2 = 2.0;
		real accel = 2.0 * c2;

		real s_ij = 0.1f;
		if (p[i].c != p[j].c)
		{
			s_ij = 0.05f;
		}
		real force_x = 0.0;
		real force_y = 0.0;
		if ( q <= 2.0 )
		{
			force_x = -s_ij * cos( 0.25f * M_PI * q ) * dpos.x / r;
			force_y = -s_ij * cos( 0.25f * M_PI * q ) * dpos.y / r;
		}

		real dens = dvel.x*gkx + dvel.y*gky;

		return MAKE_REAL3(p[j].m*(visc - pres)*gkx + force_x / p[j].m + accel*p[j].m*gkx_long, p[j].m*(visc - pres)*gky + force_y / p[j].m + accel*p[j].m*gky_long, p[j].m*dens);
	}
	else {
		return MAKE_REAL3(0.0, 0.0, 0.0);
	}
}


__global__ void calcInteractionSTM(Particle *p,
	uint *gridParticleIndex,
	uint *cellStart,
	uint *cellEnd,
	Parameters *par)
{
	uint index = threadIdx.x + blockIdx.x*blockDim.x;
	if (index < par->N) {
		register real3 result = MAKE_REAL3(0.0, 0.0, 0.0);
		#include "../../methods/interactions/interactions_2NegativeOnWallNoSlip_1PositiveOnWallNoSlip.cuh"

		p[index].rh_vel.x = result.x;
		p[index].rh_vel.y = result.y;
		p[index].rh_d = result.z;
	}
}


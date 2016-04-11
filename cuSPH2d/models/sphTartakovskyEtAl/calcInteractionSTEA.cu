/*
* calcInteractionSTEA.cu
*
*  Created on: 21-07-2014
*      Author: Kamil Szewc (kamil.szewc@gmail.com)
*/
#include "../../sph.h"
#include "../../hlp.h"
#include "../../methods/kernels.cuh"
#include "../../methods/interactions.cuh"
#include <stdio.h>
#include <math.h>

__device__ static real4 interaction(uint i, uint j, real2 dpos, real2 dvel, Particle *p, Parameters *par)
{
	//real r = sqrt(pow2(dpos.x) + pow2(dpos.y));
	real r = hypot(dpos.x, dpos.y);
	real q = r * par->I_H;
	if ((q < 2.0) && ((i != j) || ((i == j) && (q > 0.001f*par->H)))) {
		//real gkx = grad_of_kern(dpos.x, q, par->I_H);
		//real gky = grad_of_kern(dpos.y, q, par->I_H);
		real gkx_long = grad_of_kern(dpos.x, q, par->I_H);
		real gky_long = grad_of_kern(dpos.y, q, par->I_H);
		real gkx = grad_of_kernel_half(dpos.x, q, par->I_H);
		real gky = grad_of_kernel_half(dpos.y, q, par->I_H);

		real pres = (p[i].p/pow2(p[i].o)) + (p[j].p/pow2(p[j].o));
	
		real visc = 2.0*(p[i].mi*p[j].mi)*( 1.0/pow2(p[i].o) + 1.0/pow2(p[j].o) ) * (dpos.x*gkx + dpos.y*gky) / (r*r * (p[i].mi + p[j].mi));

		real c2 = 2.0;
		if (p[i].c != p[j].c)
		{
			c2 = 2.0;
		}
		real accel = c2 * (pow2(p[i].m) + pow2(p[j].m));

		real force_x = 0.0;
		real force_y = 0.0;
		real s_ij = 0.1f;
		if (p[i].c != p[j].c)
		{
			s_ij = 0.05f;
		}
		if ( q <= 2.0 )
		{
			force_x = -s_ij * cos( 0.25f * M_PI * q ) * dpos.x / r;
			force_y = -s_ij * cos( 0.25f * M_PI * q ) * dpos.y / r;
		}
		return MAKE_REAL4( visc * dvel.x - pres * gkx, visc * dvel.y - pres * gky, force_x + accel*gkx_long, force_y + accel*gky_long);//2.0 * c2 * k * gkx, 2.0 * c2 * k * gky);
	}
	else {
		return MAKE_REAL4( 0.0, 0.0, 0.0, 0.0 );
	}
}



__global__ void calcInteractionSTEA(Particle *p,
	uint *gridParticleIndex,
	uint *cellStart,
	uint *cellEnd,
	Parameters *par)
{
	uint index = threadIdx.x + blockIdx.x*blockDim.x;
	if (index < par->N) {
		register real4 result = MAKE_REAL4(0.0, 0.0, 0.0, 0.0);
		#include "../../methods/interactions/interactionsNegativeOnWallNoSlip.cuh"

		p[index].rh_vel.x = result.x / p[index].m;
		p[index].rh_vel.y = result.y / p[index].m;
		p[index].rh_vel.x += result.z / p[index].m;
		p[index].rh_vel.y += result.w / p[index].m;

	}
}


/*
* calcXsphWCL.cu
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

	if (q < 2.0)
	{
		real g = kern(q, par->I_H);

		switch (par->T_XSPH) {
		case 1:
			return MAKE_REAL3(p[j].m*p[j].vel.x*g / p[j].d, p[j].m*p[j].vel.y*g / p[j].d, g*p[j].m / p[j].d);
		case 2:
			if (p[i].c == p[j].c) {
				return MAKE_REAL3(p[j].m*p[j].vel.x*g / p[j].d, p[j].m*p[j].vel.y*g / p[j].d, g*p[j].m / p[j].d);
			}
			else
			{
				return MAKE_REAL3(0.0, 0.0, 0.0);
			}
		case 3:
			return MAKE_REAL3(p[j].m*p[j].vel.x*g, p[j].m*p[j].vel.y*g, g*p[j].m);
		default:
			return MAKE_REAL3(0.0, 0.0, 0.0);
		}

	}
	else
	{
		return MAKE_REAL3(0.0, 0.0, 0.0);
	}
}



__global__ void calcXsphWCL(Particle *p,
	uint *gridParticleIndex,
	uint *cellStart,
	uint *cellEnd,
	Parameters *par)
{
	uint index = threadIdx.x + blockIdx.x*blockDim.x;
	if (index < par->N)
	{
		register real3 result = MAKE_REAL3(0.0, 0.0, 0.0);
		#include "../../methods/interactions/interactionsPositiveOnWallNoSlip.cuh"

		p[index].rh_pos.x = result.x / result.z;
		p[index].rh_pos.y = result.y / result.z;
	}
}

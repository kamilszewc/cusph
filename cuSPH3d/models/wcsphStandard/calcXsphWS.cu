/*
* calcXsphWS.cu
*
*  Created on: 17-09-2013
*      Author: Kamil Szewc (kamil.szewc@gmail.com)
*/

#include "../../sph.h"
#include "../../hlp.h"
#include "../../methods/kernels.cuh"
#include "../../methods/interactions.cuh"

__device__ static real4 interaction(uint i, uint j, real3 dpos, real3 dvel, Particle *p, Parameters *par)
{
	real q = sqrt(pow2(dpos.x) + pow2(dpos.y) + pow2(dpos.z)) * par->I_H;

	if (q < 2.0) {
		real g = kern(q, par->KNORM);

		switch (par->T_XSPH) {
		case 1:	return MAKE_REAL4(p[j].m*p[j].vel.x*g / p[j].d, p[j].m*p[j].vel.y*g / p[j].d, p[j].m*p[j].vel.z*g / p[j].d, g*p[j].m / p[j].d);
		case 2:	if (p[i].c == p[j].c) {
			return MAKE_REAL4(p[j].m*p[j].vel.x*g / p[j].d, p[j].m*p[j].vel.y*g / p[j].d, p[j].m*p[j].vel.z*g / p[j].d, g*p[j].m / p[j].d);
				}
				else {
					return MAKE_REAL4(0.0, 0.0, 0.0, 0.0);
				}
		case 3: return MAKE_REAL4(p[j].m*p[j].vel.x*g, p[j].m*p[j].vel.y*g, p[j].m*p[j].vel.z*g, g*p[j].m);
		default:
			return MAKE_REAL4(0.0, 0.0, 0.0, 0.0);
		}

	}
	else {
		return MAKE_REAL4(0.0, 0.0, 0.0, 0.0);
	}
}



__global__ void calcXsphWS(Particle *p,
							uint *gridParticleIndex,
							uint *cellStart,
							uint *cellEnd,
							Parameters *par)
{
	uint index = threadIdx.x + blockIdx.x*blockDim.x;
	if (index < par->N) {
		register real4 result = MAKE_REAL4(0.0,0.0,0.0,0.0);
		#include "../../methods/interactions/interactionsPositiveOnWallNoSlip.cuh"

		p[index].rh_pos.x = result.x / result.w;
		p[index].rh_pos.y = result.y / result.w;
		p[index].rh_pos.z = result.z / result.w;
	}
}
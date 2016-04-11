/*
 * calcShearRate.cu
 *
 *  Created on: 23-07-2015
 *      Author: Kamil Szewc (kamil.szewc@gmail.com)
 *             
 */

#include "../../../sph.h"
#include "../../../hlp.h"
#include "../../../methods/kernels.cuh"
#include "../../../methods/interactions.cuh"
#include <stdio.h>

__device__ static real6 interaction(uint i, uint j, real3 dpos, real3 dvel, Particle *p, Parameters *par)
{
	real r = sqrt(pow2(dpos.x) + pow2(dpos.y) + pow2(dpos.z));
	real q = r * par->I_H;

	if (q < 2.0)
	{
		real gkx = grad_of_kern(dpos.x, q, par->GKNORM);
		real gky = grad_of_kern(dpos.y, q, par->GKNORM);
		real gkz = grad_of_kern(dpos.z, q, par->GKNORM);
		
		if (par->T_STRAIN_RATE == 0)
		{
			real u12 = 0.5 * (dvel.x*gky + dvel.y*gkx);
			real u13 = 0.5 * (dvel.x*gkz + dvel.z*gkx);
			real u23 = 0.5 * (dvel.y*gkz + dvel.z*gky);

			return MAKE_REAL6(dvel.x*gkx, dvel.y*gky, dvel.z*gkz, u12, u13, u23);
		}
		else
		{
			real s =  (dpos.x*gkx + dpos.y*gky + dpos.z*gkz);// / ((pow2(r) + 0.01*pow2(par->H))* p[i].d * p[j].d);
			//real s = 1.0;//0.5 * p[j].m * p[i].m * (p[i].d + p[j].d) * (pow2(dvel.x) + pow2(dvel.y) + pow2(dvel.z)) * (dpos.x*gkx + dpos.y*gky + dpos.z*gkz) / ((pow2(r) + 0.01*pow2(par->H)) * p[i].d * p[j].d);

			return MAKE_REAL6(s, 0.0, 0.0, 0.0, 0.0, 0.0);
		}
	}
	else 
	{
		return MAKE_REAL6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
	}
}


__global__ void calcShearRate(Particle *p,
	uint *gridParticleIndex,
	uint *cellStart,
	uint *cellEnd,
	Parameters *par)
{
	uint index = threadIdx.x + blockIdx.x*blockDim.x;
	if (index < par->N) {
		real6 result = MAKE_REAL6(0.0,0.0,0.0,0.0,0.0,0.0);
		#include "../../../methods/interactions/interactionsNegativeOnWallNoSlip.cuh"
		//#include "../../methods/interactions/interactionsNegativeOnWallFreeSlip.cuh"

		if (par->T_STRAIN_RATE == 0)
		{
			result.x = p[index].m * result.x / p[index].d;
			result.y = p[index].m * result.y / p[index].d;
			result.z = p[index].m * result.z / p[index].d;
			result.u = p[index].m * result.u / p[index].d;
			result.v = p[index].m * result.v / p[index].d;
			result.w = p[index].m * result.w / p[index].d;

			real strainRate = sqrt(2.0) * sqrt(pow2(result.x) + pow2(result.y) + pow2(result.z)
				+ pow2(result.u) + pow2(result.v) + pow2(result.w)
				+ pow2(result.u) + pow2(result.v) + pow2(result.w));

			//real strainRate = sqrt(4.0) * sqrt( pow2(result.u) + pow2(result.v) + pow2(result.w) ); 

			p[index].str = strainRate;
		}
		else
		{
			p[index].str = sqrt(2.0*result.x/p[index].m);
		}
	}
}



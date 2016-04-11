/*
* calcSurfactantsDiffusionBulkWHA.cu
*
*  Created on: 13-10-2014
*      Author: Kamil Szewc (kamil.szewc@gmail.com)
*/
#include <math.h>
#include "../../sph.h"
#include "../../hlp.h"
#include "../../methods/kernels.cuh"
#include "../../methods/interactions.cuh"
#include <stdio.h>

__device__ static real interaction(uint i, uint j, real2 dpos, real2 dvel, Particle *p, Parameters *par)
{
	real r = hypot(dpos.x, dpos.y);
	real q = r * par->I_H;
	if ((q < 2.0) && ((i != j) || ((i == j) && (q > 0.001f*par->I_H)))) {
		real gkx = grad_of_kern(dpos.x, q, par->I_H);
		real gky = grad_of_kern(dpos.y, q, par->I_H);

		return 2.0*(p[i].dBulk*p[j].dBulk)*(p[i].o + p[j].o) * (p[i].cBulk - p[j].cBulk) * (dpos.x*gkx + dpos.y*gky) / (r*r * (p[i].dBulk + p[j].dBulk));
	}
	else 
	{
		return 0.0;
	}
}


__global__ void calcSurfactantsDiffusionBulkWHA(Particle *p,
	uint *gridParticleIndex,
	uint *cellStart,
	uint *cellEnd,
	Parameters *par)
{
	uint index = threadIdx.x + blockIdx.x*blockDim.x;
	if (index < par->N) {
		register real result = 0.0;
		#include "../../methods/interactions/interactionsNegativeOnWallNoSlip.cuh"

		p[index].mBulk += par->DT * result;
		p[index].cBulk = p[index].mBulk * p[index].d / p[index].m;
	}
}



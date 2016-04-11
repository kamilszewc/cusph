/*
* calcInteractionWHA.cu
*
*  Created on: 13-04-2013
*      Author: Kamil Szewc (kamil.szewc@gmail.com)
*/
#include "../../sph.h"
#include "../../hlp.h"
#include "../../methods/kernels.cuh"
#include "../../methods/interactions.cuh"

__device__ static real3 interaction(uint i, uint j, real3 dpos, real3 dvel, Particle *p, Parameters *par)
{
	real r = sqrt(pow2(dpos.x) + pow2(dpos.y) + pow2(dpos.z));
	real q = r * par->I_H;
	if (q < 2.0) {
		real gkx = grad_of_kern(dpos.x, q, par->GKNORM);
		real gky = grad_of_kern(dpos.y, q, par->GKNORM);
		real gkz = grad_of_kern(dpos.z, q, par->GKNORM);

		real pres = (p[i].p*p[i].o) + (p[j].p*p[j].o);
		if ((par->T_INTERFACE_CORRECTION == 1) && (p[i].c != p[j].c)) pres += par->INTERFACE_CORRECTION * (p[i].o + p[j].o);

		real mi_i = p[i].mi+p[i].nut*p[i].di;
		real mi_j = p[j].mi+p[j].nut*p[j].di;
		real visc = 2.0*(mi_i*mi_j)*(p[i].o + p[j].o) * (dpos.x*gkx + dpos.y*gky + dpos.z*gkz) / ( (pow2(r)+0.0001*pow2(par->H)) * (mi_i + mi_j));

		return MAKE_REAL3((visc * dvel.x - pres * gkx), (visc * dvel.y - pres * gky), (visc * dvel.z - pres * gkz));
	}
	else {
		return MAKE_REAL3(0.0, 0.0, 0.0);
	}
}



__global__ void calcInteractionWHA(Particle *p,
								   uint *gridParticleIndex,
								   uint *cellStart,
								   uint *cellEnd,
								   Parameters *par)
{
	uint index = threadIdx.x + blockIdx.x*blockDim.x;
	if (index < par->N) {
		//real3 result = interactionsNegativeOnWallNoSlip<real3>(index, p, cellStart, cellEnd, par, interaction);

		register real3 result = MAKE_REAL3(0.0,0.0,0.0);
		#include "../../methods/interactions/interactionsNegativeOnWallNoSlip.cuh"

		p[index].rh_vel.x = result.x / p[index].m;
		p[index].rh_vel.y = result.y / p[index].m;
		p[index].rh_vel.z = result.z / p[index].m;
	}
}


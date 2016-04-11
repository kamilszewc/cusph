/*
 * calcDensityWS.cu
 *
 *  Created on: 17-12-2013
 *      Author: Kamil Szewc (kamil.szewc@gmail.com)
 */

#include "../../sph.h"
#include "../../hlp.h"
#include "../../methods/kernels.cuh"
#include "../../methods/interactions.cuh"

/*__device__ static real interaction(uint i, uint j, real2 dpos, real2 dvel, Particle *p, Parameters *par)
{
	real q = sqrt(pow2(dpos.x) + pow2(dpos.y)) * par->I_H;
	if (q < 2.0) return kern(q, par->KNORM);
	else return 0.0;
}


__global__ void calcDensitySTM(Particle *p,
	Particle *p,
	uint *gridParticleIndex,
	uint *cellStart,
	uint *cellEnd,
	Parameters *par)
{
	uint index = threadIdx.x + blockIdx.x*blockDim.x;
	if (index < par->N) {
		#include "../../methods/interactions_1.cuh"

		uint originalIndex = gridParticleIndex[index];
		p[originalIndex].d = result * p[originalIndex].m;
		p[index].d = result * p[index].m;
		p[originalIndex].o = result;
		p[index].o = result;
	}
}*/

__global__ void calcDensitySTM(Particle *p, Parameters *par)
{
	uint tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < par->N) {
		p[tid].d += par->DT * p[tid].rh_d;

		tid += blockDim.x * gridDim.x;
	}

}



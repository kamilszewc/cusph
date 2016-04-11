/*
* setParticlesWS.cu
*
*  Created on: 30-10-2015
*      Author: Kamil Szewc
*/
#include "../../sph.h"
#include "../../hlp.h"

__global__ void setParticlesWSDP(Particle *p, Parameters *par)
{
	unsigned int tid = threadIdx.x + blockIdx.x*blockDim.x;
	while (tid < par->N_DISPERSED_PHASE_FLUID) {
		p[tid].di = 4.0;
		p[tid].d = p[tid].o * p[tid].di;
		p[tid].m = par->XCV * par->YCV * p[tid].d / (par->NX * par->NY);

		tid += blockDim.x * gridDim.x;
	}
}

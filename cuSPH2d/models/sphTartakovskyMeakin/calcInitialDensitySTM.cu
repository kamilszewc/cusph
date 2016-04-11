/*
* calcInitialDensityWHA.cu
*
*  Created on: 02-09-2013
*      Author: Kamil Szewc (kamil.szewc@gmail.com)
*/
#include "../../sph.h"

__global__ void calcInitialDensitySTM(Particle *p, Parameters *par)
{
	uint tid = threadIdx.x + blockIdx.x*blockDim.x;
	while (tid < par->N) {
		p[tid].di = p[tid].d;
		tid += blockDim.x * gridDim.x;
	}
}

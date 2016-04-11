/*
 * calcDensityWS.cu
 *
 *  Created on: 17-12-2013
 *      Author: Kamil Szewc (kamil.szewc@gmail.com)
 */

#include "../../sph.h"

__global__ void calcDensityWS(Particle *p, Parameters *par)
{
	uint tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < par->N) {
		p[tid].d += par->DT * p[tid].rh_d;

		tid += blockDim.x * gridDim.x;
	}

}


/*
* calcPressureWHA.cu
*
*  Created on: 11-04-2013
*      Author: Kamil Szewc
*/
#include <math.h>
#include "../../sph.h"
#include "../../hlp.h"

__global__ void calcPressureWHA(Particle *p, Parameters *par)
{
	unsigned int tid = threadIdx.x + blockIdx.x*blockDim.x;
	while (tid < par->N) {
		p[tid].p = p[tid].b * (pow(p[tid].d / p[tid].di, p[tid].gamma) - 1.0);
		tid += blockDim.x * gridDim.x;
	}
}



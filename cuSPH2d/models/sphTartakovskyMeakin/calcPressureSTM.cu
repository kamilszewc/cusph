/*
* calcPressureSTM.cu
*
*  Created on: 11-04-2013
*      Author: Kamil Szewc
*/
#include "../../sph.h"
#include "../../hlp.h"

__global__ void calcPressureSTM(Particle *p, Parameters *par)
{
	unsigned int tid = threadIdx.x + blockIdx.x*blockDim.x;
	while (tid < par->N) {
		real kT = 0.2f;
		real c1 = 0.5;
		//real c2 = 2.0;
		p[tid].p = p[tid].d * kT / (1.0 - c1 * p[tid].d);// - c2 * pow2(p[tid].d);
		tid += blockDim.x * gridDim.x;
	}
}



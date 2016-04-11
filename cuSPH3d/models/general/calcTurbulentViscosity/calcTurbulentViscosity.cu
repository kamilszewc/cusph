/*
* calcTurbulentViscosity.cu
*
*  Created on: 23-07-2015
*      Author: Kamil Szewc
*         
*/
#include "../../../sph.h"
#include "../../../hlp.h"

__global__ void calcTurbulentViscosity(Particle *p, Parameters *par)
{
	unsigned int tid = threadIdx.x + blockIdx.x*blockDim.x;
	while (tid < par->N) {
		if (p[tid].phaseType == 0)
		{
			p[tid].nut = pow2(0.12f * par->DR) * p[tid].str;
		}
		tid += blockDim.x * gridDim.x;
	}
}
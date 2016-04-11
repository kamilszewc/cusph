/*
* calcTurbulentViscositySO.cu
*
*  Created on: 10-04-2015
*      Author: Kamil Szewc
*         
*/
#include "../../sph.h"
#include "../../hlp.h"

__global__ void calcTurbulentViscositySO(Particle *p, Parameters *par)
{
	unsigned int tid = threadIdx.x + blockIdx.x*blockDim.x;
	while (tid < par->N) {
		if (p[tid].phaseType == 0)
		{
			p[tid].nut = pow2(0.12 * par->DR) * sqrt(2.0) * sqrt( pow2(p[tid].str.x) + pow2(p[tid].str.y) + pow2(p[tid].str.z) + pow2(p[tid].str.w) );
		}
		tid += blockDim.x * gridDim.x;
	}
}

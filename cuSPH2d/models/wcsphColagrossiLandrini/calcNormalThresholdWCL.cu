/*
* calcNormalThresholdWHA.cu
*
*  Created on: 02-09-2013
*      Author: Kamil Szewc (kamil.szewc@gmail.com)
*/
#include "../../sph.h"

__global__ void calcNormalThresholdWCL(Particle *p, Parameters *par)
{
	uint tid = threadIdx.x + blockIdx.x*blockDim.x;
	while (tid < par->N) 
	{
		if (p[tid].n.z >(0.01f*par->I_H)) p[tid].na = 1;
		else p[tid].na = 0;
		tid += blockDim.x * gridDim.x;
	}
}

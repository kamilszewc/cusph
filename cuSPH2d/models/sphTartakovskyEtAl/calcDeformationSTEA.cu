/*
* calcDeformationSTEA.cu
*
*  Created on: 17-08-2014
*      Author: Kamil Szewc
*/
#include "../../sph.h"
#include "../../hlp.h"
#include "math.h"

__global__ void calcDeformationSTEA(Particle *p, Parameters *par)
{
	unsigned int tid = threadIdx.x + blockIdx.x*blockDim.x;
	while (tid < par->N) {
		real R = 11.0;
		real U0 = 3.0f;
		real R0 = 0.25f * R;

		real x = p[tid].pos.x - 25.0f;
		real y = p[tid].pos.y - 25.0f;
		real r  = hypot(x, y);

		p[tid].vel.x = U0 * x * ( 1.0 - pow2(y)/(R0*r) ) * exp(-r/R0) / R0;
		p[tid].vel.y = -U0 * y * ( 1.0 - pow2(x)/(R0*r) ) * exp(-r/R0) / R0;

		p[tid].nu = 0.5;
		p[tid].mi = 0.5;

		tid += blockDim.x * gridDim.x;
	}
}

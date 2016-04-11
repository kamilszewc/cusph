/*
* calcDeformationSTM.cu
*
*  Created on: 17-08-2014
*      Author: Kamil Szewc
*/
#include "../../sph.h"
#include "../../hlp.h"
#include "math.h"

__global__ void calcDeformationSTM(Particle *p, Parameters *par)
{
	unsigned int tid = threadIdx.x + blockIdx.x*blockDim.x;
	while (tid < par->N) {
		real e = 0.56;
		real omega = M_PI * e;
		real r = hypot(p[tid].pos.x, p[tid].pos.y);
		real u = atan(p[tid].pos.x / p[tid].pos.y);

		p[tid].pos.x = sqrt(2.0/sin(omega)) * r * sin(0.5*omega) * sin(u);
		p[tid].pos.y = sqrt(2.0/sin(omega)) * r * cos(0.5*omega) * cos(u);
		//p[tid].vel.x = 0.0;
		//p[tid].vel.y = 0.0;
		p[tid].nu = 0.25f;
		p[tid].mi = 0.25f;

		tid += blockDim.x * gridDim.x;
	}
}

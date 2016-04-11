/*
* calcSingleSolidParticleAcceleration.cu
*
*  Created on: 3-08-2015
*      Author: Kamil Szewc (kamil.szewc@gmail.com)
*/
#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/reduce.h>
#include "../../../sph.h"

struct conditional_operator
{
	__host__ __device__ real operator()(const Particle p) const 
	{
		if (p.phaseType == 2)
		{
			return p.rh_vel.z;
		}
		else
		{
			return 0.0;
		}
	}
};

static __global__ void setCalculatedAcceleration(Particle *p, Parameters *par, real acceleration)
{
	unsigned int tid = threadIdx.x + blockIdx.x*blockDim.x;
	while (tid < par->N) {
		if (p[tid].phaseType == 2)
		{
			p[tid].rh_vel.z = acceleration;
			p[tid].rh_vel.x = 0.0;
			p[tid].rh_vel.y = 0.0;
		}
		tid += blockDim.x * gridDim.x;
	}
}


void calcSingleSolidParticleAcceleration(int NOB, int TPB, thrust::device_vector<Particle>& p, Parameters *par)
{
	real sum = thrust::transform_reduce(p.begin(), p.end(), conditional_operator(), 0.0, thrust::plus<real>());

	Particle* pRaw = thrust::raw_pointer_cast(p.data());

	setCalculatedAcceleration<<<NOB,TPB>>>(pRaw, par, sum);
}

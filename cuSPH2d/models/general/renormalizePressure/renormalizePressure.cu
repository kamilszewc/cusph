/*
* renormalizePressure.cu
*
*  Created on: 13-09-2013
*      Author: Kamil Szewc (kamil.szewc@gmail.com)
*/

#include "../../../sph.h"
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/fill.h>
#include <stdio.h>

static __global__ void copyPressure(real *p, Particle *particle, Parameters *par)
{
	uint tid = threadIdx.x + blockIdx.x*blockDim.x;
	while (tid < par->N) 
	{
		p[tid] = particle[tid].p;
		tid += blockDim.x * gridDim.x;
	}
}

static void sortPressure(real *p, uint numParticles)
{
	thrust::sort(thrust::device_ptr<real>(p),
		thrust::device_ptr<real>(p + numParticles));
}

static __global__ void gaugePressure(real *p, Particle *particle, Parameters *par) {
	uint tid = threadIdx.x + blockIdx.x*blockDim.x;
	while (tid < par->N) 
	{
		particle[tid].p = particle[tid].p - p[0];//- 0.5*(p[par->N-1] + p[0]);
		//printf("%f\n", p[par->N-1]);
		tid += blockDim.x * gridDim.x;
	}
}


void renormalizePressure(int NOB, int TPB, Particle *particle, Parameters *par, uint numParticles)
{
	thrust::device_vector<real> help = thrust::device_vector<real>(numParticles);
	thrust::fill(help.begin(), help.end(), 0.0);
	real* helpArray = thrust::raw_pointer_cast(help.data());
	copyPressure<<<NOB,TPB>>>(helpArray, particle, par);

	sortPressure(helpArray, numParticles);

	gaugePressure<<<NOB,TPB>>>(helpArray, particle, par);

}


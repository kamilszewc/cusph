/*
* renormalizePressure.cu
*
*  Created on: 13-09-2013
*      Author: Kamil Szewc (kamil.szewc@gmail.com)
*/

#include "../../../sph.h"
#include <thrust/device_ptr.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>

__global__ void copyPressure(real *p, Particle *particle, Parameters *par)
{
	uint tid = threadIdx.x + blockIdx.x*blockDim.x;
	while (tid < par->N) {
		p[tid] = particle[tid].p;
		tid += blockDim.x * gridDim.x;
	}
}

void sortPressure(real *p, uint numParticles) {
	thrust::sort(thrust::device_ptr<real>(p),
		thrust::device_ptr<real>(p + numParticles));
}

__global__ void renormalizePressure(real *p, Particle *particle, uint numParticles) {
	uint tid = threadIdx.x + blockIdx.x*blockDim.x;
	while (tid < numParticles) {
		particle[tid].p = particle[tid].p - p[0];
		tid += blockDim.x * gridDim.x;
	}
}

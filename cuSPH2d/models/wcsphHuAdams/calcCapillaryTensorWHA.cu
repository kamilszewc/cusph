/*
 * calcCapillaryTensorWHA.cu
 *
 *  Created on: 16-07-2015
 *      Author: Kamil Szewc (kamil.szewc@gmail.com)
 *             
 */

#include "../../sph.h"
#include "../../hlp.h"
#include "../../methods/kernels.cuh"
#include "../../methods/interactions.cuh"

__global__ void calcCapillaryTensorWHA(Particle *p,
	uint *gridParticleIndex,
	uint *cellStart,
	uint *cellEnd,
	Parameters *par)
{
	uint index = threadIdx.x + blockIdx.x*blockDim.x;
	if (index < par->N) {
		if (p[index].n.z > 0.0)
		{
			p[index].ct.x = -(par->SURFACE_TENSION/p[index].n.z) * ( pow2(p[index].n.z) - p[index].n.x * p[index].n.x ) ;
			p[index].ct.y = -(par->SURFACE_TENSION/p[index].n.z) * ( - p[index].n.x * p[index].n.y);
			p[index].ct.z = -(par->SURFACE_TENSION/p[index].n.z) * ( - p[index].n.y * p[index].n.x);
			p[index].ct.w = -(par->SURFACE_TENSION/p[index].n.z) * ( pow2(p[index].n.z) - p[index].n.y * p[index].n.y);
		}
	}
}



/*
* calcSurfactantsDiffusionBulkWHA.cu
*
*  Created on: 13-10-2014
*      Author: Kamil Szewc (kamil.szewc@gmail.com)
*/
#include <math.h>
#include "../../sph.h"
#include "../../hlp.h"
#include "../../methods/kernels.cuh"
#include "../../methods/interactions.cuh"


__global__ void calcSurfactantsMassWHA(Particle *p,
	uint *gridParticleIndex,
	uint *cellStart,
	uint *cellEnd,
	Parameters *par)
{
	uint index = threadIdx.x + blockIdx.x*blockDim.x;
	if (index < par->N) {

		if (p[index].a > 0.00000001f)
		{
			p[index].mSurf = p[index].cSurf * p[index].a;
		}
		else
		{
			p[index].mSurf = 0.0;
			p[index].cSurf = 0.0;
			p[index].dSurf = 0.0;
		}
	}
}

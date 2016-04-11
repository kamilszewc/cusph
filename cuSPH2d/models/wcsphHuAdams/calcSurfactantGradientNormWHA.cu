/*
* calcSurfactantsGradientNormWHA.cu
*
*  Created on: 17-10-2014
*      Author: Kamil Szewc (kamil.szewc@gmail.com)
*/
#include <math.h>
#include "../../sph.h"
#include "../../hlp.h"
#include "../../methods/kernels.cuh"
#include "../../methods/interactions.cuh"
#include <stdio.h>

__device__ static real4 interaction(uint i, uint j, real2 dpos, real2 dvel, Particle *p, Parameters *par)
{
	real r = hypot(dpos.x, dpos.y);
	real q = r * par->I_H;
	if ((p[j].a > 0.000001f) && (q < 2.0) && ((i != j) || ((i == j) && (q > 0.001f*par->I_H)))) {
		real gkx = grad_of_kern(dpos.x, q, par->I_H);
		real gky = grad_of_kern(dpos.y, q, par->I_H);

		real val =  p[j].o * p[j].m / p[j].d;

		return MAKE_REAL4(val*gkx*dpos.x, val*gkx*dpos.y, val*gky*dpos.x, val*gky*dpos.y);
	}
	else
	{
		return MAKE_REAL4(0.0, 0.0, 0.0, 0.0);
	}
}


__global__ void calcSurfactantsGradientNormWHA(Particle *p,
	uint *gridParticleIndex,
	uint *cellStart,
	uint *cellEnd,
	Parameters *par)
{
	uint index = threadIdx.x + blockIdx.x*blockDim.x;
	if (index < par->N) {
		register real4 result = MAKE_REAL4(0.0, 0.0, 0.0, 0.0);
		#include "../../methods/interactions/interactionsNegativeOnWallNoSlip.cuh"

		real matrixDeterminant = result.w * result.x - result.y * result.z;
		real4 matrix;
		if (matrixDeterminant > 0.0)
		{
			matrix.x =  result.w / matrixDeterminant;
			matrix.y = -result.y / matrixDeterminant;
			matrix.z = -result.z / matrixDeterminant;
			matrix.w =  result.x / matrixDeterminant;
		}
		else
		{
			matrix = MAKE_REAL4(0.0, 0.0, 0.0, 0.0);
		}

		real surfGradX = (matrix.x * p[index].cSurfGrad.x + matrix.y * p[index].cSurfGrad.y) * p[index].n.z * p[index].dSurf;
		real surfGradY = (matrix.z * p[index].cSurfGrad.x + matrix.w * p[index].cSurfGrad.y) * p[index].n.z * p[index].dSurf;

		p[index].cSurfGrad.x = surfGradX;
		p[index].cSurfGrad.y = surfGradY;
	}
}



/*
* calcSurfaceTensionFromCurvatureWHA.cu
*
*  Created on: 02-09-2013
*      Author: Kamil Szewc (kamil.szewc@gmail.com)
*/
#include <math.h>
#include "../../sph.h"
#include "../../hlp.h"
#include "../../methods/kernels.cuh"
#include "../../methods/interactions.cuh"

__device__ static real3 interaction(uint i, uint j, real2 dpos, real2 dvel, Particle *p, Parameters *par)
{
	real r = hypot(dpos.x, dpos.y);
	real q = r * par->I_H;

	if (par->T_NORMAL_VECTOR < 2)
	{
		if ((q < 2.0) && ((p[i].na == 1) && (p[j].na == 1))) {
			real diffx = ((p[i].n.x / p[i].n.z) - (p[j].n.x / p[j].n.z)) * grad_of_kern(dpos.x, q, par->I_H);
			real diffy = ((p[i].n.y / p[i].n.z) - (p[j].n.y / p[j].n.z)) * grad_of_kern(dpos.y, q, par->I_H);

			return MAKE_REAL3(p[j].m*diffx / p[j].d, p[j].m*diffy / p[j].d, p[j].m * kern(q, par->I_H) / p[j].d);
		}
		else
		{
			return MAKE_REAL3(0.0, 0.0, 0.0);
		}
	}
	else
	{
		if ((q < 2.0) && ((p[i].na == 1) && (p[j].na == 1))) {
			real phi = 0.0;
			if (p[i].c != p[j].c) phi = -1.0;
			else phi = 1.0;
			real gkx = grad_of_kern(dpos.x, q, par->I_H);
			real gky = grad_of_kern(dpos.y, q, par->I_H);
			real gknorm = sqrt( pow2(gkx) + pow2(gky) );
			real diffx = 2.0 * p[j].m * ((p[i].n.x / p[i].n.z) - (phi * p[j].n.x / p[j].n.z)) * gkx / p[j].d;
			real diffy = 2.0 * p[j].m * ((p[i].n.y / p[i].n.z) - (phi * p[j].n.y / p[j].n.z)) * gky / p[j].d;
			real div = p[j].m * r * gknorm / p[j].d;

			return MAKE_REAL3(diffx, diffy, div);
		}
		else
		{
			return MAKE_REAL3(0.0, 0.0, 0.0);
		}
	}
}

__global__ void calcSurfaceTensionFromCurvatureWHA(Particle *p,
	uint *gridParticleIndex,
	uint *cellStart,
	uint *cellEnd,
	Parameters *par)
{
	uint index = threadIdx.x + blockIdx.x*blockDim.x;
	if (index < par->N) {
		register real3 result = MAKE_REAL3(0.0,0.0,0.0);
		#include "../../methods/interactions/interactionsPositiveOnWallNoSlip.cuh"

		if (result.z > 0.0) {
			p[index].cu = (result.x + result.y) / result.z;
			p[index].cw = result.z;
		}

		real help = 0.0;
		if (par->T_SURFACTANTS !=0)
		{
			help = (par->SURFACE_TENSION + ((par->SURFACE_TENSION*0.8f) - par->SURFACE_TENSION) * p[index].cSurf / 3.0e-6) * p[index].cu / p[index].d;
		}
		else
		{
			help = par->SURFACE_TENSION * p[index].cu / p[index].d;
		}

		p[index].st.x = help * p[index].n.x;
		p[index].st.y = help * p[index].n.y;

	}
}






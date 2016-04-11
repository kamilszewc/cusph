/*
* calcSoilViscositySO.cu
*
*  Created on: 15-04-2015
*      Author: Kamil Szewc
*         
*/
#include "../../sph.h"
#include "../../hlp.h"

__global__ void calcSoilViscosityWSDP(Particle *p, Parameters *par)
{
	unsigned int tid = threadIdx.x + blockIdx.x*blockDim.x;
	while (tid < par->N) {
		if ((p[tid].phaseType == 1) || (p[tid].phaseType == -1))
		{
			real strainRate = sqrt(4.0) * sqrt( pow2(p[tid].str.x) + pow2(p[tid].str.y + p[tid].str.z) + pow2(p[tid].str.w) );

			real nut = (par->SOIL_COHESION + p[tid].p * tan(par->SOIL_INTERNAL_ANGLE))  /  strainRate;

			p[tid].nut = par->SOIL_MAXIMAL_VISCOSITY / p[tid].d;

			if ( nut < par->SOIL_MAXIMAL_VISCOSITY )
			{
				p[tid].nut = nut / p[tid].d;
			}

			if (nut < par->SOIL_MINIMAL_VISCOSITY)
			{
				p[tid].nut = par->SOIL_MINIMAL_VISCOSITY / p[tid].d;
			}

		} 
		tid += blockDim.x * gridDim.x;
	}
}

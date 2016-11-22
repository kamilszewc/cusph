/*
* calcAdvectionParticlesWSDP.cu
*
*  Created on: 17-12-2013
*      Author: Kamil Szewc (kamil.szewc@gmail.com)
*/
#include "../../sph.h"

__global__ void calcAdvectionParticlesWSDP(Particle *pPDPF, Parameters *par)
{
	uint tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < par->N_DISPERSED_PHASE_FLUID) {
		pPDPF[tid].vel.x += par->DT * (pPDPF[tid].rh_vel.x + par->G_X);
		pPDPF[tid].vel.y += par->DT * (pPDPF[tid].rh_vel.y + par->G_Y);
		pPDPF[tid].pos.x += par->DT * pPDPF[tid].vel.x;
		pPDPF[tid].pos.y += par->DT * pPDPF[tid].vel.y;
		pPDPF[tid].d += par->DT * pPDPF[tid].rh_d;

		if (pPDPF[tid].pos.y <= 0.0001)
		{
			pPDPF[tid].vel.y = -pPDPF[tid].vel.y;
			pPDPF[tid].pos.y = 0.0001;
		}

		tid += blockDim.x * gridDim.x;
	}
}

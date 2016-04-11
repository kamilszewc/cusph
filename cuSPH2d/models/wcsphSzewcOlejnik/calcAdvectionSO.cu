/*
* calcAdvectionSO.cu
*
*  Created on: 16-02-2015
*      Authors: Kamil Szewc (kamil.szewc@gmail.com)
*               Michal Olejnik
*/
#include "../../sph.h"

__global__ void calcAdvectionSO(Particle *p, Parameters *par)
{
	uint tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < par->N) {
		p[tid].d += par->DT * p[tid].rh_d;

		if (p[tid].phaseType >= 0)
		{
			p[tid].vel.x += par->DT * ((1.0 - par->XSPH)*p[tid].rh_vel.x + par->XSPH*p[tid].rh_pos.x + p[tid].st.x + par->G_X);
			p[tid].vel.y += par->DT * ((1.0 - par->XSPH)*p[tid].rh_vel.y + par->XSPH*p[tid].rh_pos.y + p[tid].st.y + par->G_Y);

			p[tid].pos.x += par->DT * p[tid].vel.x;
			p[tid].pos.y += par->DT * p[tid].vel.y;
		}

		if (par->T_BOUNDARY_PERIODICITY == 0) {
			if (p[tid].pos.x > par->XCV) {
				p[tid].vel.x = -p[tid].vel.x;
				p[tid].pos.x = 2.0 * par->XCV - p[tid].pos.x;
			};
			if (p[tid].pos.x <= 0.0) {
				p[tid].vel.x = -p[tid].vel.x;
				p[tid].pos.x = -p[tid].pos.x;
			};
		}
		else {
			if (p[tid].pos.x >= par->XCV) {
				p[tid].pos.x = p[tid].pos.x - par->XCV;
			};
			if (p[tid].pos.x < 0.0) {
				p[tid].pos.x = p[tid].pos.x + par->XCV;
			};
		}

		if (par->T_BOUNDARY_PERIODICITY != 1){
			if (p[tid].pos.y > par->YCV) {
				p[tid].vel.y = -p[tid].vel.y;
				p[tid].pos.y = 2.0 * par->YCV - p[tid].pos.y;
			};
			if (p[tid].pos.y <= 0.0) {
				p[tid].vel.y = -p[tid].vel.y;
				p[tid].pos.y = -p[tid].pos.y;
			};
		}
		else {
			if (p[tid].pos.y >= par->YCV) {
				p[tid].pos.y = p[tid].pos.y - par->YCV;
			};
			if (p[tid].pos.y < 0.0) {
				p[tid].pos.y = p[tid].pos.y + par->YCV;
			};
		}

		tid += blockDim.x * gridDim.x;
	}

}

#include "../../../sph.h"
#include "../../../hlp.h"
#include <math.h>

__global__ void calcDispersedPhaseAdvection(ParticleDispersedPhase *p, Parameters *par)
{
	uint tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < par->N_DISPERSED_PHASE) {
		real u = sqrt(pow2(p[tid].vel.x - p[tid].velFl.x) + pow2(p[tid].vel.y - p[tid].velFl.y));
		real re = p[tid].dFl * p[tid].dia * u / p[tid].miFl;
		real cd = 0.0;
		if (re > 0.0)
		{
			cd = (24.0 / re) * (1.0 + 0.15 * pow((double)re, (double)0.687));
		}
		real fd = 0.5 * p[tid].dFl * pow2(p[tid].dia) * M_PI * 0.25 * cd * u;
		real m = p[tid].d * M_PI * pow3(p[tid].dia) * (1.0 / 6.0);
		p[tid].vel.x += par->DT * ( -fd*(p[tid].vel.x - p[tid].velFl.x) / m + par->G_X * (p[tid].d - p[tid].dFl) / p[tid].d );
		p[tid].vel.y += par->DT * ( -fd*(p[tid].vel.y - p[tid].velFl.y) / m + par->G_Y * (p[tid].d - p[tid].dFl) / p[tid].d );
		p[tid].pos.x += par->DT * p[tid].vel.x;
		p[tid].pos.y += par->DT * p[tid].vel.y;

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

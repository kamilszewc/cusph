#include "../../../sph.h"
#include "../../../hlp.h"
#include <stdio.h>

__global__ void calcDispersedPhaseAdvection(ParticleDispersedPhase *p, Parameters *par)
{
	uint tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < par->N_DISPERSED_PHASE) {
		real u = sqrt(pow2(p[tid].vel.x - p[tid].velFl.x) + pow2(p[tid].vel.y - p[tid].velFl.y) + pow2(p[tid].vel.z - p[tid].velFl.z));
		real re = p[tid].dFl * p[tid].dia * u / p[tid].miFl;
		real cd = 0.0;
		if (re > 0.0)
		{
			cd = (24.0 / re) * (1.0 + 0.15f * pow((double)re, (double)0.687));
		}
		real fd = 0.5 * p[tid].dFl * pow2(p[tid].dia) * M_PI * 0.25f * cd * u;
		real m = p[tid].d * M_PI * pow3(p[tid].dia) * (1.0 / 6.0);
		p[tid].vel.x += par->DT * (-fd*(p[tid].vel.x - p[tid].velFl.x) / m + par->G_X * (p[tid].d - p[tid].dFl) / p[tid].d);
		p[tid].vel.y += par->DT * (-fd*(p[tid].vel.y - p[tid].velFl.y) / m + par->G_Y * (p[tid].d - p[tid].dFl) / p[tid].d);
		p[tid].vel.z += par->DT * (-fd*(p[tid].vel.z - p[tid].velFl.z) / m + par->G_Z * (p[tid].d - p[tid].dFl) / p[tid].d);
		p[tid].pos.x += par->DT * p[tid].vel.x;
		p[tid].pos.y += par->DT * p[tid].vel.y;
		p[tid].pos.z += par->DT * p[tid].vel.z;

		if (par->T_BOUNDARY_PERIODICITY == 0) //X,Y
		{
			if (p[tid].pos.x > par->XCV)
			{
				p[tid].vel.x = -p[tid].vel.x;
				p[tid].pos.x = 2.0 * par->XCV - p[tid].pos.x;
			}
			if (p[tid].pos.x <= 0.0)
			{
				p[tid].vel.x = -p[tid].vel.x;
				p[tid].pos.x = -p[tid].pos.x;
			}

			if (p[tid].pos.y > par->YCV)
			{
				p[tid].vel.y = -p[tid].vel.y;
				p[tid].pos.y = 2.0 * par->YCV - p[tid].pos.y;
			}
			if (p[tid].pos.y <= 0.0)
			{
				p[tid].vel.y = -p[tid].vel.y;
				p[tid].pos.y = -p[tid].pos.y;
			}
		}
		else
		{
			if (p[tid].pos.x > par->XCV) p[tid].pos.x -= par->XCV;
			if (p[tid].pos.x <= 0.0)    p[tid].pos.x += par->XCV;
			if (p[tid].pos.y > par->YCV) p[tid].pos.y -= par->YCV;
			if (p[tid].pos.y <= 0.0)    p[tid].pos.y += par->YCV;
		}

		if ((par->T_BOUNDARY_PERIODICITY == 0) || (par->T_BOUNDARY_PERIODICITY == 2))  //Z
		{

			if (p[tid].pos.z > par->ZCV)
			{
				p[tid].vel.z = -p[tid].vel.z;
				p[tid].pos.z = 2.0 * par->ZCV - p[tid].pos.z;
			}
			if (p[tid].pos.z <= 0.0)
			{
				p[tid].vel.z = -p[tid].vel.z;
				p[tid].pos.z = -p[tid].pos.z;
			}
		}
		else
		{
			if (p[tid].pos.z > par->ZCV) p[tid].pos.z -= par->ZCV;
			if (p[tid].pos.z <= 0.0)    p[tid].pos.z += par->ZCV;
		}

		tid += blockDim.x * gridDim.x;
	}
}
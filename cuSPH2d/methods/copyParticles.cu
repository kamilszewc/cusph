/*
* @file copyParticles.cu
* @since 26-07-2015
* @author Kamil Szewc (kamil.szewc@gmail.com)
*/
#include <math.h>
#include "../sph.h"
#include "../hlp.h"
#include "interactions.cuh"

__global__ void copyParticles(Particle *pSort, Particle *p, uint *gridParticleIndex, bool sorted, Parameters *par, uint N)
{
	uint index = threadIdx.x + blockIdx.x*blockDim.x;
	if (index < N) {
		uint sortedIndex;
		if (sorted == false) sortedIndex = index;
		else sortedIndex = gridParticleIndex[index];

		int id = p[sortedIndex].id;
		int phaseId = p[sortedIndex].phaseId;
		int phaseType = p[sortedIndex].phaseType;
		real2 pos = p[sortedIndex].pos;
		real2 rh_pos = p[sortedIndex].rh_pos;
		real2 vel = p[sortedIndex].vel;
		real2 rh_vel = p[sortedIndex].rh_vel;
		real h = p[sortedIndex].h;
		real m = p[sortedIndex].m;
		real pp = p[sortedIndex].p;
		real ph = p[sortedIndex].ph;
		real phs = p[sortedIndex].phs;
		real d = p[sortedIndex].d;
		real rh_d = p[sortedIndex].rh_d;
		real di = p[sortedIndex].di;
		real nu = p[sortedIndex].nu;
		real mi = p[sortedIndex].mi;
		real4 str = p[sortedIndex].str;
		real nut = p[sortedIndex].nut;
		real4 tau = p[sortedIndex].tau;
		real gamma = p[sortedIndex].gamma;
		real s = p[sortedIndex].s;
		real b = p[sortedIndex].b;
		real o = p[sortedIndex].o;
		real c = p[sortedIndex].c;
		real3 n = p[sortedIndex].n;
		int na = p[sortedIndex].na;
		real cu = p[sortedIndex].cu;
		real2 st = p[sortedIndex].st;
		real cs = p[sortedIndex].cs;
		real cw = p[sortedIndex].cw;
		real4 ct = p[sortedIndex].ct;
		real cBulk = p[sortedIndex].cBulk;
		real dBulk = p[sortedIndex].dBulk;
		real mBulk = p[sortedIndex].mBulk;
		real cSurf = p[sortedIndex].cSurf;
		real dSurf = p[sortedIndex].dSurf;
		real mSurf = p[sortedIndex].mSurf;
		real2 cSurfGrad = p[sortedIndex].cSurfGrad;
		real a = p[sortedIndex].a;

		pSort[index].id = id;
		pSort[index].phaseId = phaseId;
		pSort[index].phaseType = phaseType;
		pSort[index].pos = pos;
		pSort[index].rh_pos = rh_pos;
		pSort[index].vel = vel;
		pSort[index].rh_vel = rh_vel;
		pSort[index].h = h;
		pSort[index].m = m;
		pSort[index].p = pp;
		pSort[index].ph = ph;
		pSort[index].phs = phs;
		pSort[index].d = d;
		pSort[index].rh_d = rh_d;
		pSort[index].di = di;
		pSort[index].nu = nu;
		pSort[index].mi = mi;
		pSort[index].str = str;
		pSort[index].nut = nut;
		pSort[index].tau = tau;
		pSort[index].gamma = gamma;
		pSort[index].s = s;
		pSort[index].b = b;
		pSort[index].o = o;
		pSort[index].c = c;
		pSort[index].n = n;
		pSort[index].na = na;
		pSort[index].cu = cu;
		pSort[index].st = st;
		pSort[index].cs = cs;
		pSort[index].cw = cw;
		pSort[index].ct = ct;
		pSort[index].cBulk = cBulk;
		pSort[index].dBulk = dBulk;
		pSort[index].mBulk = mBulk;
		pSort[index].cSurf = cSurf;
		pSort[index].dSurf = dSurf;
		pSort[index].mSurf = mSurf;
		pSort[index].cSurfGrad = cSurfGrad;
		pSort[index].a = a;
	}
}


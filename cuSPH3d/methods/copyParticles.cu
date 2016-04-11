/*
* copyParticles.cu
*
*  Created on: 26-07-2015
*  Author: Kamil Szewc (kamil.szewc@gmail.com)
*/
#include <math.h>
#include "../sph.h"
#include "../hlp.h"
#include "interactions.cuh"

__global__ void copyParticles(Particle *pSort, Particle *p, uint *gridParticleIndex, bool sorted, Parameters *par)
{
	uint index = threadIdx.x + blockIdx.x*blockDim.x;
	if (index < par->N) {
		uint sortedIndex;
		if (sorted == false) sortedIndex = index;
		else sortedIndex = gridParticleIndex[index];

		int id = p[sortedIndex].id;
		int phaseId = p[sortedIndex].phaseId;
		int phaseType = p[sortedIndex].phaseType;
		real3 pos = p[sortedIndex].pos;
		real3 rh_pos = p[sortedIndex].rh_pos;
		real3 vel = p[sortedIndex].vel;
		real3 rh_vel = p[sortedIndex].rh_vel;
		real m = p[sortedIndex].m;
		real pp = p[sortedIndex].p;
		real ph = p[sortedIndex].ph;
		real d = p[sortedIndex].d;
		real rh_d = p[sortedIndex].rh_d;
		real di = p[sortedIndex].di;
		real nu = p[sortedIndex].nu;
		real mi = p[sortedIndex].mi;
		real str = p[sortedIndex].str;
		real nut = p[sortedIndex].nut;
		real gamma = p[sortedIndex].gamma;
		real s = p[sortedIndex].s;
		real b = p[sortedIndex].b;
		real o = p[sortedIndex].o;
		real c = p[sortedIndex].c;
		real4 n = p[sortedIndex].n;
		int na = p[sortedIndex].na;
		real cu = p[sortedIndex].cu;
		real3 st = p[sortedIndex].st;
		real cs = p[sortedIndex].cs;
		real cw = p[sortedIndex].cw;

		pSort[index].id = id;
		pSort[index].phaseId = phaseId;
		pSort[index].phaseType = phaseType;
		pSort[index].pos = pos;
		pSort[index].rh_pos = rh_pos;
		pSort[index].vel = vel;
		pSort[index].rh_vel = rh_vel;
		pSort[index].m = m;
		pSort[index].p = pp;
		pSort[index].ph = ph;
		pSort[index].d = d;
		pSort[index].rh_d = rh_d;
		pSort[index].di = di;
		pSort[index].nu = nu;
		pSort[index].mi = mi;
		pSort[index].str = str;
		pSort[index].nut = nut;
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
	}
}

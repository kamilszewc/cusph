/*
* calcDispersedPhaseFields.cu
*
*  Created on: 29-06-2014
*      Author: Kamil Szewc (kamil.szewc@gmail.com)
*/
#include "../../../sph.h"
#include "../../../hlp.h"
#include "../../../methods/kernels.cuh"
#include "../../../methods/calcGridHash.cuh"
#include "../../../methods/calcGridPos.cuh"
#include "../../../methods/calcRelPosVelFreeSlip.cuh"

__device__ static real4 interaction(uint i, uint j, real2 dpos, real2 dvel, Particle *p, Parameters *par)
{
	real r = sqrt(pow2(dpos.x) + pow2(dpos.y));
	real q = r * par->I_H;

	if (q < 2.0) 
	{
		real k = kern(q, par->I_H);

		return MAKE_REAL4(p[j].vel.x*k*p[j].m / p[j].d, p[j].vel.y*k*p[j].m / p[j].d, k*p[j].m, p[j].mi * k*p[j].m / p[j].d);
	}
	else 
	{
		return MAKE_REAL4(0.0, 0.0, 0.0, 0.0);
	}
}



__global__ void calcDispersedPhaseField(Particle *p,
										uint *gridParticleIndex,
										uint *cellStart,
										uint *cellEnd,
										ParticleDispersedPhase *pDispersedPhase,
										Parameters *par)
{
	uint index = threadIdx.x + blockIdx.x*blockDim.x;
	if (index < par->N_DISPERSED_PHASE) {
		real2 pos = MAKE_REAL2(pDispersedPhase[index].pos.x, pDispersedPhase[index].pos.y);

		int2 gridPos = calcGridPos(pos, par);
		uint gridHash0 = calcGridHash(gridPos, par);
		real4 result = MAKE_REAL4(0.0, 0.0, 0.0, 0.0);

		for (int y = -1; y <= 1; y++) {
			for (int x = -1; x <= 1; x++) {
				int2 gridPos2;
				gridPos2.x = gridPos.x + x;
				gridPos2.y = gridPos.y + y;

				uint gridHash = calcGridHash(gridPos2, par);
				uint startIndex = cellStart[gridHash];

				if (startIndex != 0xffffffff) {
					uint endIndex = cellEnd[gridHash];

					for (uint j = startIndex; j < endIndex; j++) {
						real2 dpos, dvel;
						real2 pos1 = pDispersedPhase[index].pos;
						real2 pos2 = p[j].pos;
						real2 vel1 = pDispersedPhase[index].vel;
						real2 vel2 = p[j].vel;
						calcRelPosVelFreeSlip(pos1, pos2, vel1, vel2, 0, &dpos, &dvel, par);
						result += interaction(index, j, dpos, dvel, p, par);
					}

				}
			}
		}

		pDispersedPhase[index].velFl.x = result.x;
		pDispersedPhase[index].velFl.y = result.y;
		pDispersedPhase[index].dFl = result.z;
		pDispersedPhase[index].miFl = result.w;

	}
}

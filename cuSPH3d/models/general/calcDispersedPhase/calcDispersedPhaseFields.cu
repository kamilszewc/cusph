/*
* calcDispersedPhaseFields.cu
*
*  Created on: 29-06-2014
*      Author: Kamil Szewc (kamil.szewc@gmail.com)
*/
#include "../../../sph.h"
#include "../../../hlp.h"
#include "../../../methods/calcRelPosVelNoSlip.cuh"
#include "../../../methods/calcRelPosVelFreeSlip.cuh"
#include "../../../methods/calcGridPos.cuh"
#include "../../../methods/calcGridHash.cuh"
#include "../../../methods/kernels.cuh"


__device__ static real6 interaction(uint i, uint j, real3 dpos, Particle *p, Parameters *par)
{
	real r = sqrt(pow2(dpos.x) + pow2(dpos.y) + pow2(dpos.z));
	real q = r * par->I_H;
	if (q < 2.0) {
		real k = kern(q, par->KNORM);

		return MAKE_REAL6(p[j].vel.x*k*p[j].m / p[j].d, p[j].vel.y*k*p[j].m / p[j].d, p[j].vel.z*k*p[j].m / p[j].d, k*p[j].m, p[j].mi*k*p[j].m / p[j].d, 0.0);
	}
	else {
		return MAKE_REAL6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
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
		real3 pos = MAKE_REAL3(pDispersedPhase[index].pos.x, pDispersedPhase[index].pos.y, pDispersedPhase[index].pos.z);

		int3 gridPos = calcGridPos(pos, par);
		uint gridHash0 = calcGridHash(gridPos, par);
		real6 result = MAKE_REAL6(0.0,0.0,0.0,0.0,0.0,0.0);

		for (int z=-1; z<=1; z++) {
			for (int y=-1; y<=1; y++) {
				for (int x=-1; x<=1; x++) {
					int3 gridPos2;
					gridPos2.x = gridPos.x + x;
					gridPos2.y = gridPos.y + y;
					gridPos2.z = gridPos.z + z;

					uint gridHash = calcGridHash(gridPos2, par);
					uint startIndex = cellStart[gridHash];

					if (startIndex != 0xffffffff) {
						uint endIndex = cellEnd[gridHash];

						for (uint j=startIndex; j<endIndex; j++) {
							real3 dpos, dvel;
							real3 pos1 = pDispersedPhase[index].pos;
							real3 pos2 = p[j].pos;
							real3 vel1 = p[index].vel;
							real3 vel2 = p[j].vel;
							calcRelPosVelFreeSlip(pos1, pos2, vel1, vel2, 0, &dpos, &dvel, par);
							result += interaction(index, j, dpos, p, par);
						}
					}
				}
			}
		}

		pDispersedPhase[index].velFl.x = result.x;
		pDispersedPhase[index].velFl.y = result.y;
		pDispersedPhase[index].velFl.z = result.z;
		pDispersedPhase[index].dFl = result.u;
		pDispersedPhase[index].miFl = result.v;

	}
}

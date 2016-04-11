/*
 * calcHydrostaticPressure.cu
 *
 *  Created on: 29-05-2015
 *      Author: Kamil Szewc (kamil.szewc@gmail.com)
 */

#include "../../../sph.h"
#include "../../../hlp.h"
#include "../../../methods/kernels.cuh"
#include "../../../methods/interactions.cuh"

#include <stdio.h>


__device__ static real interaction(uint i, uint j, real3 dpos, real3 dvel, Particle *p, Parameters *par)
{
	real r = sqrt(pow2(dpos.x) + pow2(dpos.y) + pow2(dpos.z));
	real q = r * par->I_H;

	if (q < 2.0)
	{
		real k = kern(q, par->I_H);

		return k*p[i].m/p[i].d;
	}
	else 
	{
		return 0.0;
	}
}


__global__ void calcHydrostaticPressure(Particle *p,
	uint *gridParticleIndex,
	uint *cellStart,
	uint *cellEnd,
	Parameters *par)
{
	uint index = threadIdx.x + blockIdx.x*blockDim.x;
	if (index < par->N) {
		
		p[index].ph = 0.0;
		real height = 0.0;

		real3 pos = MAKE_REAL3(p[index].pos.x, p[index].pos.y, p[index].pos.z);

		int3 gridPos = calcGridPos(pos, par);

		//if (p[index].id == 20060) printf("--------------\n");
		for (int zc = par->NZC-1; zc >= gridPos.z; zc--)
		{
			int3 gridPosCheck;
			gridPosCheck.x = gridPos.x;
			gridPosCheck.y = gridPos.y;
			gridPosCheck.z = zc;
			uint gridHash = calcGridHash(gridPosCheck, par);
			uint startIndex = cellStart[gridHash];
			
			if (startIndex == 0xffffffff) continue;

			for (int i=5; i>=0; i--)
			{
				real z = (real)zc*2.0*par->H + i*2.0*par->H/5.0;
				real result = 0.0;

				//if (p[index].id == 20060) printf("y=%f ", y);

				for (int iy = -1; iy <= 1; iy++) {
					for (int ix = -1; ix <= 1; ix++) {
						for (int iz = -1; iz <= 1; iz++) {
							gridPosCheck.x = gridPos.x + ix;
							gridPosCheck.y = gridPos.y + iy;
							gridPosCheck.z = zc + iz;

							uint gridHash = calcGridHash(gridPosCheck, par);
							uint startIndex = cellStart[gridHash];

							if (startIndex != 0xffffffff)
							{
								uint endIndex = cellEnd[gridHash];

								for (uint j = startIndex; j < endIndex; j++)
								{
									real3 dpos, dvel;
									real3 pos1 = p[index].pos;
									pos1.z = z;
									real3 pos2 = p[j].pos;
									real3 vel1 = p[j].vel;
									real3 vel2 = p[j].vel;

									calcRelPosVelNoSlip(pos1, pos2, vel1, vel2, 0, &dpos, &dvel, par);
									result += interaction(index, j, dpos, dvel, p, par);
								}
							}
						}
					}
				}
					//if (p[index].id == 20060 ) printf("result=%f\n", result);
				if (result >= 0.5)
				{
					height = z;
					break;
				}
			}

			if (height != 0.0) break;
		}

		if (height > 0.0)
		{
			p[index].ph = p[index].di * fabs(par->G_Z) * (height-p[index].pos.z);
		}
	}
}
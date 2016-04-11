/*
* calcInteractionParticlesOnFluid.cu
*
*  Created on: 27-10-2015
*      Author: Kamil Szewc (kamil.szewc@gmail.com)
*/
#include "../../sph.h"
#include "../../hlp.h"
#include "../../methods/kernels.cuh"
#include "../../methods/calcGridHash.cuh"
#include "../../methods/calcGridPos.cuh"
#include "../../methods/calcRelPosVelNoSlip.cuh"
#include "wcsphStandardDispersedPhase.h"

__device__ static float calcK(real2 dvel, real dLiquid, real oLiquid, real oDust, real d, real visc)
{
	real u = sqrt(pow2(dvel.x) + pow2(dvel.y));

	real re = d * u / visc;

	real cd = 0.0;
	if (re > 0.0) cd = 24.0 * sqrt(1.0 + (3.0 * re / 16.0)) / re;

	return (3.0 / 4.0) * dLiquid * oLiquid * oDust * cd * u * pow(oLiquid, -2.65) / d;
}

__device__ static real6 interaction(uint j, Particle *p, uint i, Particle *pFluid, real2 dpos, real2 dvel, Parameters *par)
{
	real r = sqrt(pow2(dpos.x) + pow2(dpos.y));
	real q = r * par->I_H;

	if (q < 2.0)
	{
		real gkx = grad_of_kern(dpos.x, q, par->I_H);
		real gky = grad_of_kern(dpos.y, q, par->I_H);
		real k = kern_kwon_monaghan(q, par->I_H);
			
		real D = 0.0005;

		real K = calcK(dvel, pFluid[i].d, pFluid[i].o, p[j].o, D, pFluid[i].nu);

		real pres = p[j].m * p[j].o / p[j].d;

		real visc = 2.0 * p[j].m * K * (dvel.x*dpos.x + dvel.y*dpos.y) * k / ((pow2(r) + 0.01*pow2(par->H)) * p[j].d);
		
		real dens = dvel.x*gkx + dvel.y*gky;

		return MAKE_REAL6(pres * gkx, pres * gky, visc * dpos.x, visc * dpos.y, p[j].m * dens, 0.0);
	}
	else
	{
		return MAKE_REAL6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
	}
}



__global__ void calcInteractionParticlesOnFluidWSDP(Particle *p,
	uint *gridParticleIndex,
	uint *cellStart,
	uint *cellEnd,
	Particle *pPDPF,
	Parameters *par)
{
	uint index = threadIdx.x + blockIdx.x*blockDim.x;
	if (index < par->N) {

		real2 pos = MAKE_REAL2(p[index].pos.x, p[index].pos.y);

		int2 gridPos = calcGridPos(pos, par);
		uint gridHash0 = calcGridHash(gridPos, par);
		real6 result = MAKE_REAL6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

		for (int y = -1; y <= 1; y++) {
			for (int x = -1; x <= 1; x++) {
				int2 gridPos2;
				gridPos2.x = gridPos.x + x;
				gridPos2.y = gridPos.y + y;
				if ((gridPos2.x < 0) || (gridPos2.x > par->NXC - 1) || (gridPos2.y < 0) || (gridPos2.y > par->NYC - 1)) continue;

				uint gridHash = calcGridHash(gridPos2, par);
				uint startIndex = cellStart[gridHash];

				if (startIndex != 0xffffffff)
				{
					uint endIndex = cellEnd[gridHash];

					for (uint j = startIndex; j < endIndex; j++)
					{
						real2 dpos, dvel;
						real2 pos1 = p[index].pos;
						real2 pos2 = pPDPF[j].pos;
						real2 vel1 = p[index].vel;
						real2 vel2 = pPDPF[j].vel;
						calcRelPosVelNoSlip(pos1, pos2, vel1, vel2, 0, &dpos, &dvel, par);
						result += interaction(j, pPDPF, index, p, dpos, dvel, par);

						if (((gridPos.x == 0) && (gridPos2.x == 0)) || ((gridPos.x == par->NXC - 1) && (gridPos2.x == par->NXC - 1)) ||
							((gridPos.y == 0) && (gridPos2.y == 0)) || ((gridPos.y == par->NYC - 1) && (gridPos2.y == par->NYC - 1)))
						{
							if (par->T_BOUNDARY_PERIODICITY != 1)
							{
								if (gridPos.y == gridPos2.y)
								{
									if (gridPos.y == par->NYC - 1)
									{
										calcRelPosVelNoSlip(pos2, pos1, vel2, vel1, 1, &dpos, &dvel, par);
										result += interaction(j, pPDPF, index, p, dpos, dvel, par);
									}
									if (gridPos.y == 0)
									{
										calcRelPosVelNoSlip(pos2, pos1, vel2, vel1, 3, &dpos, &dvel, par);
										result += interaction(j, pPDPF, index, p, dpos, dvel, par);
									}
								}
							}
							if (par->T_BOUNDARY_PERIODICITY == 0)
							{
								if (gridPos.x == gridPos2.x)
								{
									if (gridPos2.x == 0)
									{
										calcRelPosVelNoSlip(pos2, pos1, vel2, vel1, 4, &dpos, &dvel, par);
										result += interaction(j, pPDPF, index, p, dpos, dvel, par);
									}
									if (gridPos2.x == par->NXC - 1)
									{
										calcRelPosVelNoSlip(pos2, pos1, vel2, vel1, 2, &dpos, &dvel, par);
										result += interaction(j, pPDPF, index, p, dpos, dvel, par);
									}
								}
								if ((gridPos.x == gridPos2.x) && (gridPos.y == gridPos.y))
								{
									if ((gridPos.x == 0) && (gridPos.y == 0))
									{
										calcRelPosVelNoSlip(pos2, pos1, vel2, vel1, 7, &dpos, &dvel, par);
										result += interaction(j, pPDPF, index, p, dpos, dvel, par);
									}
									if ((gridPos.x == par->NXC - 1) && (gridPos.y == 0))
									{
										calcRelPosVelNoSlip(pos2, pos1, vel2, vel1, 6, &dpos, &dvel, par);
										result += interaction(j, pPDPF, index, p, dpos, dvel, par);
									}
									if ((gridPos.x == 0) && (gridPos.y == par->NYC - 1))
									{
										calcRelPosVelNoSlip(pos2, pos1, vel2, vel1, 8, &dpos, &dvel, par);
										result += interaction(j, pPDPF, index, p, dpos, dvel, par);
									}
									if ((gridPos.x == par->NXC - 1) && (gridPos.y == par->NYC - 1))
									{
										calcRelPosVelNoSlip(pos2, pos1, vel2, vel1, 5, &dpos, &dvel, par);
										result += interaction(j, pPDPF, index, p, dpos, dvel, par);
									}
								}
							}

						}
					}

				}
			}
		}

		if ((par->T_BOUNDARY_PERIODICITY > 0) && ((gridPos.x == 0) || (gridPos.x == par->NXC - 1)))
		{
			for (int y = -1; y <= 1; y++)
			{
				int2 gridPos2;
				if (gridPos.x == 0) gridPos2.x = par->NXC - 1;
				if (gridPos.x == par->NXC - 1) gridPos2.x = 0;
				gridPos2.y = gridPos.y + y;
				if ((gridPos2.y < 0) || (gridPos2.y > par->NYC - 1)) continue;

				uint gridHash = calcGridHash(gridPos2, par);
				uint startIndex = cellStart[gridHash];

				if (startIndex != 0xffffffff)
				{
					uint endIndex = cellEnd[gridHash];

					for (uint j = startIndex; j < endIndex; j++)
					{
						real2 dpos, dvel;
						real2 pos1 = p[index].pos;
						if (gridPos.x == 0) pos1.x += par->XCV;
						if (gridPos.x == par->NXC - 1) pos1.x -= par->XCV;
						real2 pos2 = pPDPF[j].pos;
						real2 vel1 = p[index].vel;
						real2 vel2 = pPDPF[j].vel;

						calcRelPosVelNoSlip(pos1, pos2, vel1, vel2, 0, &dpos, &dvel, par);
						result += interaction(j, pPDPF, index, p, dpos, dvel, par);
					}

				}
			}
		}


		if ((par->T_BOUNDARY_PERIODICITY == 1) && ((gridPos.y == 0) || (gridPos.y == par->NYC - 1)))
		{
			for (int x = -1; x <= 1; x++)
			{
				int2 gridPos2;
				if (gridPos.y == 0) gridPos2.y = par->NYC - 1;
				if (gridPos.y == par->NYC - 1) gridPos2.y = 0;
				gridPos2.x = gridPos.x + x;
				if ((gridPos2.x < 0) || (gridPos2.x > par->NXC - 1)) continue;

				uint gridHash = calcGridHash(gridPos2, par);
				uint startIndex = cellStart[gridHash];

				if (startIndex != 0xffffffff)
				{
					uint endIndex = cellEnd[gridHash];

					for (uint j = startIndex; j < endIndex; j++)
					{
						real2 dpos, dvel;
						real2 pos1 = p[index].pos;
						if (gridPos.y == 0) pos1.y += par->YCV;
						if (gridPos.y == par->NYC - 1) pos1.y -= par->YCV;
						real2 pos2 = pPDPF[j].pos;
						real2 vel1 = p[index].vel;
						real2 vel2 = pPDPF[j].vel;

						calcRelPosVelNoSlip(pos1, pos2, vel1, vel2, 0, &dpos, &dvel, par);
						result += interaction(j, pPDPF, index, p, dpos, dvel, par);
					}

				}
			}
		}

		if (par->T_BOUNDARY_PERIODICITY == 1)
		{
			if (((gridPos.x == 0) && (gridPos.y == 0))
				|| (gridPos.x == 0) && (gridPos.y == par->NYC - 1)
				|| (gridPos.x == par->NXC - 1) && (gridPos.y == 0)
				|| (gridPos.x == par->NXC - 1) && (gridPos.y == par->NYC - 1))
			{
				int2 gridPos2;

				if (gridPos.x == 0) gridPos2.x = par->NXC - 1;
				else gridPos2.x = 0;

				if (gridPos.y == 0) gridPos2.y = par->NYC - 1;
				else gridPos2.y = 0;

				uint gridHash = calcGridHash(gridPos2, par);
				uint startIndex = cellStart[gridHash];

				if (startIndex != 0xffffffff)
				{
					uint endIndex = cellEnd[gridHash];
					for (uint j = startIndex; j < endIndex; j++)
					{
						real2 dpos, dvel;
						real2 pos1 = p[index].pos;
						if (gridPos.x == 0) pos1.x += par->XCV;
						else pos1.x -= par->XCV;
						if (gridPos.y == 0) pos1.y += par->YCV;
						else pos1.y -= par->YCV;
						real2 pos2 = pPDPF[j].pos;
						real2 vel1 = p[index].vel;
						real2 vel2 = pPDPF[j].vel;

						calcRelPosVelNoSlip(pos1, pos2, vel1, vel2, 0, &dpos, &dvel, par);
						result += interaction(j, pPDPF, index, p, dpos, dvel, par);
					}
				}
			}

		}

		if (par->T_BOUNDARY_PERIODICITY == 2)
		{
			if (((gridPos.x == 0) && (gridPos.y == 0))
				|| (gridPos.x == 0) && (gridPos.y == par->NYC - 1)
				|| (gridPos.x == par->NXC - 1) && (gridPos.y == 0)
				|| (gridPos.x == par->NXC - 1) && (gridPos.y == par->NYC - 1))
			{
				int2 gridPos2;

				if (gridPos.x == 0) gridPos2.x = par->NXC - 1;
				else gridPos2.x = 0;

				gridPos2.y = gridPos.y;

				uint gridHash = calcGridHash(gridPos2, par);
				uint startIndex = cellStart[gridHash];

				if (startIndex != 0xffffffff)
				{
					uint endIndex = cellEnd[gridHash];
					for (uint j = startIndex; j < endIndex; j++)
					{
						real2 dpos, dvel;
						real2 pos1 = p[index].pos;
						if (gridPos.x == 0) pos1.x += par->XCV;
						else pos1.x -= par->XCV;
						real2 pos2 = pPDPF[j].pos;
						real2 vel1 = p[index].vel;
						real2 vel2 = pPDPF[j].vel;

						if (gridPos.y == par->NYC - 1)
						{
							calcRelPosVelNoSlip(pos2, pos1, vel2, vel1, 1, &dpos, &dvel, par);
							result += interaction(j, pPDPF, index, p, dpos, dvel, par);
						}
						if (gridPos.y == 0)
						{
							calcRelPosVelNoSlip(pos2, pos1, vel2, vel1, 3, &dpos, &dvel, par);
							result += interaction(j, pPDPF, index, p, dpos, dvel, par);
						}

					}
				}
			}

		}

		p[index].rh_vel.x = -( (result.x * p[index].p) + result.z) / p[index].d;
		p[index].rh_vel.y = -( (result.y * p[index].p) + result.u) / p[index].d;
		p[index].rh_m = -result.v / (pPDPF[0].di);
	}
}

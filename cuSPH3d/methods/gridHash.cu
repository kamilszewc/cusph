/*
*  gridHash.cuh
*
*  Author: Kamil Szewc (kamil.szewc@gmail.com)
*  Modified on: 11-04-2013
*
*/

#include "../sph.h"
#include "../methods/calcGridPos.cuh"
#include "../methods/calcGridHash.cuh"

__global__ void gridHash(
	uint *gridParticleHash,  // output: grid hashes
	uint *gridParticleIndex, // output: particle indices
	Particle *p,     // input: particle array
	Parameters *par) // input: parameters
{
	uint tid = threadIdx.x + blockIdx.x*blockDim.x;
	while (tid < par->N) {
		volatile real3 pos = p[tid].pos;

		// get address in grid
		int3 gridPos = calcGridPos(MAKE_REAL3(pos.x, pos.y, pos.z), par);
		uint hash = calcGridHash(gridPos, par);

		gridParticleHash[tid] = hash;
		gridParticleIndex[tid] = tid;

		tid += blockDim.x*gridDim.x;
	}
}


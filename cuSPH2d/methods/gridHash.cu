/*
*  @file gridHash.cuh
*  @author Kamil Szewc (kamil.szewc@gmail.com)
*  @since 11-04-2013
*/
#include "../sph.h"
#include "../hlp.h"
#include "../methods/calcGridPos.cuh"
#include "../methods/calcGridHash.cuh"

/**
 * @brief Calculates array of grid hashes and array of particle indices
 * @param[out] gridParticleHash Grid hashes
 * @param[out] gridParticleIndex Particle indices
 * @param[in] p Particle array
 * @param[in] par Parameters
 * @param[in] N Number of particles
 */
__global__ void gridHash(
	uint *gridParticleHash,  // output: grid hashes
	uint *gridParticleIndex, // output: particle indices
	Particle *p,     // input: particle array
	Parameters *par, // input: parameters
	uint N)			// input: number of particles
{
	uint tid = threadIdx.x + blockIdx.x*blockDim.x;
	while (tid < N) {
		volatile real2 pos = p[tid].pos;

		int2 gridPos = calcGridPos(MAKE_REAL2(pos.x, pos.y), par);
		uint hash = calcGridHash(gridPos, par);

		gridParticleHash[tid] = hash;
		gridParticleIndex[tid] = tid;

		tid += blockDim.x*gridDim.x;
	}
}


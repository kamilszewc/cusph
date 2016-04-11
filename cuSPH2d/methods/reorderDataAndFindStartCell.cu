/*
*  @file creorderDataAndFindStartCell.cuh
*  @author Kamil Szewc (kamil.szewc@gmail.com)
*  @since 14-12-2014
*/

#include "../sph.h"

/**
 * @brief Creates array of cell start indices, cell end indices
 * @param[out] cellStart Array of cell start indices
 * @param[out] cellEnd Array of cell end indices
 * @param[out] pSort Not used anymore
 * @param[in] gridParticleHash Sorted array of grid hashes
 * @param[in] gridParticleIndex Sorted array of particle indices
 * @param[in] p Not used anymore
 * @param[in] numParticles Number of particles
 */
__global__ void reorderDataAndFindCellStart(
	uint *cellStart,  // output: cell start index
	uint *cellEnd,    // output: cell end index
	Particle *pSort, // output: reordered particle array
	uint *gridParticleHash, // input: sorted grid hashes
	uint *gridParticleIndex,// input: sorted particle indices
	Particle *p,            // input: particle array
	uint numParticles) // input: number of particles
{
	extern __shared__ uint sharedHash[];    // blockSize + 1 elements
	uint index = blockIdx.x*blockDim.x + threadIdx.x;

	uint hash;

	if (index < numParticles)
	{
		hash = gridParticleHash[index];
		sharedHash[threadIdx.x + 1] = hash;

		if (index > 0 && threadIdx.x == 0)
		{
			sharedHash[0] = gridParticleHash[index - 1];
		}
	}
	__syncthreads();

	if (index < numParticles)
	{
		if (index == 0 || hash != sharedHash[threadIdx.x])
		{
			cellStart[hash] = index;

			if (index > 0)
			{
				cellEnd[sharedHash[threadIdx.x]] = index;
			}
		}

		if (index == numParticles - 1)
		{
			cellEnd[hash] = index + 1;
		}
	}
}






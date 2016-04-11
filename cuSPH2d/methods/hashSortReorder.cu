/*
*  @file hashSortReorder.cu
*  @author Kamil Szewc (kamil.szewc@gmail.com)
*  @since 26-09-2014
*/

#include <cuda_runtime.h>
#include "../sph.h"
#include "../errlog.h"

__global__ void gridHash(uint *gridParticleHash, uint *gridParticleIndex, Particle *p, Parameters *par, uint N);
void sortParticles(uint *dGridParticleHash, uint *dGridParticleIndex, uint numParticles);
__global__ void reorderDataAndFindCellStart(uint *cellStart, uint *cellEnd, Particle *p_sort, uint *gridParticleHash, uint *gridParticleIndex, Particle *p, uint numParticles);


void hashSortReorder(
	int NOB,  // input: number of blocks
	int TPB,  // input: number of threads per block
	Particle *p,     // input: particle data
	Parameters *par, // input: parameters
	Particle *pSort, // output: sorted particle data
	uint *gridParticleHash,  // input/output: grid hashes->sorted grid hashes
	uint *gridParticleIndex, // input/output: particle indices->sorted particle indices
	uint *cellStart,  // output: cell start index
	uint *cellEnd,    // output: cell start index
	int numParticles) // input: number of particles
{
	STARTLOG("logs/methods.log");

	gridHash <<<NOB, TPB>>>(gridParticleHash, gridParticleIndex, p, par, numParticles);
	HANDLE_CUDA_KERNEL_RUNTIME_ERROR("gridHash");

	sortParticles(gridParticleHash, gridParticleIndex, numParticles);

	uint smemSize = sizeof(uint)*(TPB + 1);
	reorderDataAndFindCellStart <<<NOB, TPB, smemSize>>>(cellStart, cellEnd, pSort, gridParticleHash, gridParticleIndex, p, numParticles);
	HANDLE_CUDA_KERNEL_RUNTIME_ERROR("reorderDataAndFindCellStart");
}

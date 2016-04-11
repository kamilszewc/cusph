/*
* @file sortParticles.cu
* @author Kamil Szewc (kamil.szewc@gmail.com)
* @since 13-04-2013
*/

#include <thrust/device_ptr.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include "../sph.h"

/**
 * Sorts particles.
 */
void sortParticles(
	uint *gridParticleHash,  // input/output: grid hashes
	uint *gridParticleIndex, // input/output: particle indices
	uint numParticles)       // input: number of particles
{
	thrust::sort_by_key(thrust::device_ptr<uint>(gridParticleHash),
		thrust::device_ptr<uint>(gridParticleHash + numParticles),
		thrust::device_ptr<uint>(gridParticleIndex));
}

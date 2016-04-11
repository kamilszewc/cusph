/*
*  reorderDataAndFindStartCell.cu
*
*  Author: Kamil Szewc (kamil.szewc@gmail.com)
*  Modified on: 12-04-2013
*
*/

#include "../sph.h"

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

	// handle case when no. of particles not multiple of block size
	if (index < numParticles)
	{
		hash = gridParticleHash[index];

		// Load hash data into shared memory so that we can look
		// at neighboring particle's hash value without loading
		// two hash values per thread
		sharedHash[threadIdx.x + 1] = hash;

		if (index > 0 && threadIdx.x == 0)
		{
			// first thread in block must load neighbor particle hash
			sharedHash[0] = gridParticleHash[index - 1];
		}
	}
	__syncthreads();

	if (index < numParticles)
	{
		// If this particle has a different cell index to the previous
		// particle then it must be the first particle in the cell,
		// so store the index of this particle in the cell.
		// As it isn't the first particle, it must also be the cell end of
		// the previous particle's cell

		if (index == 0 || hash != sharedHash[threadIdx.x])
		{
			cellStart[hash] = index;

			if (index > 0)
				cellEnd[sharedHash[threadIdx.x]] = index;
		}

		if (index == numParticles - 1)
		{
			cellEnd[hash] = index + 1;
		}

		// Now use the sorted index to reorder the pos and vel data
		/*uint sortedIndex = gridParticleIndex[index];

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
		pSort[index].cw = cw;*/
	}
}






/**
 * @file hashSortReorder.cuh
 * @author Kamil Szewc (kamil.szewc@gmail.com)
 */
#if !defined(__HASH_SORT_REORDER_H__)
#define __HASH_SORT_REORDER_H__

/**
 * @brief Sorts particle array, creates sorted grid hashes and sorted particle indices, creates cell start index and cell end index arrays.
 * @param[in] NOB Number of blocks (CUDA device)
 * @param[in] TPB Number of threads per block (CUDA device)
 * @param[in] p Particle array
 * @param[in] par Parameters
 * @param[out] pSort Sorted particle array
 * @param[in,out] gridParticleHash Array of sorted grid hashes
 * @param[in,out] gridParticleIndex Array of sorted particle indices
 * @param[out] cellStart Cell start index array
 * @param[out] cellEnd Cell end index array
 * @param[in] numParticles Length of particle array
 */
void hashSortReorder(int NOB, int TPB, Particle *p, Parameters *par, Particle *pSort, uint *gridParticleHash, uint *gridParticleIndex, uint *cellStart, uint *cellEnd, int numParticles);

#endif

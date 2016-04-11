/**
 * @file copyParticles.cuh
 * @author Kamil Szewc (kamil.szewc@gmail.com)
 */

#if !defined(__COPY_PARTICLES_CUH__)
#define __COPY_PARTICLES_CUH__

/**
 * @brief Copies particle's from one array - can sort particles in memory
 * @param[in,out] pSort Sorted array of particle
 * @param[in,out] p Array of particle
 * @param[in] gridParticleIndex
 * @param[in] sorted If true: copy in sorted manner, if false: just copy
 * @param[in] par Parameters
 * @param[in] N Length of array
 */
__global__ void copyParticles(Particle *pSort, Particle *p, uint *gridParticleIndex, bool sorted, Parameters *par, uint N);

#endif

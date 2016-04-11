#if !defined(__COPY_PARTICLES_CUH__)
#define __COPY_PARTICLES_CUH__

__global__ void copyParticles(Particle *pSort, Particle *p, uint *gridParticleIndex, bool sorted, Parameters *par);

#endif
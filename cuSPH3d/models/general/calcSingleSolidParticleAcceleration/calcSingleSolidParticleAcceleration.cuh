#if !defined(__CALC_SINGLE_SOLID_PARTICLE_ACCELERATION__)
#define __CALC_SINGLE_SOLID_PARTICLE_ACCELERATION__

void calcSingleSolidParticleAcceleration(int NOB, int TPB, thrust::device_vector<Particle>& p, Parameters *par);


#endif
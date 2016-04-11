#if !defined(__RENORMALIZE_PRESSURE__)
#define __RENORMALIZE_PRESSURE__

void renormalizePressure(int NOB, int TPB, Particle *particle, Parameters *par, uint numParticles);

#endif

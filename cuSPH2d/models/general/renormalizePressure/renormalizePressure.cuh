#if !defined(__RENORMALIZE_PRESSURE__)
#define __RENORMALIZE_PRESSURE__

/**
 * @brief Renormalize pressure to remove negative values
 * @param[in] NOB Number of blocks
 * @param[in] TPB Number of threads per block
 * @param[in,out] particle Particle array
 * @param[in] par Parameters
 * @param[in] numParticle Number of particles
 */
void renormalizePressure(int NOB, int TPB, Particle *particle, Parameters *par, uint numParticles);

#endif

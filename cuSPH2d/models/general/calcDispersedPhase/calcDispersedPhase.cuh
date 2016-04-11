#if !defined(__DISPERSED_PHASE_CUH__)
#define __DISPERSED_PHASE_CUH__

/**
 * @brief Calculates interactions between fluid and dispersed phase.
 * @param[in,out] p Particle array
 * @param[in] gridParticleIndex Particle-in-cell list of indexes (sorted)
 * @param[in] cellStart Array of beginnings of cell in gridParticleIndex
 * @param[in] cellEnd Array of ends of cell in gridParticleIndex
 * @param[in,out] pDispersedPhase ParticleDispersedPhase array
 * @param[in] par Parameters
 */
__global__ void calcDispersedPhaseField(Particle *p,
										uint *gridParticleIndex,
										uint *cellStart,
										uint *cellEnd,
										ParticleDispersedPhase *pDispersedPhase,
										Parameters *par);

/**
 * @brief Calculates avelocity and position of dispersed phase particles after interaction with fluid
 * @param[in,out] p Array of ParticleDispersedPhase
 * @param[in] par Parameters
 */
__global__ void calcDispersedPhaseAdvection(ParticleDispersedPhase *p, Parameters *par);

#endif

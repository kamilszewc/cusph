#if !defined(__CALC_TIME_STEP_CUH__)
#define __CALC_TIME_STEP_CUH__

/**
 * @brief Calculates time step
 * @param[in] p Particle array (device)
 * @param[in,out] par Parameters (device)
 * @param[in,out] parHost Parameters (host)
 */
void calcTimeStep(thrust::device_vector<Particle>& p, Parameters *par, Parameters *parHost);

/**
 * @brief Calculates time step with some correction to CFL (check definition, EXPERIMENTAL)
 * @param[in] p Particle array (device)
 * @param[in,out] par Parameters (device)
 * @param[in,out] parHost Parameters (host)
 * @param[in] value Value of correction parameter
 */
void calcTimeStep(thrust::device_vector<Particle>& p, Parameters *par, Parameters *parHost, const real value);

#endif

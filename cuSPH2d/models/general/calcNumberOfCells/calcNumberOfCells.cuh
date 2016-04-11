#if !defined(__CALC_NUMBER_OF_CELLS_CUH__)
#define __CALC_NUMBER_OF_CELLS_CUH__

/**
 * @brief Calculates number of cells when variable smoothing length is used.
 * @param[in] p Array of particles (device)
 * @param[in,out] par Parameters (device)
 * @param[in,out] parHost Parameters (host)
 */
void calcNumberOfCells(thrust::device_vector<Particle>& p, Parameters *par, Parameters *parHost);

#endif

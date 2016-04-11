#if !defined(__CALC_TIME_STEP_CUH__)
#define __CALC_TIME_STEP_CUH__

void calcTimeStep(thrust::device_vector<Particle>& p, Parameters *par, Parameters *parHost);

void calcTimeStep(thrust::device_vector<Particle>& p, Parameters *par, Parameters *parHost, const real value);

#endif
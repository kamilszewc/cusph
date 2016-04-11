/*
* @file kernels.cuh
* @author Kamil Szewc (kamil.szewc@gmail.com)
* @since 13-04-2013
*/

#if !defined(__KERNELS_CUH__)
#define __KERNELS_CUH__

#include "../hlp.h"

/**
 * @brief To choose kernel type.
 * @desc 0. Wendland (1995), 1. Tartakovsky Meakin (2005), 2. Lucy (1977)
 */
#define KERNEL_TYPE 0

#if KERNEL_TYPE == 0

/**
 * @brief Calculates kernel function
 * @param[in] q \f$=|r|/h\f$
 * @param[in] i_h \f$=1.0/h \f$
 * @return Value
 */
__device__ static real kern(real q, real i_h)
{
	if (q < 2.0)
		return M_1_PI * pow2(i_h) * 0.21875 * pow4(2.0 - q) * (q + 0.5);
	else
		return 0.0;
}

/**
 * @brief Calculates gradient of kernel function
 * @param[in] x One component of distance vector between two particles
 * @param[in] q \f$ =|r|/h \f$
 * @param[in] i_h \f$ =1.0/h \f$
 * @return Value
 */
__device__ static real grad_of_kern(real x, real q, real i_h)
{
	if (q < 2.0)
		return -M_1_PI * pow4(i_h) * 1.09375 * x * pow3(2.0 - q);
	else
		return 0.0;
}

/**
 * @brief Calculates half-range kernel function
 * @param[in] q \f$=|r|/h\f$
 * @param[in] i_h \f$=1.0/h \f$
 * @return Value
 */
__device__ static real kern_half(real q, real i_h)
{
	if (q < 1.0)
		return M_1_PI * pow2(i_h) * 0.21875 * pow4(2.0 - 2.0*q) * (2.0*q + 0.5) * 4.0;
	else
		return 0.0;
}

/**
 * @brief Calculates half-range gradient of kernel function
 * @param[in] x One component of distance vector between two particles
 * @param[in] q \f$ =|r|/h \f$
 * @param[in] i_h \f$ =1.0/h \f$
 * @return Value
 */
__device__ static real grad_of_kernel_half(real x, real q, real i_h)
{
	if (q < 1.0)
		return -M_1_PI * pow4(i_h) * (0.21875*80.0) * x * pow3(2.0 - 2.0*q);
	else
		return 0.0;
}

#elif KERNEL_TYPE == 1
// Tartakovsky and Meakin (2005)

__device__ static real kern(real q, real i_h)
{
	real value = 0.0;

	if (q <= 2.0)
		value += pow5(3.0 - 1.5 * q);
	if (q <= 4.0/3.0)
		value -= 6.0 * pow5(2.0 - 1.5 * q);
	if (q <= 2.0/3.0)
		value += 15.0 * pow5(1.0 - 1.5 * q);

	return value * M_1_PI * pow2(i_h) * (0.25 * 63.0 / 478.0);
}	

__device__ static real grad_of_kern(real x, real q, real i_h)
{
	real value = 0.0;

	if (q <= 2.0)
		value += 7.5 * pow4(3.0 - 1.5 * q) * x / q;
	if (q <= 4.0/3.0)
		value -= 6.0 * 7.5 * pow4(2.0 - 1.5 * q) * x / q;
	if (q <= 2.0/3.0)
		value += 15.0 * 7.5 * pow4(1.0 - 1.5 * q) * x / q;

	return  -value * M_1_PI * pow4(i_h) * (0.25 * 63.0 / 478.0);
}

__device__ static real kern_half(real q, real i_h)
{
	real value = 0.0;

	if (q <= 1.0)
		value += pow5(3.0 - 3.0 * q);
	if (q <= 2.0/3.0)
		value -= 6.0 * pow5(2.0 - 3.0 * q);
	if (q <= 1.0/3.0)
		value += 15.0 * pow5(1.0 - 3.0 * q);

	return value * M_1_PI * pow2(i_h) * (63.0 / 478.0);
}	

__device__ static real grad_of_kernel_half(real x, real q, real i_h)
{
	real value = 0.0;

	if (q <= 1.0)
		value += 15.0 * pow4(3.0 - 3.0 * q) * x / q;
	if (q <= 2.0/3.0)
		value -= 6.0 * 15.0 * pow4(2.0 - 3.0 * q) * x / q;
	if (q <= 1.0/3.0)
		value += 15.0 * 15.0 * pow4(1.0 - 3.0 * q) * x / q;

	return  -value * M_1_PI * pow4(i_h) * (63.0 / 478.0);
}

#elif KERNEL_TYPE == 2
// Lucy (1977)

__device__ static real kern(real q, real i_h)
{
	if (q <= 2.0)
		 return M_1_PI * pow2(i_h) * (5.0/4.0) * (1.0 + 1.5 * q) * pow3(1.0 - 0.5 * q);
	else 
		return 0.0;
}	

__device__ static real grad_of_kern(real x, real q, real i_h)
{
	if (q <= 2.0)
		return - GM_1_PI * pow4(i_h) * (15.0/4.0) * pow2(1.0 - 0.5 * q) * x;
	else
		return 0.0;
}

__device__ static real kern_half(real q, real i_h)
{
	if (q <= 1.0)
		return M_1_PI * pow2(i_h) * 5.0 * (1.0 + 3.0 * q) * pow3(1.0 - q);
	else
		return 0.0;
}	

__device__ static real grad_of_kernel_half(real x, real q, real i_h)
{
	if (q <= 1.0)
		return - M_1_PI * pow4(i_h) * (60.0) * pow2(1.0 - q) * x;
	else
		return  0.0;
}

#endif


/**
 * @brief Calculates gradient of kernel function for dispersed phase-fluid interactions in 2-fluid model
 * @desc Proposed by Price, Kwon and Monaghan
 * @param[in] x One component of distance vector between two particles
 * @param[in] q \f$ =|r|/h \f$
 * @param[in] i_h \f$ =1.0/h \f$
 * @return Value
 */
__device__ static real kern_kwon_monaghan(real q, real i_h)
{
	return  M_1_PI * 0.39375 * pow2(i_h) * pow2(q) * (0.5 + q) * pow4(2.0 - q);
}

#endif

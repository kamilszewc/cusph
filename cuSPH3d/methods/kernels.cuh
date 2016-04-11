/*
*  kernels.cuh
*
*  Author: Kamil Szewc (kamil.szewc@gmail.com)
*  Modified on: 27-09-2014
*
*/

__host__ __device__ static real kern(real q, real KNORM)
{
	if (q < 2.0)
		return KNORM * (42.0/256.0) * pow4(2.0 - q) * (q + 0.5);
	else
		return 0.0;
}


__host__ __device__ static real grad_of_kern(real x, real q, real GKNORM)
{
	if (q < 2.0)
		return GKNORM * (-5.0*42.0/256.0) * x * pow3(2.0 - q);
	else
		return 0.0;
}

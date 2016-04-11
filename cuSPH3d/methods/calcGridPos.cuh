/*
* calcGridPos.cuh
*
*  Author: Kamil Szewc (kamil.szewc@gmail.com)
*  Modified on: 09-10-2013
*
*/

/* Calculates position of particle on the auxiliary grid */

__device__ static int3 calcGridPos(
	real3 pos,      // input: particle position
	Parameters *par) // input: parameters
{
	int3 gridPos;
	gridPos.x = floor(0.5 * pos.x * par->I_H);
	gridPos.y = floor(0.5 * pos.y * par->I_H);
	gridPos.z = floor(0.5 * pos.z * par->I_H);
	return gridPos;
}


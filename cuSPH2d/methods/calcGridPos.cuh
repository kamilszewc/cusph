/**
* @file calcGridPos.cuh
* @author Kamil Szewc (kamil.szewc@gmail.com)
* @since 09-10-2013
*/

#if !defined(__CALC_GRID_POS_CUH__)
#define __CALC_GRID_POS_CUH__

/**
 * @brief Calculates position of particle on the auxiliary grid.
 * @param[in] pos Particle position
 * @param[in] par Parameters
 */

__device__ static int2 calcGridPos(
	real2 pos,
	Parameters *par)
{
	int2 gridPos;
	gridPos.x = floor(0.5 * pos.x * par->I_H);
	gridPos.y = floor(0.5 * pos.y * par->I_H);
	return gridPos;
}

#endif


/**
* @file calcGridHash.cuh
* @author Kamil Szewc (kamil.szewc@gmail.com)
* @since 24-01-2015
*/

#if !defined(__CALC_GRID_HASH_CUH__)
#define __CALC_GRID_HASH_CUH__

/* Calculate hashes (ids) of cells from grid coordinates */

//__device__ static uint calcGridHash(
//	int2 gridPos,    // input: cell coordinates	
//	Parameters *par) // input: parameters
//{
//	gridPos.x = gridPos.x & (par->NXC - 1);
//	gridPos.y = gridPos.y & (par->NYC - 1);
//	return __umul24(gridPos.y, par->NXC) + gridPos.x;
//}

/**
 * @brief Calculates hash (id) of the cell for given cell coordinates
 * @param[in] grisPos Cell coordinates
 * @param[in] par Parameters
 * @return hash id
 */
__device__ static uint calcGridHash(
	int2 gridPos,    // input: cell coordinates	
	Parameters *par) // input: parameters
{
	gridPos.x = gridPos.x % par->NXC;
	gridPos.y = gridPos.y % par->NYC;

	return (uint)(gridPos.y) * (uint)(par->NXC) + (uint)(gridPos.x);
}
/*__device__ static uint calcGridHash(
	int2 gridPos,    // input: cell cooridinates
	Parameters *par) // input: parameters
{
	if (par->NXC % 2 == 0)
	{
		gridPos.x = gridPos.x & (par->NXC - 1);
		gridPos.y = gridPos.y & (par->NYC - 1);
		return __umul24(gridPos.y, par->NXC) + gridPos.x;
	}
	else
	{
		gridPos.x = gridPos.x % (par->NXC - 1);
		gridPos.y = gridPos.y % (par->NYC - 1);
		return (uint)(gridPos.y) * (uint)(par->NXC) + (uint)(gridPos.x);
	}
	
}*/

#endif

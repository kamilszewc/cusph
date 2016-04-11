/*
*  calcGridHash.cuh
*
*  Author: Kamil Szewc (kamil.szewc@gmail.com)
*  Modified on: 09-10-2013
*
*/

/* Calculate hashes (ids) of cells from grid coordinates */

/*__device__ static uint calcGridHash(
	int3 gridPos,    // input: cell coordinates
	Parameters *par) // input: parameters
{
	gridPos.x = gridPos.x & (par->NXC - 1);
	gridPos.y = gridPos.y & (par->NYC - 1);
	gridPos.z = gridPos.z & (par->NZC - 1);
	return __umul24(__umul24(gridPos.z, par->NYC), par->NXC) +
		__umul24(gridPos.y, par->NXC) + gridPos.x;
}*/

__device__ static uint calcGridHash(
	int3 gridPos,    // input: cell coordinates
	Parameters *par) // input: parameters
{
	gridPos.x = gridPos.x % par->NXC;
	gridPos.y = gridPos.y % par->NYC;
	gridPos.z = gridPos.z % par->NZC;
	return ((uint)gridPos.z * (uint)par->NYC) * (uint)par->NXC + ((uint)gridPos.y * (uint)par->NXC) + (uint)gridPos.x;
}


/*
*  calcRelPosVelFreeSlip.cuh
*
*  Author: Kamil Szewc (kamil.szewc@gmail.com)
*  Modified on: 27-09-2014
*
*/

/*
*           ________
*         /   T    /|
*        /________/ | 
*        |       | E|
*        |   S   | /
*  z|    |_______|/
*   | /y     
*   |/____x
*/

#if !defined(__CALC_REL_POS_VEL_FREE_SLIP_CUH__)
#define __CALC_REL_POS_VEL_FREE_SLIP_CUH__

__host__ __device__ static void calcRelPosVelFreeSlip(
	real3 pos,   // input: position of particle 1
	real3 pos2,  // input: position of particle 2
	real3 vel,   // input: velocity of particle 1
	real3 vel2,  // input: velocity of particle 2
	uint T,       // input: boundary type indicator
	real3 *dpos, // output: relative position
	real3 *dvel, // output: relative velocity
	Parameters *par) // input: parameters
{
	switch (T) {
	case 0:
		dpos->x = pos.x - pos2.x;
		dpos->y = pos.y - pos2.y;
		dpos->z = pos.z - pos2.z;
		dvel->x = vel.x - vel2.x;
		dvel->y = vel.y - vel2.y;
		dvel->z = vel.z - vel2.z;
		break;

		// Walls
	case 1: // North
		dpos->x = pos.x - pos2.x;
		dpos->y = 2.0*par->YCV - pos.y - pos2.y;
		dpos->z = pos.z - pos2.z;
		dvel->x = vel.x - vel2.x;
		dvel->y = -vel.y - vel2.y;
		dvel->z = vel.z - vel2.z;
		break;
	case 2: // East
		dpos->x = 2.0*par->XCV - pos.x - pos2.x;
		dpos->y = pos.y - pos2.y;
		dpos->z = pos.z - pos2.z;
		dvel->x = -vel.x - vel2.x;
		dvel->y = vel.y - vel2.y;
		dvel->z = vel.z - vel2.z;
		break;
	case 3: // South
		dpos->x = pos.x - pos2.x;
		dpos->y = -pos.y - pos2.y;
		dpos->z = pos.z - pos2.z;
		dvel->x = vel.x - vel2.x;
		dvel->y = -vel.y - vel2.y;
		dvel->z = vel.z - vel2.z;
		break;
	case 4: // West 
		dpos->x = -pos.x - pos2.x;
		dpos->y = pos.y - pos2.y;
		dpos->z = pos.z - pos2.z;
		dvel->x = -vel.x - vel2.x;
		dvel->y = vel.y - vel2.y;
		dvel->z = vel.z - vel2.z;
		break;
	case 5: // Bottom 
		dpos->x = pos.x - pos2.x;
		dpos->y = pos.y - pos2.y;
		dpos->z = -pos.z - pos2.z;
		dvel->x = vel.x - vel2.x;
		dvel->y = vel.y - vel2.y;
		dvel->z = -vel.z - vel2.z;
		break;
	case 6: // Top
		dpos->x = pos.x - pos2.x;
		dpos->y = pos.y - pos2.y;
		dpos->z = 2.0*par->ZCV - pos.z - pos2.z;
		dvel->x = vel.x - vel2.x;
		dvel->y = vel.y - vel2.y;
		dvel->z = -vel.z - vel2.z;
		break;

		// Edges
	case 7: // South-Bottom
		dpos->x = pos.x - pos2.x;
		dpos->y = -pos.y - pos2.y;
		dpos->z = -pos.z - pos2.z;
		dvel->x = vel.x - vel2.x;
		dvel->y = -vel.y - vel2.y;
		dvel->z = -vel.z - vel2.z;
		break;
	case 8: // North-Bottom
		dpos->x = pos.x - pos2.x;
		dpos->y = 2.0*par->YCV - pos.y - pos2.y;
		dpos->z = -pos.z - pos2.z;
		dvel->x = vel.x - vel2.x;
		dvel->y = -vel.y - vel2.y;
		dvel->z = -vel.z - vel2.z;
		break;
	case 9: // South-Top
		dpos->x = pos.x - pos2.x;
		dpos->y = -pos.y - pos2.y;
		dpos->z = 2.0*par->ZCV - pos.z - pos2.z;
		dvel->x = vel.x - vel2.x;
		dvel->y = -vel.y - vel2.y;
		dvel->z = -vel.z - vel2.z;
		break;
	case 10: // North-Top
		dpos->x = pos.x - pos2.x;
		dpos->y = 2.0*par->YCV - pos.y - pos2.y;
		dpos->z = 2.0*par->ZCV - pos.z - pos2.z;
		dvel->x = vel.x - vel2.x;
		dvel->y = -vel.y - vel2.y;
		dvel->z = -vel.z - vel2.z;
		break;
	case 11: // West-Bottom
		dpos->x = -pos.x - pos2.x;
		dpos->y = pos.y - pos2.y;
		dpos->z = -pos.z - pos2.z;
		dvel->x = -vel.x - vel2.x;
		dvel->y = vel.y - vel2.y;
		dvel->z = -vel.z - vel2.z;
		break;
	case 12: // East-Bottom
		dpos->x = 2.0*par->XCV - pos.x - pos2.x;
		dpos->y = pos.y - pos2.y;
		dpos->z = -pos.z - pos2.z;
		dvel->x = -vel.x - vel2.x;
		dvel->y = vel.y - vel2.y;
		dvel->z = -vel.z - vel2.z;
		break;
	case 13: // West-Top
		dpos->x = -pos.x - pos2.x;
		dpos->y = pos.y - pos2.y;
		dpos->z = 2.0*par->ZCV - pos.z - pos2.z;
		dvel->x = -vel.x - vel2.x;
		dvel->y = vel.y - vel2.y;
		dvel->z = -vel.z - vel2.z;
		break;
	case 14: // East-Top
		dpos->x = 2.0*par->XCV - pos.x - pos2.x;
		dpos->y = pos.y - pos2.y;
		dpos->z = 2.0*par->ZCV - pos.z - pos2.z;
		dvel->x = -vel.x - vel2.x;
		dvel->y = vel.y - vel2.y;
		dvel->z = -vel.z - vel2.z;
		break;
	case 15: // South-West
		dpos->x = -pos.x - pos2.x;
		dpos->y = -pos.y - pos2.y;
		dpos->z = pos.z - pos2.z;
		dvel->x = -vel.x - vel2.x;
		dvel->y = -vel.y - vel2.y;
		dvel->z = vel.z - vel2.z;
		break;
	case 16: // South-East
		dpos->x = 2.0*par->XCV - pos.x - pos2.x;
		dpos->y = -pos.y - pos2.y;
		dpos->z = pos.z - pos2.z;
		dvel->x = -vel.x - vel2.x;
		dvel->y = -vel.y - vel2.y;
		dvel->z = vel.z - vel2.z;
		break;
	case 17: // North-West
		dpos->x = -pos.x - pos2.x;
		dpos->y = 2.0*par->YCV - pos.y - pos2.y;
		dpos->z = pos.z - pos2.z;
		dvel->x = -vel.x - vel2.x;
		dvel->y = -vel.y - vel2.y;
		dvel->z = vel.z - vel2.z;
		break;
	case 18: // North-East
		dpos->x = 2.0*par->XCV - pos.x - pos2.x;
		dpos->y = 2.0*par->YCV - pos.y - pos2.y;
		dpos->z = pos.z - pos2.z;
		dvel->x = -vel.x - vel2.x;
		dvel->y = -vel.y - vel2.y;
		dvel->z = vel.z - vel2.z;
		break;

		// Corners
	case 19: // South-West-Bottom
		dpos->x = -pos.x - pos2.x;
		dpos->y = -pos.y - pos2.y;
		dpos->z = -pos.z - pos2.z;
		dvel->x = -vel.x - vel2.x;
		dvel->y = -vel.y - vel2.y;
		dvel->z = -vel.z - vel2.z;
		break;
	case 20: // South-East-Bottom
		dpos->x = 2.0*par->XCV - pos.x - pos2.x;
		dpos->y = -pos.y - pos2.y;
		dpos->z = -pos.z - pos2.z;
		dvel->x = -vel.x - vel2.x;
		dvel->y = -vel.y - vel2.y;
		dvel->z = -vel.z - vel2.z;
		break;
	case 21: // North-West-Bottom
		dpos->x = -pos.x - pos2.x;
		dpos->y = 2.0*par->YCV - pos.y - pos2.y;
		dpos->z = -pos.z - pos2.z;
		dvel->x = -vel.x - vel2.x;
		dvel->y = -vel.y - vel2.y;
		dvel->z = -vel.z - vel2.z;
		break;
	case 22: // North-East-Bottom
		dpos->x = 2.0*par->XCV - pos.x - pos2.x;
		dpos->y = 2.0*par->YCV - pos.y - pos2.y;
		dpos->z = -pos.z - pos2.z;
		dvel->x = -vel.x - vel2.x;
		dvel->y = -vel.y - vel2.y;
		dvel->z = -vel.z - vel2.z;
		break;
	case 23: // South-West-Top
		dpos->x = -pos.x - pos2.x;
		dpos->y = -pos.y - pos2.y;
		dpos->z = 2.0*par->ZCV - pos.z - pos2.z;
		dvel->x = -vel.x - vel2.x;
		dvel->y = -vel.y - vel2.y;
		dvel->z = -vel.z - vel2.z;
		break;
	case 24: // South-East-Top
		dpos->x = 2.0*par->XCV - pos.x - pos2.x;
		dpos->y = -pos.y - pos2.y;
		dpos->z = 2.0*par->ZCV - pos.z - pos2.z;
		dvel->x = -vel.x - vel2.x;
		dvel->y = -vel.y - vel2.y;
		dvel->z = -vel.z - vel2.z;
		break;
	case 25: // North-West-Top
		dpos->x = -pos.x - pos2.x;
		dpos->y = 2.0*par->YCV - pos.y - pos2.y;
		dpos->z = 2.0*par->ZCV - pos.z - pos2.z;
		dvel->x = -vel.x - vel2.x;
		dvel->y = -vel.y - vel2.y;
		dvel->z = -vel.z - vel2.z;
		break;
	case 26: // North-Wast-Top
		dpos->x = 2.0*par->XCV -pos.x - pos2.x;
		dpos->y = 2.0*par->YCV -pos.y - pos2.y;
		dpos->z = 2.0*par->ZCV -pos.z - pos2.z;
		dvel->x = -vel.x - vel2.x;
		dvel->y = -vel.y - vel2.y;
		dvel->z = -vel.z - vel2.z;
		break;
	default:
		break;
	}
}

#endif

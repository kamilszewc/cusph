/*
*  @file calcRelPosVelNoSlip.cuh
*  @author Kamil Szewc (kamil.szewc@gmail.com)
*  @since 27-09-2014
*/

#if !defined(__CALC_REL_POS_VEL_NO_SLIP_CUH__)
#define __CALC_REL_POS_VEL_NO_SLIP_CUH__

/**
 *	@brief Calculates the relative position and velocity between two particles (no-sleep condition on wall)
 *	@param[in] pos Position of particle 1
 *	@param[in] pos2 Position of particle 2
 *	@param[in] vel Velocity of particle 1
 *	@param[in] vel2 Velocity of particle 2
 *	@param[in] T Boundary type indicator (Ghost-particle type of boundary)
 *	@param[out] dpos Relative position
 *	@param[out] dvel Relative velocity
 *	@param[in] par Parameters
 */
__device__ static void calcRelPosVelNoSlip(
	real2 pos,   // input: position of particle 1 
	real2 pos2,  // input: position of particle 2
	real2 vel,   // input: velocity of particle 1
	real2 vel2,  // input: velocity of particle 2
	uint T,       // input: boundary type indicator
	real2 *dpos, // output: relative position
	real2 *dvel, // output: relative velocity
	Parameters *par) // input: parameters
{

	switch (T) {
	case 0:
		dpos->x = pos.x - pos2.x;
		dpos->y = pos.y - pos2.y;
		dvel->x = vel.x - vel2.x;
		dvel->y = vel.y - vel2.y;
		break;
	case 1: /* N */
		dpos->x = pos.x - pos2.x;
		dpos->y = 2.0*par->YCV - pos.y - pos2.y;
		dvel->x = 2.0*par->V_N - vel.x - vel2.x;
		dvel->y = -vel.y - vel2.y;
		break;
	case 2: /* E */
		dpos->x = 2.0*par->XCV - pos.x - pos2.x;
		dpos->y = pos.y - pos2.y;
		dvel->x = -vel.x - vel2.x;
		dvel->y = 2.0*par->V_E - vel.y - vel2.y;
		break;
	case 3: /* S */
		dpos->x = pos.x - pos2.x;
		dpos->y = -pos.y - pos2.y;
		dvel->x = 2.0*par->V_S - vel.x - vel2.x;
		dvel->y = -vel.y - vel2.y;
		break;
	case 4: /* W */
		dpos->x = -pos.x - pos2.x;
		dpos->y = pos.y - pos2.y;
		dvel->x = -vel.x - vel2.x;
		dvel->y = 2.0*par->V_W - vel.y - vel2.y;
		break;
	case 5: /* NE */
		dpos->x = 2.0*par->XCV - pos.x - pos2.x;
		dpos->y = 2.0*par->YCV - pos.y - pos2.y;
		dvel->x = -vel.x - vel2.x;
		dvel->y = -vel.y - vel2.y;
		break;
	case 6: /* SE */
		dpos->x = 2.0*par->XCV - pos.x - pos2.x;
		dpos->y = -pos.y - pos2.y;
		dvel->x = -vel.x - vel2.x;
		dvel->y = -vel.y - vel2.y;
		break;
	case 7: /* SW */
		dpos->x = -pos.x - pos2.x;
		dpos->y = -pos.y - pos2.y;
		dvel->x = -vel.x - vel2.x;
		dvel->y = -vel.y - vel2.y;
		break;
	case 8: /* NW */
		dpos->x = -pos.x - pos2.x;
		dpos->y = 2.0*par->YCV - pos.y - pos2.y;
		dvel->x = -vel.x - vel2.x;
		dvel->y = -vel.y - vel2.y;
		break;
	default:
		break;
	}
}

#endif

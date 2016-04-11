
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

real3 pos = MAKE_REAL3(p[index].pos.x, p[index].pos.y, p[index].pos.z);

int3 gridPos = calcGridPos(pos, par);
uint gridHash0 = calcGridHash(gridPos, par);
uint gridHash, startIndex, endIndex;

int3 gridPos2;
real3 dpos, dvel;
real3 pos1, pos2;
real3 vel1, vel2;
real3 dvelSlip;

for (int z = -1; z <= 1; z++) {
	for (int y = -1; y <= 1; y++) {
		for (int x = -1; x <= 1; x++) {
			gridPos2.x = gridPos.x + x;
			gridPos2.y = gridPos.y + y;
			gridPos2.z = gridPos.z + z;
			if ((gridPos2.x < 0) || (gridPos2.x > par->NXC - 1)
				|| (gridPos2.y < 0) || (gridPos2.y > par->NYC - 1)
				|| (gridPos2.z < 0) || (gridPos2.z > par->NZC - 1)) continue;

			gridHash = calcGridHash(gridPos2, par);
			startIndex = cellStart[gridHash];

			if (startIndex != 0xffffffff) {
				endIndex = cellEnd[gridHash];

				for (uint j = startIndex; j < endIndex; j++) {
					pos1 = p[index].pos;
					pos2 = p[j].pos;
					vel1 = p[index].vel;
					vel2 = p[j].vel;

					calcRelPosVelFreeSlip(pos1, pos2, vel1, vel2, 0, &dpos, &dvelSlip, par);
					result += interaction(index, j, dpos, dvelSlip, p, par);

					if (par->T_BOUNDARY_PERIODICITY != 1)
					{
						if (((gridPos.z == 0) && (gridPos2.z == 0)) || ((gridPos.z == par->NZC - 1) && (gridPos2.z == par->NZC - 1)) ||
							((gridPos.x == 0) && (gridPos2.x == 0)) || ((gridPos.x == par->NXC - 1) && (gridPos2.x == par->NXC - 1)) ||
							((gridPos.y == 0) && (gridPos2.y == 0)) || ((gridPos.y == par->NYC - 1) && (gridPos2.y == par->NYC - 1)))
						{
							if (gridPos.z == gridPos2.z)
							{
								if (gridPos2.z == 0)
								{
									calcRelPosVelFreeSlip(pos2, pos1, vel2, vel1, 5, &dpos, &dvelSlip, par);
									result -= interaction(j, index, dpos, dvelSlip, p, par);
								}
								if (gridPos2.z == par->NZC - 1)
								{
									calcRelPosVelFreeSlip(pos2, pos1, vel2, vel1, 6, &dpos, &dvelSlip, par);
									result -= interaction(j, index, dpos, dvelSlip, p, par);
								}
							}

							if (par->T_BOUNDARY_PERIODICITY == 0)
							{
								if (gridPos.y == gridPos2.y)
								{
									if (gridPos.y == par->NYC - 1)
									{
										calcRelPosVelFreeSlip(pos2, pos1, vel2, vel1, 1, &dpos, &dvelSlip, par);
										result -= interaction(j, index, dpos, dvelSlip, p, par);
									}
									if (gridPos.y == 0)
									{
										calcRelPosVelFreeSlip(pos2, pos1, vel2, vel1, 3, &dpos, &dvelSlip, par);
										result -= interaction(j, index, dpos, dvelSlip, p, par);
									}
								}

								if (gridPos.x == gridPos2.x)
								{
									if (gridPos2.x == 0)
									{
										calcRelPosVelFreeSlip(pos2, pos1, vel2, vel1, 4, &dpos, &dvelSlip, par);
										result -= interaction(j, index, dpos, dvelSlip, p, par);
									}
									if (gridPos2.x == par->NXC - 1)
									{
										calcRelPosVelFreeSlip(pos2, pos1, vel2, vel1, 2, &dpos, &dvelSlip, par);
										result -= interaction(j, index, dpos, dvelSlip, p, par);
									}
								}

								// Edges
								if ((gridPos.z == 0) && (gridPos2.z == 0))
								{
									if (gridPos.y == gridPos2.y)
									{
										if (gridPos.y == 0)
										{ // South-Bottom
											calcRelPosVelFreeSlip(pos2, pos1, vel2, vel1, 7, &dpos, &dvelSlip, par);
											result -= interaction(j, index, dpos, dvelSlip, p, par);
										}
										if (gridPos.y == par->NYC - 1)
										{ // North-Bottom 
											calcRelPosVelFreeSlip(pos2, pos1, vel2, vel1, 8, &dpos, &dvelSlip, par);
											result -= interaction(j, index, dpos, dvelSlip, p, par);
										}
									}
									if (gridPos.x == gridPos2.x)
									{
										if (gridPos.x == 0)
										{ // West-Bottom
											calcRelPosVelFreeSlip(pos2, pos1, vel2, vel1, 11, &dpos, &dvelSlip, par);
											result -= interaction(j, index, dpos, dvelSlip, p, par);
										}
										if (gridPos.x == par->NXC - 1)
										{ // East-Bottom
											calcRelPosVelFreeSlip(pos2, pos1, vel2, vel1, 12, &dpos, &dvelSlip, par);
											result -= interaction(j, index, dpos, dvelSlip, p, par);
										}
									}
								}
								if ((gridPos.z == par->NZC - 1) && (gridPos2.z == par->NZC - 1))
								{
									if (gridPos.y == gridPos2.y)
									{
										if (gridPos.y == 0)
										{ // South-Top
											calcRelPosVelFreeSlip(pos2, pos1, vel2, vel1, 9, &dpos, &dvelSlip, par);
											result -= interaction(j, index, dpos, dvelSlip, p, par);
										}
										if (gridPos.y == par->NYC - 1)
										{ // North-Top
											calcRelPosVelFreeSlip(pos2, pos1, vel2, vel1, 10, &dpos, &dvelSlip, par);
											result -= interaction(j, index, dpos, dvelSlip, p, par);
										}
									}
									if (gridPos.x == gridPos2.x)
									{
										if (gridPos.x == 0)
										{ // West-Top
											calcRelPosVelFreeSlip(pos2, pos1, vel2, vel1, 13, &dpos, &dvelSlip, par);
											result -= interaction(j, index, dpos, dvelSlip, p, par);
										}
										if (gridPos.x == par->NXC - 1)
										{ // East-Top
											calcRelPosVelFreeSlip(pos2, pos1, vel2, vel1, 14, &dpos, &dvelSlip, par);
											result -= interaction(j, index, dpos, dvelSlip, p, par);
										}
									}
								}
								if (gridPos.x == gridPos2.x)
								{
									if ((gridPos.y == 0) && (gridPos2.y == 0))
									{
										if (gridPos.x == 0)
										{ // South-West
											calcRelPosVelFreeSlip(pos2, pos1, vel2, vel1, 15, &dpos, &dvelSlip, par);
											result -= interaction(j, index, dpos, dvelSlip, p, par);
										}
										if (gridPos.x == par->NXC - 1)
										{ // South-East
											calcRelPosVelFreeSlip(pos2, pos1, vel2, vel1, 16, &dpos, &dvelSlip, par);
											result -= interaction(j, index, dpos, dvelSlip, p, par);
										}
									}
									if ((gridPos.y == par->NYC - 1) && (gridPos2.y == par->NYC - 1))
									{
										if (gridPos.x == 0)
										{ // North-West
											calcRelPosVelFreeSlip(pos2, pos1, vel2, vel1, 17, &dpos, &dvelSlip, par);
											result -= interaction(j, index, dpos, dvelSlip, p, par);
										}
										if (gridPos.x == par->NXC - 1)
										{ // North-East
											calcRelPosVelFreeSlip(pos2, pos1, vel2, vel1, 18, &dpos, &dvelSlip, par);
											result -= interaction(j, index, dpos, dvelSlip, p, par);
										}
									}
								}


								// Corners
								if (gridHash0 == gridHash)
								{
									if ((gridPos.x == 0) && (gridPos.y == 0) && (gridPos.z == 0))
									{ //SWB
										calcRelPosVelFreeSlip(pos2, pos1, vel2, vel1, 19, &dpos, &dvelSlip, par);
										result -= interaction(j, index, dpos, dvelSlip, p, par);
									}
									if ((gridPos.x == 0) && (gridPos.y == 0) && (gridPos.z == par->NZC - 1))
									{ //WTS
										calcRelPosVelFreeSlip(pos2, pos1, vel2, vel1, 23, &dpos, &dvelSlip, par);
										result -= interaction(j, index, dpos, dvelSlip, p, par);
									}
									if ((gridPos.x == par->NXC - 1) && (gridPos.y == 0) && (gridPos.z == 0))
									{//SBE
										calcRelPosVelFreeSlip(pos2, pos1, vel2, vel1, 20, &dpos, &dvelSlip, par);
										result -= interaction(j, index, dpos, dvelSlip, p, par);
									}
									if ((gridPos.x == par->NXC - 1) && (gridPos.y == 0) && (gridPos.z == par->NZC - 1))
									{//ETS
										calcRelPosVelFreeSlip(pos2, pos1, vel2, vel1, 24, &dpos, &dvelSlip, par);
										result -= interaction(j, index, dpos, dvelSlip, p, par);
									}
									if ((gridPos.x == par->NXC - 1) && (gridPos.y == par->NYC - 1) && (gridPos.z == par->NZC - 1))
									{ //ETN
										calcRelPosVelFreeSlip(pos2, pos1, vel2, vel1, 26, &dpos, &dvelSlip, par);
										result -= interaction(j, index, dpos, dvelSlip, p, par);
									}
									if ((gridPos.x == par->NXC - 1) && (gridPos.y == par->NYC - 1) && (gridPos.z == 0))
									{//EBN
										calcRelPosVelFreeSlip(pos2, pos1, vel2, vel1, 22, &dpos, &dvelSlip, par);
										result -= interaction(j, index, dpos, dvelSlip, p, par);
									}
									if ((gridPos.x == 0) && (gridPos.y == par->NYC - 1) && (gridPos.z == 0))
									{ //WBN
										calcRelPosVelFreeSlip(pos2, pos1, vel2, vel1, 21, &dpos, &dvelSlip, par);
										result -= interaction(j, index, dpos, dvelSlip, p, par);
									}
									if ((gridPos.x == 0) && (gridPos.y == par->NYC - 1) && (gridPos.z == par->NZC - 1))
									{ //WTN
										calcRelPosVelFreeSlip(pos2, pos1, vel2, vel1, 25, &dpos, &dvelSlip, par);
										result -= interaction(j, index, dpos, dvelSlip, p, par);
									}
								}
							}
						}
					}
				}
			}

		}
	}
}


//// Periodic horizontal sides
if (par->T_BOUNDARY_PERIODICITY > 0)
{
	if ((gridPos.y == 0) || (gridPos.y == par->NYC - 1))
	{
		for (int x = -1; x <= 1; x++) {
			for (int z = -1; z <= 1; z++) {

				if (gridPos.y == 0) gridPos2.y = par->NYC - 1;
				if (gridPos.y == par->NYC - 1) gridPos2.y = 0;
				gridPos2.x = gridPos.x + x;
				gridPos2.z = gridPos.z + z;
				if ((gridPos2.x < 0) || (gridPos2.x > par->NXC - 1) || (gridPos2.z < 0) || (gridPos2.z > par->NZC - 1)) continue;

				gridHash = calcGridHash(gridPos2, par);
				startIndex = cellStart[gridHash];

				if (startIndex != 0xffffffff)
				{
					endIndex = cellEnd[gridHash];

					for (uint j = startIndex; j < endIndex; j++)
					{
						pos1 = p[index].pos;
						if (gridPos.y == 0) pos1.y += par->YCV;
						if (gridPos.y == par->NYC - 1) pos1.y -= par->YCV;
						pos2 = p[j].pos;
						vel1 = p[index].vel;
						vel2 = p[j].vel;

						calcRelPosVelFreeSlip(pos1, pos2, vel1, vel2, 0, &dpos, &dvelSlip, par);
						result += interaction(index, j, dpos, dvelSlip, p, par);
					}
				}
			}
		}
	}

	if ((gridPos.x == 0) || (gridPos.x == par->NXC - 1))
	{
		for (int y = -1; y <= 1; y++) {
			for (int z = -1; z <= 1; z++) {
				if (gridPos.x == 0) gridPos2.x = par->NXC - 1;
				if (gridPos.x == par->NXC - 1) gridPos2.x = 0;
				gridPos2.y = gridPos.y + y;
				gridPos2.z = gridPos.z + z;
				if ((gridPos2.y < 0) || (gridPos2.y > par->NYC - 1)) continue;
				if ((gridPos2.z < 0) || (gridPos2.z > par->NZC - 1)) continue;

				gridHash = calcGridHash(gridPos2, par);
				startIndex = cellStart[gridHash];

				if (startIndex != 0xffffffff)
				{
					endIndex = cellEnd[gridHash];

					for (uint j = startIndex; j < endIndex; j++)
					{
						pos1 = p[index].pos;
						if (gridPos.x == 0) pos1.x += par->XCV;
						if (gridPos.x == par->NXC - 1) pos1.x -= par->XCV;
						pos2 = p[j].pos;
						vel1 = p[index].vel;
						vel2 = p[j].vel;

						calcRelPosVelFreeSlip(pos1, pos2, vel1, vel2, 0, &dpos, &dvelSlip, par);
						result += interaction(index, j, dpos, dvelSlip, p, par);
					}
				}
			}
		}
	}

	// Edges
	if ((gridPos.x == 0) && (gridPos.y == 0) || (gridPos.x == par->NXC - 1) && (gridPos.y == 0)
		|| (gridPos.x == 0) && (gridPos.y == par->NYC - 1) || (gridPos.x == par->NXC - 1) && (gridPos.y == par->NYC - 1))
	{
		for (int z = -1; z <= 1; z++) {
			int3 gridPos2;

			if (gridPos.x == 0) gridPos2.x = par->NXC - 1;
			else gridPos2.x = 0;

			if (gridPos.y == 0) gridPos2.y = par->NYC - 1;
			else gridPos2.y = 0;

			gridPos2.z = gridPos.z + z;

			if ((gridPos2.z < 0) || (gridPos2.z > par->NZC - 1)) continue;

			uint gridHash = calcGridHash(gridPos2, par);
			uint startIndex = cellStart[gridHash];

			if (startIndex != 0xffffffff)
			{
				uint endIndex = cellEnd[gridHash];

				for (uint j = startIndex; j < endIndex; j++)
				{
					real3 dpos, dvel;
					real3 pos1 = p[index].pos;
					if (gridPos.x == 0) pos1.x += par->XCV;
					else pos1.x -= par->XCV;
					if (gridPos.y == 0) pos1.y += par->YCV;
					else pos1.y -= par->YCV;
					real3 pos2 = p[j].pos;
					real3 vel1 = p[index].vel;
					real3 vel2 = p[j].vel;

					calcRelPosVelFreeSlip(pos1, pos2, vel1, vel2, 0, &dpos, &dvelSlip, par);
					result += interaction(index, j, dpos, dvelSlip, p, par);
				}
			}
		}
	}
}

// Periodic vertical side
if ((par->T_BOUNDARY_PERIODICITY == 1) && ((gridPos.z == 0) || (gridPos.z == par->NZC - 1)))
{
	for (uint y = -1; y <= 1; y++) {
		for (uint x = -1; x <= 1; x++)
		{
			if (gridPos.z == 0) gridPos2.z = par->NZC - 1;
			if (gridPos.z == par->NZC - 1) gridPos2.z = 0;
			gridPos2.x = gridPos.x + x;
			gridPos2.y = gridPos.y + y;
			if ((gridPos2.y < 0) || (gridPos2.y > par->NYC - 1) || (gridPos2.x < 0) || (gridPos2.x > par->NXC - 1)) continue;

			gridHash = calcGridHash(gridPos2, par);
			startIndex = cellStart[gridHash];

			if (startIndex != 0xffffffff)
			{
				endIndex = cellEnd[gridHash];

				for (uint j = startIndex; j < endIndex; j++)
				{
					pos1 = p[index].pos;
					if (gridPos.z == 0) pos1.z += par->ZCV;
					if (gridPos.z == par->NZC - 1) pos1.z -= par->ZCV;
					pos2 = p[j].pos;
					vel1 = p[index].vel;
					vel2 = p[j].vel;

					calcRelPosVelFreeSlip(pos1, pos2, vel1, vel2, 0, &dpos, &dvelSlip, par);
					result += interaction(index, j, dpos, dvelSlip, p, par);
				}
			}
		}
	}
}

// Periodic (full) edges
if (par->T_BOUNDARY_PERIODICITY == 1)
{
	if ((gridPos.x == 0) && (gridPos.z == 0) || (gridPos.x == par->NXC - 1) && (gridPos.z == 0)
		|| (gridPos.x == 0) && (gridPos.z == par->NZC - 1) || (gridPos.x == par->NXC - 1) && (gridPos.z == par->NZC - 1))
	{
		for (int y = -1; y <= 1; y++) {
			int3 gridPos2;

			if (gridPos.x == 0) gridPos2.x = par->NXC - 1;
			else gridPos2.x = 0;

			if (gridPos.z == 0) gridPos2.z = par->NZC - 1;
			else gridPos2.z = 0;

			gridPos2.y = gridPos.y + y;

			if ((gridPos2.y < 0) || (gridPos2.y > par->NYC - 1)) continue;

			uint gridHash = calcGridHash(gridPos2, par);
			uint startIndex = cellStart[gridHash];

			if (startIndex != 0xffffffff)
			{
				uint endIndex = cellEnd[gridHash];

				for (uint j = startIndex; j < endIndex; j++)
				{
					real3 dpos, dvel;
					real3 pos1 = p[index].pos;
					if (gridPos.x == 0) pos1.x += par->XCV;
					else pos1.x -= par->XCV;
					if (gridPos.z == 0) pos1.z += par->ZCV;
					else pos1.z -= par->ZCV;
					real3 pos2 = p[j].pos;
					real3 vel1 = p[index].vel;
					real3 vel2 = p[j].vel;

					calcRelPosVelFreeSlip(pos1, pos2, vel1, vel2, 0, &dpos, &dvelSlip, par);
					result += interaction(index, j, dpos, dvelSlip, p, par);
				}
			}
		}
	}

	if ((gridPos.y == 0) && (gridPos.z == 0) || (gridPos.y == par->NYC - 1) && (gridPos.z == 0)
		|| (gridPos.y == 0) && (gridPos.z == par->NZC - 1) || (gridPos.y == par->NYC - 1) && (gridPos.z == par->NZC - 1))
	{
		for (int x = -1; x <= 1; x++) {
			int3 gridPos2;

			if (gridPos.y == 0) gridPos2.y = par->NYC - 1;
			else gridPos2.y = 0;

			if (gridPos.z == 0) gridPos2.z = par->NZC - 1;
			else gridPos2.z = 0;

			gridPos2.x = gridPos.x + x;

			if ((gridPos2.x < 0) || (gridPos2.x > par->NXC - 1)) continue;

			uint gridHash = calcGridHash(gridPos2, par);
			uint startIndex = cellStart[gridHash];

			if (startIndex != 0xffffffff)
			{
				uint endIndex = cellEnd[gridHash];

				for (uint j = startIndex; j < endIndex; j++)
				{
					real3 dpos, dvel;
					real3 pos1 = p[index].pos;
					if (gridPos.y == 0) pos1.y += par->YCV;
					else pos1.y -= par->YCV;
					if (gridPos.z == 0) pos1.z += par->ZCV;
					else pos1.z -= par->ZCV;
					real3 pos2 = p[j].pos;
					real3 vel1 = p[index].vel;
					real3 vel2 = p[j].vel;

					calcRelPosVelFreeSlip(pos1, pos2, vel1, vel2, 0, &dpos, &dvelSlip, par);
					result += interaction(index, j, dpos, dvelSlip, p, par);
				}
			}
		}
	}

	//Corners
	if ((gridPos.x == 0) && (gridPos.y == 0) && (gridPos.z == 0)
		|| (gridPos.x == 0) && (gridPos.y == 0) && (gridPos.z == par->NZC - 1)
		|| (gridPos.x == 0) && (gridPos.y == par->NYC - 1) && (gridPos.z == 0)
		|| (gridPos.x == 0) && (gridPos.y == par->NYC - 1) && (gridPos.z == par->NZC - 1)
		|| (gridPos.x == par->NXC - 1) && (gridPos.y == 0) && (gridPos.z == 0)
		|| (gridPos.x == par->NXC - 1) && (gridPos.y == 0) && (gridPos.z == par->NZC - 1)
		|| (gridPos.x == par->NXC - 1) && (gridPos.y == par->NYC - 1) && (gridPos.z == 0)
		|| ((gridPos.x == par->NXC - 1) && (gridPos.y == par->NYC - 1)) && (gridPos.z == par->NZC - 1))
	{
		int3 gridPos2;

		if (gridPos.x == 0) gridPos2.x = par->NXC - 1;
		else gridPos2.x = 0;

		if (gridPos.y == 0) gridPos2.y = par->NYC - 1;
		else gridPos2.y = 0;

		if (gridPos.z == 0) gridPos2.z = par->NZC - 1;
		else gridPos2.z = 0;

		uint gridHash = calcGridHash(gridPos2, par);
		uint startIndex = cellStart[gridHash];

		if (startIndex != 0xffffffff)
		{
			uint endIndex = cellEnd[gridHash];
			for (uint j = startIndex; j < endIndex; j++)
			{
				real3 dpos, dvel;
				real3 pos1 = p[index].pos;
				if (gridPos.x == 0) pos1.x += par->XCV;
				else pos1.x -= par->XCV;
				if (gridPos.y == 0) pos1.y += par->YCV;
				else pos1.y -= par->YCV;
				if (gridPos.z == 0) pos1.z += par->ZCV;
				else pos1.z -= par->ZCV;
				real3 pos2 = p[j].pos;
				real3 vel1 = p[index].vel;
				real3 vel2 = p[j].vel;

				calcRelPosVelFreeSlip(pos1, pos2, vel1, vel2, 0, &dpos, &dvelSlip, par);
				result += interaction(index, j, dpos, dvelSlip, p, par);
			}
		}
	}
}

// Periodic horizontal edges
if (par->T_BOUNDARY_PERIODICITY == 2)
{
	if ((gridPos.x == 0) && (gridPos.z == 0) || (gridPos.x == par->NXC - 1) && (gridPos.z == 0)
		|| (gridPos.x == 0) && (gridPos.z == par->NZC - 1) || (gridPos.x == par->NXC - 1) && (gridPos.z == par->NZC - 1))
	{
		for (int y = -1; y <= 1; y++) {
			int3 gridPos2;

			if (gridPos.x == 0) gridPos2.x = par->NXC - 1;
			else gridPos2.x = 0;

			gridPos2.z = gridPos.z;

			gridPos2.y = gridPos.y + y;

			if ((gridPos2.y < 0) || (gridPos2.y > par->NYC - 1)) continue;

			uint gridHash = calcGridHash(gridPos2, par);
			uint startIndex = cellStart[gridHash];

			if (startIndex != 0xffffffff)
			{
				uint endIndex = cellEnd[gridHash];

				for (uint j = startIndex; j < endIndex; j++)
				{
					real3 dpos, dvel;
					real3 pos1 = p[index].pos;
					if (gridPos.x == 0) pos1.x += par->XCV;
					else pos1.x -= par->XCV;
					real3 pos2 = p[j].pos;
					real3 vel1 = p[index].vel;
					real3 vel2 = p[j].vel;

					if (gridPos.z == 0)
					{
						calcRelPosVelFreeSlip(pos2, pos1, vel2, vel1, 5, &dpos, &dvelSlip, par);
						result -= interaction(j, index, dpos, dvelSlip, p, par);
					}
					if (gridPos.z == par->NZC - 1)
					{
						calcRelPosVelFreeSlip(pos2, pos1, vel2, vel1, 6, &dpos, &dvelSlip, par);
						result -= interaction(j, index, dpos, dvelSlip, p, par);
					}

				}
			}
		}
	}

	if ((gridPos.y == 0) && (gridPos.z == 0) || (gridPos.y == par->NYC - 1) && (gridPos.z == 0)
		|| (gridPos.y == 0) && (gridPos.z == par->NZC - 1) || (gridPos.y == par->NYC - 1) && (gridPos.z == par->NZC - 1))
	{
		for (int x = -1; x <= 1; x++) {
			int3 gridPos2;

			if (gridPos.y == 0) gridPos2.y = par->NYC - 1;
			else gridPos2.y = 0;

			gridPos2.z = gridPos.z;

			gridPos2.x = gridPos.x + x;

			if ((gridPos2.x < 0) || (gridPos2.x > par->NXC - 1)) continue;

			uint gridHash = calcGridHash(gridPos2, par);
			uint startIndex = cellStart[gridHash];

			if (startIndex != 0xffffffff)
			{
				uint endIndex = cellEnd[gridHash];

				for (uint j = startIndex; j < endIndex; j++)
				{
					real3 dpos, dvel;
					real3 pos1 = p[index].pos;
					if (gridPos.y == 0) pos1.y += par->YCV;
					else pos1.y -= par->YCV;
					real3 pos2 = p[j].pos;
					real3 vel1 = p[index].vel;
					real3 vel2 = p[j].vel;

					if (gridPos.z == 0)
					{
						calcRelPosVelFreeSlip(pos2, pos1, vel2, vel1, 5, &dpos, &dvelSlip, par);
						result -= interaction(j, index, dpos, dvelSlip, p, par);
					}
					if (gridPos.z == par->NZC - 1)
					{
						calcRelPosVelFreeSlip(pos2, pos1, vel2, vel1, 6, &dpos, &dvelSlip, par);
						result -= interaction(j, index, dpos, dvelSlip, p, par);
					}
				}
			}
		}
	}


	//Corners
	if ((gridPos.x == 0) && (gridPos.y == 0) && (gridPos.z == 0)
		|| (gridPos.x == 0) && (gridPos.y == 0) && (gridPos.z == par->NZC - 1)
		|| (gridPos.x == 0) && (gridPos.y == par->NYC - 1) && (gridPos.z == 0)
		|| (gridPos.x == 0) && (gridPos.y == par->NYC - 1) && (gridPos.z == par->NZC - 1)
		|| (gridPos.x == par->NXC - 1) && (gridPos.y == 0) && (gridPos.z == 0)
		|| (gridPos.x == par->NXC - 1) && (gridPos.y == 0) && (gridPos.z == par->NZC - 1)
		|| (gridPos.x == par->NXC - 1) && (gridPos.y == par->NYC - 1) && (gridPos.z == 0)
		|| ((gridPos.x == par->NXC - 1) && (gridPos.y == par->NYC - 1)) && (gridPos.z == par->NZC - 1))
	{
		int3 gridPos2;

		if (gridPos.x == 0) gridPos2.x = par->NXC - 1;
		else gridPos2.x = 0;

		if (gridPos.y == 0) gridPos2.y = par->NYC - 1;
		else gridPos2.y = 0;

		gridPos2.z = gridPos.z;

		uint gridHash = calcGridHash(gridPos2, par);
		uint startIndex = cellStart[gridHash];

		if (startIndex != 0xffffffff)
		{
			uint endIndex = cellEnd[gridHash];
			for (uint j = startIndex; j < endIndex; j++)
			{
				real3 dpos, dvel;
				real3 pos1 = p[index].pos;
				if (gridPos.x == 0) pos1.x += par->XCV;
				else pos1.x -= par->XCV;
				if (gridPos.y == 0) pos1.y += par->YCV;
				else pos1.y -= par->YCV;
				real3 pos2 = p[j].pos;
				real3 vel1 = p[index].vel;
				real3 vel2 = p[j].vel;

				if (gridPos.z == 0)
				{
					calcRelPosVelFreeSlip(pos2, pos1, vel2, vel1, 5, &dpos, &dvelSlip, par);
					result -= interaction(j, index, dpos, dvelSlip, p, par);
				}
				if (gridPos.z == par->NZC - 1)
				{
					calcRelPosVelFreeSlip(pos2, pos1, vel2, vel1, 6, &dpos, &dvelSlip, par);
					result -= interaction(j, index, dpos, dvelSlip, p, par);
				}
			}
		}
	}
}
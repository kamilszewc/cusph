
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

register real3 pos = MAKE_REAL3(p[index].pos.x, p[index].pos.y, p[index].pos.z);

register int3 gridPos = calcGridPos(pos, par);
register uint gridHash0 = calcGridHash(gridPos, par);

register int3 gridPos2;
register real3 dpos, dvel;
register real3 pos1, pos2;
register real3 vel1, vel2;

register uint gridHash;
register uint startIndex;
register uint endIndex;
register int x, y, z;
register uint j;

for (z = -1; z <= 1; z++) {
	for (y = -1; y <= 1; y++) {
		for (x = -1; x <= 1; x++) {

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

				for (j = startIndex; j < endIndex; j++) {
					pos1 = p[index].pos;
					pos2 = p[j].pos;
					vel1 = p[index].vel;
					vel2 = p[j].vel;

					calcRelPosVelNoSlip(pos1, pos2, vel1, vel2, 0, &dpos, &dvel, par);
					result += interaction(index, j, dpos, dvel, p, par);

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
									calcRelPosVelNoSlip(pos2, pos1, vel2, vel1, 5, &dpos, &dvel, par);
									result += interaction(j, index, dpos, dvel, p, par);
								}
								if (gridPos2.z == par->NZC - 1)
								{
									calcRelPosVelNoSlip(pos2, pos1, vel2, vel1, 6, &dpos, &dvel, par);
									result += interaction(j, index, dpos, dvel, p, par);
								}
							}

							if (par->T_BOUNDARY_PERIODICITY == 0)
							{
								// Walls
								if (gridPos.y == gridPos2.y)
								{
									if (gridPos.y == par->NYC - 1)
									{
										calcRelPosVelNoSlip(pos2, pos1, vel2, vel1, 1, &dpos, &dvel, par);
										result += interaction(j, index, dpos, dvel, p, par);
									}
									if (gridPos.y == 0)
									{
										calcRelPosVelNoSlip(pos2, pos1, vel2, vel1, 3, &dpos, &dvel, par);
										result += interaction(j, index, dpos, dvel, p, par);
									}
								}

								if (gridPos.x == gridPos2.x)
								{
									if (gridPos2.x == 0)
									{
										calcRelPosVelNoSlip(pos2, pos1, vel2, vel1, 4, &dpos, &dvel, par);
										result += interaction(j, index, dpos, dvel, p, par);
									}
									if (gridPos2.x == par->NXC - 1)
									{
										calcRelPosVelNoSlip(pos2, pos1, vel2, vel1, 2, &dpos, &dvel, par);
										result += interaction(j, index, dpos, dvel, p, par);
									}
								}

								//Krawedzie
								if ((gridPos.z == 0) && (gridPos2.z == 0))
								{
									if (gridPos.y == gridPos2.y)
									{
										if (gridPos.y == 0)
										{
											calcRelPosVelNoSlip(pos2, pos1, vel2, vel1, 7, &dpos, &dvel, par);
											result += interaction(j, index, dpos, dvel, p, par);
										}
										if (gridPos.y == par->NYC - 1)
										{
											calcRelPosVelNoSlip(pos2, pos1, vel2, vel1, 8, &dpos, &dvel, par);
											result += interaction(j, index, dpos, dvel, p, par);
										}
									}
									if (gridPos.x == gridPos2.x)
									{
										if (gridPos.x == 0)
										{
											calcRelPosVelNoSlip(pos2, pos1, vel2, vel1, 11, &dpos, &dvel, par);
											result += interaction(j, index, dpos, dvel, p, par);
										}
										if (gridPos.x == par->NXC - 1)
										{
											calcRelPosVelNoSlip(pos2, pos1, vel2, vel1, 12, &dpos, &dvel, par);
											result += interaction(j, index, dpos, dvel, p, par);
										}
									}
								}
								if ((gridPos.z == par->NZC - 1) && (gridPos2.z == par->NZC - 1))
								{
									if (gridPos.y == gridPos2.y)
									{
										if (gridPos.y == 0)
										{
											calcRelPosVelNoSlip(pos2, pos1, vel2, vel1, 9, &dpos, &dvel, par);
											result += interaction(j, index, dpos, dvel, p, par);
										}
										if (gridPos.y == par->NYC - 1)
										{
											calcRelPosVelNoSlip(pos2, pos1, vel2, vel1, 10, &dpos, &dvel, par);
											result += interaction(j, index, dpos, dvel, p, par);
										}
									}
									if (gridPos.x == gridPos2.x)
									{
										if (gridPos.x == 0)
										{
											calcRelPosVelNoSlip(pos2, pos1, vel2, vel1, 13, &dpos, &dvel, par);
											result += interaction(j, index, dpos, dvel, p, par);
										}
										if (gridPos.x == par->NXC - 1)
										{
											calcRelPosVelNoSlip(pos2, pos1, vel2, vel1, 14, &dpos, &dvel, par);
											result += interaction(j, index, dpos, dvel, p, par);
										}
									}
								}
								if (gridPos.x == gridPos2.x)
								{
									if ((gridPos.y == 0) && (gridPos2.y == 0))
									{
										if (gridPos.x == 0)
										{
											calcRelPosVelNoSlip(pos2, pos1, vel2, vel1, 15, &dpos, &dvel, par);
											result += interaction(j, index, dpos, dvel, p, par);
										}
										if (gridPos.x == par->NXC - 1)
										{
											calcRelPosVelNoSlip(pos2, pos1, vel2, vel1, 16, &dpos, &dvel, par);
											result += interaction(j, index, dpos, dvel, p, par);
										}
									}
									if ((gridPos.y == par->NYC - 1) && (gridPos2.y == par->NYC - 1))
									{
										if (gridPos.x == 0)
										{
											calcRelPosVelNoSlip(pos2, pos1, vel2, vel1, 17, &dpos, &dvel, par);
											result += interaction(j, index, dpos, dvel, p, par);
										}
										if (gridPos.x == par->NXC - 1)
										{
											calcRelPosVelNoSlip(pos2, pos1, vel2, vel1, 18, &dpos, &dvel, par);
											result += interaction(j, index, dpos, dvel, p, par);
										}
									}
								}

								// Narozniki
								if (gridHash0 == gridHash)
								{
									if ((gridPos.x == 0) && (gridPos.y == 0) && (gridPos.z == 0))
									{ //SWB
										calcRelPosVelNoSlip(pos2, pos1, vel2, vel1, 19, &dpos, &dvel, par);
										result += interaction(j, index, dpos, dvel, p, par);
									}
									if ((gridPos.x == 0) && (gridPos.y == 0) && (gridPos.z == par->NZC - 1))
									{ //WTS
										calcRelPosVelNoSlip(pos2, pos1, vel2, vel1, 23, &dpos, &dvel, par);
										result += interaction(j, index, dpos, dvel, p, par);
									}
									if ((gridPos.x == par->NXC - 1) && (gridPos.y == 0) && (gridPos.z == 0))
									{//SBE
										calcRelPosVelNoSlip(pos2, pos1, vel2, vel1, 20, &dpos, &dvel, par);
										result += interaction(j, index, dpos, dvel, p, par);
									}
									if ((gridPos.x == par->NXC - 1) && (gridPos.y == 0) && (gridPos.z == par->NZC - 1))
									{//ETS
										calcRelPosVelNoSlip(pos2, pos1, vel2, vel1, 24, &dpos, &dvel, par);
										result += interaction(j, index, dpos, dvel, p, par);
									}
									if ((gridPos.x == par->NXC - 1) && (gridPos.y == par->NYC - 1) && (gridPos.z == par->NZC - 1))
									{ //ETN
										calcRelPosVelNoSlip(pos2, pos1, vel2, vel1, 26, &dpos, &dvel, par);
										result += interaction(index, j, dpos, dvel, p, par);
									}
									if ((gridPos.x == par->NXC - 1) && (gridPos.y == par->NYC - 1) && (gridPos.z == 0))
									{//EBN
										calcRelPosVelNoSlip(pos2, pos1, vel2, vel1, 22, &dpos, &dvel, par);
										result += interaction(j, index, dpos, dvel, p, par);
									}
									if ((gridPos.x == 0) && (gridPos.y == par->NYC - 1) && (gridPos.z == 0))
									{ //WBN
										calcRelPosVelNoSlip(pos2, pos1, vel2, vel1, 21, &dpos, &dvel, par);
										result += interaction(j, index, dpos, dvel, p, par);
									}
									if ((gridPos.x == 0) && (gridPos.y == par->NYC - 1) && (gridPos.z == par->NZC - 1))
									{ //WTN
										calcRelPosVelNoSlip(pos2, pos1, vel2, vel1, 25, &dpos, &dvel, par);
										result += interaction(j, index, dpos, dvel, p, par);
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

//// Periodic horizontal
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

						calcRelPosVelNoSlip(pos1, pos2, vel1, vel2, 0, &dpos, &dvel, par);
						result += interaction(index, j, dpos, dvel, p, par);
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

						calcRelPosVelNoSlip(pos1, pos2, vel1, vel2, 0, &dpos, &dvel, par);
						result += interaction(index, j, dpos, dvel, p, par);
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

					calcRelPosVelNoSlip(pos1, pos2, vel1, vel2, 0, &dpos, &dvel, par);
					result += interaction(index, j, dpos, dvel, p, par);
				}
			}
		}
	}
}

// Periodic vertical side
if ((par->T_BOUNDARY_PERIODICITY == 1) && ((gridPos.z == 0) || (gridPos.z == par->NZC - 1)))
{
	for (y = -1; y <= 1; y++) {
		for (x = -1; x <= 1; x++)
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

				for (j = startIndex; j < endIndex; j++)
				{
					pos1 = p[index].pos;
					if (gridPos.z == 0) pos1.z += par->ZCV;
					if (gridPos.z == par->NZC - 1) pos1.z -= par->ZCV;
					pos2 = p[j].pos;
					vel1 = p[index].vel;
					vel2 = p[j].vel;

					calcRelPosVelNoSlip(pos1, pos2, vel1, vel2, 0, &dpos, &dvel, par);
					result += interaction(index, j, dpos, dvel, p, par);
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

					calcRelPosVelNoSlip(pos1, pos2, vel1, vel2, 0, &dpos, &dvel, par);
					result += interaction(index, j, dpos, dvel, p, par);
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

					calcRelPosVelNoSlip(pos1, pos2, vel1, vel2, 0, &dpos, &dvel, par);
					result += interaction(index, j, dpos, dvel, p, par);
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

				calcRelPosVelNoSlip(pos1, pos2, vel1, vel2, 0, &dpos, &dvel, par);
				result += interaction(index, j, dpos, dvel, p, par);
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
						calcRelPosVelNoSlip(pos2, pos1, vel2, vel1, 5, &dpos, &dvel, par);
						result += interaction(j, index, dpos, dvel, p, par);
					}
					if (gridPos.z == par->NZC - 1)
					{
						calcRelPosVelNoSlip(pos2, pos1, vel2, vel1, 6, &dpos, &dvel, par);
						result += interaction(j, index, dpos, dvel, p, par);
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
						calcRelPosVelNoSlip(pos2, pos1, vel2, vel1, 5, &dpos, &dvel, par);
						result += interaction(j, index, dpos, dvel, p, par);
					}
					if (gridPos.z == par->NZC - 1)
					{
						calcRelPosVelNoSlip(pos2, pos1, vel2, vel1, 6, &dpos, &dvel, par);
						result += interaction(j, index, dpos, dvel, p, par);
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
					calcRelPosVelNoSlip(pos2, pos1, vel2, vel1, 5, &dpos, &dvel, par);
					result += interaction(j, index, dpos, dvel, p, par);
				}
				if (gridPos.z == par->NZC - 1)
				{
					calcRelPosVelNoSlip(pos2, pos1, vel2, vel1, 6, &dpos, &dvel, par);
					result += interaction(j, index, dpos, dvel, p, par);
				}
			}
		}
	}
}

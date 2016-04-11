real2 pos = MAKE_REAL2(p[index].pos.x, p[index].pos.y);

int2 gridPos = calcGridPos(pos, par);
uint gridHash0 = calcGridHash(gridPos, par);

for (int y = -1; y <= 1; y++) {
	for (int x = -1; x <= 1; x++) {
		int2 gridPos2;
		gridPos2.x = gridPos.x + x;
		gridPos2.y = gridPos.y + y;
		if ((gridPos2.x < 0) || (gridPos2.x > par->NXC - 1) || (gridPos2.y < 0) || (gridPos2.y > par->NYC - 1)) continue;

		uint gridHash = calcGridHash(gridPos2, par);
		uint startIndex = cellStart[gridHash];

		if (startIndex != 0xffffffff) 
		{
			uint endIndex = cellEnd[gridHash];

			for (uint j = startIndex; j < endIndex; j++) 
			{
				real2 dpos, dvelSlip;
				real2 pos1 = p[index].pos;
				real2 pos2 = p[j].pos;
				real2 vel1 = p[index].vel;
				real2 vel2 = p[j].vel;
				calcRelPosVelFreeSlip(pos1, pos2, vel1, vel2, 0, &dpos, &dvelSlip, par);
				result += interaction(index, j, dpos, dvelSlip, p, par);

				if (((gridPos.x == 0) && (gridPos2.x == 0)) || ((gridPos.x == par->NXC - 1) && (gridPos2.x == par->NXC - 1)) ||
					((gridPos.y == 0) && (gridPos2.y == 0)) || ((gridPos.y == par->NYC - 1) && (gridPos2.y == par->NYC - 1)))
				{
					if (par->T_BOUNDARY_PERIODICITY != 1)
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
					}
					if (par->T_BOUNDARY_PERIODICITY == 0)
					{
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
						if ((gridPos.x == gridPos2.x) && (gridPos.y == gridPos.y))
						{
							if ((gridPos.x == 0) && (gridPos.y == 0))
							{
								calcRelPosVelFreeSlip(pos2, pos1, vel2, vel1, 7, &dpos, &dvelSlip, par);
								result -= interaction(j, index, dpos, dvelSlip, p, par);
							}
							if ((gridPos.x == par->NXC - 1) && (gridPos.y == 0))
							{
								calcRelPosVelFreeSlip(pos2, pos1, vel2, vel1, 6, &dpos, &dvelSlip, par);
								result -= interaction(j, index, dpos, dvelSlip, p, par);
							}
							if ((gridPos.x == 0) && (gridPos.y == par->NYC - 1))
							{
								calcRelPosVelFreeSlip(pos2, pos1, vel2, vel1, 8, &dpos, &dvelSlip, par);
								result -= interaction(j, index, dpos, dvelSlip, p, par);
							}
							if ((gridPos.x == par->NXC - 1) && (gridPos.y == par->NYC - 1))
							{
								calcRelPosVelFreeSlip(pos2, pos1, vel2, vel1, 5, &dpos, &dvelSlip, par);
								result -= interaction(j, index, dpos, dvelSlip, p, par);
							}
						}
					}

				}
			}

		}
	}
}

if ((par->T_BOUNDARY_PERIODICITY > 0) && ((gridPos.x == 0) || (gridPos.x == par->NXC - 1)))
{
	for (int y = -1; y <= 1; y++)
	{
		int2 gridPos2;
		if (gridPos.x == 0) gridPos2.x = par->NXC - 1;
		if (gridPos.x == par->NXC - 1) gridPos2.x = 0;
		gridPos2.y = gridPos.y + y;
		if ((gridPos2.y < 0) || (gridPos2.y > par->NYC - 1)) continue;

		uint gridHash = calcGridHash(gridPos2, par);
		uint startIndex = cellStart[gridHash];

		if (startIndex != 0xffffffff) 
		{
			uint endIndex = cellEnd[gridHash];

			for (uint j = startIndex; j < endIndex; j++)
			{
				real2 dpos, dvelSlip;
				real2 pos1 = p[index].pos;
				if (gridPos.x == 0) pos1.x += par->XCV;
				if (gridPos.x == par->NXC - 1) pos1.x -= par->XCV;
				real2 pos2 = p[j].pos;
				real2 vel1 = p[index].vel;
				real2 vel2 = p[j].vel;

				calcRelPosVelFreeSlip(pos1, pos2, vel1, vel2, 0, &dpos, &dvelSlip, par);
				result += interaction(index, j, dpos, dvelSlip, p, par);
			}

		}
	}
}


if ((par->T_BOUNDARY_PERIODICITY == 1) && ((gridPos.y == 0) || (gridPos.y == par->NYC - 1)))
{
	for (int x = -1; x <= 1; x++)
	{
		int2 gridPos2;
		if (gridPos.y == 0) gridPos2.y = par->NYC - 1;
		if (gridPos.y == par->NYC - 1) gridPos2.y = 0;
		gridPos2.x = gridPos.x + x;
		if ((gridPos2.x < 0) || (gridPos2.x > par->NXC - 1)) continue;

		uint gridHash = calcGridHash(gridPos2, par);
		uint startIndex = cellStart[gridHash];

		if (startIndex != 0xffffffff)
		{
			uint endIndex = cellEnd[gridHash];

			for (uint j = startIndex; j < endIndex; j++)
			{
				real2 dpos, dvelSlip;
				real2 pos1 = p[index].pos;
				if (gridPos.y == 0) pos1.y += par->YCV;
				if (gridPos.y == par->NYC - 1) pos1.y -= par->YCV;
				real2 pos2 = p[j].pos;
				real2 vel1 = p[index].vel;
				real2 vel2 = p[j].vel;

				calcRelPosVelFreeSlip(pos1, pos2, vel1, vel2, 0, &dpos, &dvelSlip, par);
				result += interaction(index, j, dpos, dvelSlip, p, par);
			}

		}
	}
}

if (par->T_BOUNDARY_PERIODICITY == 1)
{
	if (((gridPos.x == 0) && (gridPos.y == 0))
		|| (gridPos.x == 0) && (gridPos.y == par->NYC - 1)
		|| (gridPos.x == par->NXC - 1) && (gridPos.y == 0)
		|| (gridPos.x == par->NXC - 1) && (gridPos.y == par->NYC - 1))
	{
		int2 gridPos2;

		if (gridPos.x == 0) gridPos2.x = par->NXC - 1;
		else gridPos2.x = 0;

		if (gridPos.y == 0) gridPos2.y = par->NYC - 1;
		else gridPos2.y = 0;

		uint gridHash = calcGridHash(gridPos2, par);
		uint startIndex = cellStart[gridHash];

		if (startIndex != 0xffffffff)
		{
			uint endIndex = cellEnd[gridHash];
			for (uint j = startIndex; j < endIndex; j++)
			{
				real2 dpos, dvelSlip;
				real2 pos1 = p[index].pos;
				if (gridPos.x == 0) pos1.x += par->XCV;
				else pos1.x -= par->XCV;
				if (gridPos.y == 0) pos1.y += par->YCV;
				else pos1.y -= par->YCV;
				real2 pos2 = p[j].pos;
				real2 vel1 = p[index].vel;
				real2 vel2 = p[j].vel;

				calcRelPosVelFreeSlip(pos1, pos2, vel1, vel2, 0, &dpos, &dvelSlip, par);
				result += interaction(index, j, dpos, dvelSlip, p, par);
			}
		}
	}

}

if (par->T_BOUNDARY_PERIODICITY == 2) 
{
	if (((gridPos.x == 0) && (gridPos.y == 0))
		|| (gridPos.x == 0) && (gridPos.y == par->NYC - 1)
		|| (gridPos.x == par->NXC - 1) && (gridPos.y == 0)
		|| (gridPos.x == par->NXC - 1) && (gridPos.y == par->NYC - 1))
	{
		int2 gridPos2;

		if (gridPos.x == 0) gridPos2.x = par->NXC - 1;
		else gridPos2.x = 0;

		gridPos2.y = gridPos.y;

		uint gridHash = calcGridHash(gridPos2, par);
		uint startIndex = cellStart[gridHash];

		if (startIndex != 0xffffffff) 
		{
			uint endIndex = cellEnd[gridHash];
			for (uint j = startIndex; j < endIndex; j++) 
			{
				real2 dpos, dvelSlip;
				real2 pos1 = p[index].pos;
				if (gridPos.x == 0) pos1.x += par->XCV;
				else pos1.x -= par->XCV;
				real2 pos2 = p[j].pos;
				real2 vel1 = p[index].vel;
				real2 vel2 = p[j].vel;

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
		}
	}

}

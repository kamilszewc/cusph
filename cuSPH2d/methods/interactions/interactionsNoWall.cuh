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
				real2 dpos, dvel;
				real2 pos1 = p[index].pos;
				real2 pos2 = p[j].pos;
				real2 vel1 = p[index].vel;
				real2 vel2 = p[j].vel;
				calcRelPosVelNoSlip(pos1, pos2, vel1, vel2, 0, &dpos, &dvel, par);
				result += interaction(index, j, dpos, dvel, p, par);
			}

		}
	}
}

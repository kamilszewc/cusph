/*
* calcNumberOfCells.cu
*
*  Created on: 1-09-2015
*      Author: Kamil Szewc (kamil.szewc@gmail.com)
*/

#include "../../../sph.h"
#include "../../../hlp.h"
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <iostream>

struct compareSmoothingLength
{
	__host__ __device__
		bool operator()(const Particle& p1, const Particle & p2)
	{
		return p1.h < p2.h;
	}
};

static __global__ void setNumberOfCellsAndHAtGPU(Parameters *par, int nxc, int nyc, real h)
{
	par->NXC = nxc;
	par->NYC = nyc;
	par->NC = nxc * nyc;
	par->H = h;
	par->I_H = 1.0 / h;
}

void calcNumberOfCells(thrust::device_vector<Particle>& p, Parameters *par, Parameters *parHost)
{
	if (parHost->T_VARIABLE_H != 0)
	{
		thrust::device_vector<Particle>::iterator iterator = thrust::max_element(p.begin(), p.end(), compareSmoothingLength());
		thrust::host_vector<Particle> pMaxSmoothingLength(iterator, iterator + 1);

		real maxSmoothingLength = pMaxSmoothingLength[0].h;
		
		

		parHost->NXC = (int)(0.5 * (parHost->XCV + maxSmoothingLength) / maxSmoothingLength);
		parHost->NYC = (int)(0.5 * (parHost->YCV + maxSmoothingLength) / maxSmoothingLength);
		parHost->NC = parHost->NXC * parHost->NYC;
		parHost->H = maxSmoothingLength;
		parHost->I_H = 1.0 / maxSmoothingLength;

		setNumberOfCellsAndHAtGPU<<<1,1>>> (par, parHost->NXC, parHost->NYC, parHost->H);
	}
}

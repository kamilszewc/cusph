/*
* calcTimeStep.cu
*
*  Created on: 3-11-2014
*      Author: Kamil Szewc (kamil.szewc@gmail.com)
*/

#include "../../../sph.h"
#include "../../../hlp.h"
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <iostream>


struct compareVelocity
{
	__host__ __device__
	bool operator()(const Particle& p1, const Particle & p2)
	{
		real vel_1 = sqrt(pow2(p1.vel.x) + pow2(p1.vel.y)) + p1.s;
		real vel_2 = sqrt(pow2(p2.vel.x) + pow2(p2.vel.y)) + p2.s;

		return vel_1 < vel_2;
	}
};

struct compareViscosity
{
	__host__ __device__
	bool operator()(const Particle& p1, const Particle & p2)
	{
		return p1.mi < p2.mi;
	}
};

static __global__ void setTimeStepAtGPU(Parameters *par, real dt)
{
	par->DT = dt;
}


void calcTimeStep(thrust::device_vector<Particle>& p, Parameters *par, Parameters *parHost)
{
	if (parHost->T_TIME_STEP == 1)
	{
		thrust::device_vector<Particle>::iterator iterator = thrust::max_element(p.begin(), p.end(), compareVelocity());
		thrust::host_vector<Particle> pMaxVelocity(iterator, iterator+1);

		iterator = thrust::max_element(p.begin(), p.end(), compareViscosity());
		thrust::host_vector<Particle> pMaxViscosity(iterator, iterator+1);

		real maxVelocity = sqrt(pow2(pMaxVelocity[0].vel.x) + pow2(pMaxVelocity[0].vel.y)) + pMaxVelocity[0].c;
		real maxViscosity = pMaxViscosity[0].mi;

		real timeStepVelocity = 0.04 * 0.25 * parHost->H / maxVelocity;
		real timeStepViscosity = 0.04 * 0.125 * pow2(parHost->H) / maxViscosity;

		if (timeStepVelocity > timeStepViscosity)
		{
			parHost->DT = timeStepViscosity;
		}
		else
		{
			parHost->DT = timeStepVelocity;
		}

		//std::cout << timeStepVelocity << " " << timeStepViscosity << " " << parHost->DT << std::endl;
	}

	setTimeStepAtGPU<<<1,1>>>(par, parHost->DT);
}

void calcTimeStep(thrust::device_vector<Particle>& p, Parameters *par, Parameters *parHost, const real value)
{
	calcTimeStep(p, par, parHost);
	parHost->DT *= value;
	setTimeStepAtGPU<<<1,1>>>(par, parHost->DT);
}

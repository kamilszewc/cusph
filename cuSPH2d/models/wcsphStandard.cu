/*
*  wcsphColagrossiLandrini.cuh
*
*  Author: Kamil Szewc (kamil.szewc@gmail.com)
*  Modified on: 26-09-2014
*
*/
#include <thrust/device_vector.h>
#include "../sph.h"
#include "wcsphStandard/wcsphStandard.cuh"
#include "general/calcNumberOfCells/calcNumberOfCells.cuh"
#include "general/calcTimeStep/calcTimeStep.cuh"
#include "general/renormalizePressure/renormalizePressure.cuh"
#include "general/smoothingDensity/smoothingDensity.cuh"
#include "../methods/hashSortReorder.cuh"
#include "../methods/copyParticles.cuh"
#include "../errlog.h"

/**
 * The main loop of the WCSPH Standard model.
 */
void modelWcsphStandard(int NOB, int TPB,
	thrust::device_vector<Particle>& pVector,
	Particle *pSort,
	uint *gridParticleHash,
	uint *gridParticleIndex,
	uint *cellStart,
	uint *cellEnd,
	Parameters *par,
	Parameters *parHost,
	real time)
{
	STARTLOG("logs/models.log");

	static int step = 1;

	Particle* p = thrust::raw_pointer_cast(pVector.data());

	calcNumberOfCells(pVector, par, parHost);
	calcTimeStep(pVector, par, parHost);

	hashSortReorder(NOB, TPB, p, par, pSort, gridParticleHash, gridParticleIndex, cellStart, cellEnd, parHost->N);
	copyParticles << <NOB, TPB >> >(pSort, p, gridParticleIndex, true, par, parHost->N);

	if ( (parHost->T_SMOOTHING_DENSITY != 0) && (step%parHost->T_SMOOTHING_DENSITY == 0) )
	{
		smoothingDensity <<<NOB, TPB>>>(pSort, gridParticleIndex, cellStart, cellEnd, par);
		HANDLE_CUDA_KERNEL_RUNTIME_ERROR("smoothingDensity");
	}

	calcPressureWS <<<NOB, TPB>>>(pSort, par);
	HANDLE_CUDA_KERNEL_RUNTIME_ERROR("calcPressureWS");

	if (parHost->T_RENORMALIZE_PRESSURE == 1)
	{
		renormalizePressure(NOB, TPB, pSort, par, parHost->N);
	}

	calcInteractionWS <<<NOB, TPB>>>(pSort, gridParticleIndex, cellStart, cellEnd, par);
	HANDLE_CUDA_KERNEL_RUNTIME_ERROR("calcInteractionWS");

	if (parHost->T_XSPH != 0) 
	{
		calcXsphWS <<<NOB, TPB>>>(pSort, gridParticleIndex, cellStart, cellEnd, par);
		HANDLE_CUDA_KERNEL_RUNTIME_ERROR("calcXsphWS");
	}

	calcAdvectionWS <<<NOB, TPB>>>(pSort, par, step*parHost->DT);
	HANDLE_CUDA_KERNEL_RUNTIME_ERROR("calcAdvectionWS");

	copyParticles << <NOB, TPB >> >(p, pSort, gridParticleIndex, false, par, parHost->N);

	step++;
}

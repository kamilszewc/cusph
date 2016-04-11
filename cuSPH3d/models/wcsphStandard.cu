/*
*  wcsphStandard.cu
*
*  Author: Kamil Szewc (kamil.szewc@gmail.com)
*  Modified on: 27-09-2014
*
*/
#include <thrust/device_vector.h>
#include "../sph.h"
#include "../errlog.h"
#include "wcsphStandard.cuh"
#include "general/calcTimeStep/calcTimeStep.cuh"
#include "general/smoothingDensity/smoothingDensity.cuh"
#include "../methods/hashSortReorder.cuh"
#include "../methods/copyParticles.cuh"


void modelWcsphStandard(int NOB, int TPB,
	thrust::device_vector<Particle>& pVector,
	Particle *pSort,
	ParticleBasic *pOld,
	uint *gridParticleHash,
	uint *gridParticleIndex,
	uint *cellStart,
	uint *cellEnd,
	Parameters *par,
	Parameters *parHost,
	real time)
{
	STARTLOG("logs/models.log");

	Particle* p = thrust::raw_pointer_cast(pVector.data());

	calcTimeStep(pVector, par, parHost);

	hashSortReorder(NOB, TPB, p, par, pSort, gridParticleHash, gridParticleIndex, cellStart, cellEnd, parHost->N);
	copyParticles << <NOB, TPB >> >(pSort, p, gridParticleIndex, true, par);

	static int step = 1;
	if ((parHost->T_SMOOTHING_DENSITY != 0) && (step%parHost->T_SMOOTHING_DENSITY == 0))
	{
		smoothingDensity << <NOB, TPB >> >(pSort, gridParticleIndex, cellStart, cellEnd, par);
		HANDLE_CUDA_KERNEL_RUNTIME_ERROR("smoothingDensity");
	}
	step++;

	calcPressureWS <<<NOB, TPB>>>(pSort, par);
	HANDLE_CUDA_KERNEL_RUNTIME_ERROR("calcPressureWS");

	calcInteractionWS <<<NOB, TPB>>>(pSort, gridParticleIndex, cellStart, cellEnd, par);
	HANDLE_CUDA_KERNEL_RUNTIME_ERROR("calcInteractionWS");

	if (parHost->T_SURFACE_TENSION != 0) {
		// No surface tension model
	}

	if (parHost->T_XSPH != 0) {
		calcXsphWS <<<NOB, TPB>>>(pSort, gridParticleIndex, cellStart, cellEnd, par);
		HANDLE_CUDA_KERNEL_RUNTIME_ERROR("calcXsphWS");
	}

	calcAdvectionWS <<<NOB, TPB>>>(pSort, par);
	HANDLE_CUDA_KERNEL_RUNTIME_ERROR("calcAdvectionWS");

	copyParticles << <NOB, TPB >> >(p, pSort, gridParticleIndex, false, par);
}

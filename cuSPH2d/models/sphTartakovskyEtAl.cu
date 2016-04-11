/*
*  sphTartakovskyEtAl.cuh
*
*  Author: Kamil Szewc (kamil.szewc@gmail.com)
*  Modified on: 26-09-2014
*
*/

#include <iostream>
#include <thrust/device_vector.h>
#include "../sph.h"
#include "../hlp.h"
#include "sphTartakovskyEtAl/sphTartakovskyEtAl.cuh"
#include "general/calcNumberOfCells/calcNumberOfCells.cuh"
#include "general/calcTimeStep/calcTimeStep.cuh"
#include "general/renormalizePressure/renormalizePressure.cuh"
#include "../methods/hashSortReorder.cuh"
#include "../methods/copyParticles.cuh"
#include "../errlog.h"

void modelTartakovskyEtAl(int NOB, int TPB,
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

	Particle* p = thrust::raw_pointer_cast(pVector.data());

	calcNumberOfCells(pVector, par, parHost);
	calcTimeStep(pVector, par, parHost);

	static bool isConverted = false;
	if (isConverted == false)
	{
		std::cout << "Convertion..." << std::endl;
		calcDeformationSTEA <<<NOB, TPB>>>(p, par);
		HANDLE_CUDA_KERNEL_RUNTIME_ERROR("calcDeformationSTEA");
		isConverted = true;
	}

	hashSortReorder(NOB, TPB, p, par, pSort, gridParticleHash, gridParticleIndex, cellStart, cellEnd, parHost->N);
	copyParticles << <NOB, TPB >> >(pSort, p, gridParticleIndex, true, par, parHost->N);

	calcDensitySTEA <<<NOB, TPB>>>(pSort, gridParticleIndex, cellStart, cellEnd, par);
	HANDLE_CUDA_KERNEL_RUNTIME_ERROR("calcDensitySTEA");

	calcPressureSTEA <<<NOB, TPB>>>(pSort, par);
	HANDLE_CUDA_KERNEL_RUNTIME_ERROR("calcPressureSTEA");

	calcInteractionSTEA <<<NOB, TPB>>>(pSort, gridParticleIndex, cellStart, cellEnd, par);
	HANDLE_CUDA_KERNEL_RUNTIME_ERROR("calcInteractionSTEA");

	calcAdvectionSTEA <<<NOB, TPB>>>(pSort, par);
	HANDLE_CUDA_KERNEL_RUNTIME_ERROR("calcAdvectionSTEA");

	copyParticles << <NOB, TPB >> >(p, pSort, gridParticleIndex, false, par, parHost->N);
}

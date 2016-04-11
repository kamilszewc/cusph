/*
*  sphTartakovskyMeakin.cuh
*
*  Author: Kamil Szewc (kamil.szewc@gmail.com)
*  Modified on: 26-09-2014
*
*/

#include <thrust/device_vector.h>
#include "../sph.h"
#include "sphTartakovskyMeakin/sphTartakovskyMeakin.cuh"
#include "general/calcNumberOfCells/calcNumberOfCells.cuh"
#include "general/calcTimeStep/calcTimeStep.cuh"
#include "general/renormalizePressure/renormalizePressure.cuh"
#include "../methods/hashSortReorder.cuh"
#include "../methods/copyParticles.cuh"
#include <iostream>
#include "../errlog.h"


void modelSphTartakovskyMeakin(int NOB, int TPB,
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
	/*static bool isConverted = false;
	if (isConverted == false)
	{
		std::cout << "Convertion..." << std::endl;
		calcDeformationSTM << <NOB, TPB >> >(p, par);
		isConverted = true;
	}*/
	STARTLOG("logs/models.log");

	Particle* p = thrust::raw_pointer_cast(pVector.data());

	calcNumberOfCells(pVector, par, parHost);
	calcTimeStep(pVector, par, parHost);

	hashSortReorder(NOB, TPB, p, par, pSort, gridParticleHash, gridParticleIndex, cellStart, cellEnd, parHost->N);
	copyParticles << <NOB, TPB >> >(pSort, p, gridParticleIndex, true, par, parHost->N);

	//calcDensitySTM << <NOB, TPB >> >(p, pSort, gridParticleIndex, cellStart, cellEnd, par);

	calcDensitySTM <<<NOB, TPB>>>(pSort, par);
	HANDLE_CUDA_KERNEL_RUNTIME_ERROR("calcDensitySTM");

	calcPressureSTM <<<NOB, TPB>>>(pSort, par);
	HANDLE_CUDA_KERNEL_RUNTIME_ERROR("calcPressureSTM");

	calcInteractionSTM <<<NOB, TPB>>>(pSort, gridParticleIndex, cellStart, cellEnd, par);
	HANDLE_CUDA_KERNEL_RUNTIME_ERROR("calcInteractionSTM");

	calcAdvectionSTM <<<NOB, TPB>>>(pSort, par);
	HANDLE_CUDA_KERNEL_RUNTIME_ERROR("calcAdvectionSTM");

	copyParticles << <NOB, TPB >> >(p, pSort, gridParticleIndex, false, par, parHost->N);
}


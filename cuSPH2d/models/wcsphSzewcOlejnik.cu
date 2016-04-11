/*
*  wcsphSzewcOlejnik.cu
*
*  Author: Kamil Szewc (kamil.szewc@gmail.com)
*          Michal Olejnik
*  Modified on: 16-02-2015
*
*/
#include <thrust/device_vector.h>
#include "../sph.h"
#include "wcsphSzewcOlejnik/wcsphSzewcOlejnik.cuh"
#include "general/calcTimeStep/calcTimeStep.cuh"
#include "general/calcNumberOfCells/calcNumberOfCells.cuh"
#include "general/renormalizePressure/renormalizePressure.cuh"
#include "general/calcHydrostaticPressure/calcHydrostaticPressure.cuh"
#include "general/smoothHydrostaticPressure/smoothHydrostaticPressure.cuh"
#include "general/calcStrainTensor/calcStrainTensor.cuh"
#include "general/smoothingDensity/smoothingDensity.cuh"
#include "general/calcChezyViscosity/calcChezyViscosity.cuh"
#include "../methods/hashSortReorder.cuh"
#include "../methods/copyParticles.cuh"
#include "../errlog.h"

void modelWcsphSzewcOlejnik(int NOB, int TPB,
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

	hashSortReorder(NOB, TPB, p, par, pSort, gridParticleHash, gridParticleIndex, cellStart, cellEnd, parHost->N);
	copyParticles << <NOB, TPB >> >(pSort, p, gridParticleIndex, true, par, parHost->N);

	static int step = 1;
	if ( (parHost->T_SMOOTHING_DENSITY != 0) && (step%parHost->T_SMOOTHING_DENSITY == 0) )
	{
		smoothingDensity <<<NOB, TPB>>>(pSort, gridParticleIndex, cellStart, cellEnd, par);
		HANDLE_CUDA_KERNEL_RUNTIME_ERROR("smoothingDensity");
	}
	step++;


	if (parHost->T_HYDROSTATIC_PRESSURE > 0)
	{
		calcHydrostaticPressure <<<NOB, TPB>>>(pSort, gridParticleIndex, cellStart, cellEnd, par);
		HANDLE_CUDA_KERNEL_RUNTIME_ERROR("calcHydrostaticPressure");
	}
	if (parHost->T_HYDROSTATIC_PRESSURE == 2)
	{
		smoothHydrostaticPressure <<<NOB, TPB>>>(pSort, gridParticleIndex, cellStart, cellEnd, par);
		HANDLE_CUDA_KERNEL_RUNTIME_ERROR("smoothHydrostaticPressure");
	}

	calcPressureSO <<<NOB, TPB>>>(pSort, par);
	HANDLE_CUDA_KERNEL_RUNTIME_ERROR("calcPressureSO");


	if ( (parHost->T_STRAIN_TENSOR !=0) || (parHost->T_TURBULENCE != 0) || (parHost->T_SOIL != 0) )
	{
		calcStrainTensor <<<NOB, TPB>>>(pSort, gridParticleIndex, cellStart, cellEnd, par);
		HANDLE_CUDA_KERNEL_RUNTIME_ERROR("calcStrainTensor");
	}

	if (parHost->T_TURBULENCE != 0)
	{
		calcTurbulentViscositySO <<<NOB, TPB>>>(pSort, par);
		HANDLE_CUDA_KERNEL_RUNTIME_ERROR("calcTurbulentViscositySO");
	}

	if (parHost->T_SOIL != 0)
	{
		calcSoilViscositySO <<<NOB, TPB>>>(pSort, par);
		HANDLE_CUDA_KERNEL_RUNTIME_ERROR("calcSoilViscositySO");
	}
	if (parHost->T_SOIL == 2)
	{
		calcChezyViscosity <<<NOB, TPB>>>(pSort, gridParticleIndex, cellStart, cellEnd, par);
		HANDLE_CUDA_KERNEL_RUNTIME_ERROR("calcChezyViscosity");
	}

	calcInteractionSO <<<NOB, TPB>>>(pSort, gridParticleIndex, cellStart, cellEnd, par);
	HANDLE_CUDA_KERNEL_RUNTIME_ERROR("calcInteractionSO");

	if (parHost->T_SURFACE_TENSION != 0) {
		// No surface-tension
	}

	if (parHost->T_XSPH != 0) {
		// No XSPH
	}

	calcAdvectionSO <<<NOB, TPB>>>(pSort, par);
	HANDLE_CUDA_KERNEL_RUNTIME_ERROR("calcAdvectionSO");

	copyParticles << <NOB, TPB >> >(p, pSort, gridParticleIndex, false, par, parHost->N);
}

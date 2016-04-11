/*
*  wcsphColagrossiLandrini.cu
*
*  Author: Kamil Szewc (kamil.szewc@gmail.com)
*  Modified on: 27-09-2014
*
*/
#include <thrust/device_vector.h>
#include "../sph.h"
#include "../errlog.h"
#include "wcsphSzewcOlejnik.cuh"
#include "general/calcTimeStep/calcTimeStep.cuh"
#include "general/calcHydrostaticPressure/calcHydrostaticPressure.cuh"
#include "general/calcShearRate/calcShearRate.cuh"
#include "general/calcDispersedPhase/calcDispersedPhase.cuh"
#include "general/calcTurbulentViscosity/calcTurbulentViscosity.cuh"
#include "general/calcSingleSolidParticleAcceleration/calcSingleSolidParticleAcceleration.cuh"
#include "general/smoothingDensity/smoothingDensity.cuh"
#include "../methods/hashSortReorder.cuh"
#include "../methods/copyParticles.cuh"


void modelWcsphSzewcOlejnik(int NOB, int TPB,
	thrust::device_vector<Particle>& pVector,
	Particle *pSort,
	ParticleBasic *pOld,
	uint *gridParticleHash,
	uint *gridParticleIndex,
	uint *cellStart,
	uint *cellEnd,
	thrust::device_vector<ParticleDispersedPhase>& pDispersedPhaseVector,
	Parameters *par,
	Parameters *parHost,
	real time)
{
	STARTLOG("logs/models.log");

	Particle* p = thrust::raw_pointer_cast(pVector.data());
	ParticleDispersedPhase* pDispersedPhase = thrust::raw_pointer_cast(pDispersedPhaseVector.data());

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

	if (parHost->T_HYDROSTATIC_PRESSURE !=0)
	{
		calcHydrostaticPressure <<<NOB, TPB>>>(pSort, gridParticleIndex, cellStart, cellEnd, par);
		HANDLE_CUDA_KERNEL_RUNTIME_ERROR("calcHydrostaticPressure");
	}

	calcPressureSO <<<NOB, TPB>>>(pSort, par);
	HANDLE_CUDA_KERNEL_RUNTIME_ERROR("calcPressureSO");

	if ( (parHost->T_TURBULENCE != 0) || (parHost->T_SOIL != 0) )
	{
		calcShearRate <<<NOB, TPB>>>(pSort, gridParticleIndex, cellStart, cellEnd, par);
		HANDLE_CUDA_KERNEL_RUNTIME_ERROR("calcShearRate");
	}

	if (parHost->T_TURBULENCE != 0)
	{
		calcTurbulentViscosity <<<NOB, TPB>>>(pSort, par);
		HANDLE_CUDA_KERNEL_RUNTIME_ERROR("calcTurbulentViscosity");
	}

	if (parHost->T_SOIL != 0)
	{
		calcSoilViscositySO <<<NOB, TPB>>>(pSort, par);
		HANDLE_CUDA_KERNEL_RUNTIME_ERROR("calcSoilViscositySO");
	}

	calcInteractionSO << <NOB, TPB >> >(pSort, gridParticleIndex, cellStart, cellEnd, par);
	HANDLE_CUDA_KERNEL_RUNTIME_ERROR("calcInteractionSO");

	if (parHost->T_SURFACE_TENSION != 0) {
	}

	if (parHost->T_XSPH != 0) {
	}

	if (parHost->T_DISPERSED_PHASE > 0)
	{
		calcDispersedPhaseField << <(parHost->N_DISPERSED_PHASE + TPB - 1) / TPB, TPB >> >(pSort, gridParticleIndex, cellStart, cellEnd, pDispersedPhase, par);
		HANDLE_CUDA_KERNEL_RUNTIME_ERROR("calcDispersedPhaseField");

		calcDispersedPhaseAdvection << <(parHost->N_DISPERSED_PHASE + TPB - 1) / TPB, TPB >> >(pDispersedPhase, par);
		HANDLE_CUDA_KERNEL_RUNTIME_ERROR("calcDispersedPhaseAdvection");
	}

	copyParticles << <NOB, TPB >> >(p, pSort, gridParticleIndex, false, par);

	if (parHost->T_SOLID_PARTICLE != 0)
	{
		calcSingleSolidParticleAcceleration(NOB, TPB, pVector, par);
	}

	calcAdvectionSO <<<NOB, TPB>>>(p, par);
	HANDLE_CUDA_KERNEL_RUNTIME_ERROR("calcAdvectionSO");
}
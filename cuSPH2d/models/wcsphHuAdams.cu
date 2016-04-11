/*
*  wcsphHuAdams.cuh
*
*  Author: Kamil Szewc (kamil.szewc@gmail.com)
*  Modified on: 26-09-2014
*
*/
#include <thrust/device_vector.h>
#include <iostream>
#include <cuda_runtime.h>
#include "../sph.h"
#include "wcsphHuAdams/wcsphHuAdams.h"
#include "general/calcTimeStep/calcTimeStep.cuh"
#include "general/calcNumberOfCells/calcNumberOfCells.cuh"
#include "general/renormalizePressure/renormalizePressure.cuh"
#include "general/calcDispersedPhase/calcDispersedPhase.cuh"
#include "general/calcStrainTensor/calcStrainTensor.cuh"
#include "../methods/hashSortReorder.cuh"
#include "../methods/copyParticles.cuh"
#include "../errlog.h"


void modelWcsphHuAdams(int NOB, int TPB,
	thrust::device_vector<Particle>& pVector,
	Particle *pSort,
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

	static Particle* p = thrust::raw_pointer_cast(pVector.data());
	static ParticleDispersedPhase* pDispersedPhase = thrust::raw_pointer_cast(pDispersedPhaseVector.data());

	calcNumberOfCells(pVector, par, parHost);
	calcTimeStep(pVector, par, parHost);

	hashSortReorder(NOB, TPB, p, par, pSort, gridParticleHash, gridParticleIndex, cellStart, cellEnd, parHost->N);
	copyParticles <<<NOB, TPB>>>(pSort, p, gridParticleIndex, true, par, parHost->N);

	calcDensityWHA <<<NOB, TPB>>>(pSort, gridParticleIndex, cellStart, cellEnd, par);
	HANDLE_CUDA_KERNEL_RUNTIME_ERROR("calcDensityWHA");

	calcPressureWHA <<<NOB, TPB>>>(pSort, par);
	HANDLE_CUDA_KERNEL_RUNTIME_ERROR("calcPressureWHA");

	if (parHost->T_RENORMALIZE_PRESSURE > 0) {
		renormalizePressure(NOB, TPB, pSort, par, parHost->N);
	}

	if ( (parHost->T_STRAIN_TENSOR !=0) || (parHost->T_TURBULENCE != 0) || (parHost->T_SOIL != 0) )
	{
		calcStrainTensor <<<NOB, TPB>>>(pSort, gridParticleIndex, cellStart, cellEnd, par);
		HANDLE_CUDA_KERNEL_RUNTIME_ERROR("calcStrainTensor");
	}

	calcInteractionWHA <<<NOB, TPB>>>(pSort, gridParticleIndex, cellStart, cellEnd, par);
	HANDLE_CUDA_KERNEL_RUNTIME_ERROR("calcInteractionWHA");


	if (parHost->T_SURFACE_TENSION != 0)
	{
		// Normal vector calculation method
		if ( (parHost->T_NORMAL_VECTOR == 0) || (parHost->T_NORMAL_VECTOR == 1) )
		{
			calcSmoothedColorWHA <<<NOB, TPB>>>(pSort, gridParticleIndex, cellStart, cellEnd, par);
			HANDLE_CUDA_KERNEL_RUNTIME_ERROR("calcSmoothedColorWHA");

			calcNormalFromSmoothedColorWHA <<<NOB, TPB>>>(pSort, gridParticleIndex, cellStart, cellEnd, par);
			HANDLE_CUDA_KERNEL_RUNTIME_ERROR("calcNormalFromSmoothedColorWHA");
		}
		else if (parHost->T_NORMAL_VECTOR == 2)
		{
			calcNormalWHA <<<NOB, TPB>>>(pSort, gridParticleIndex, cellStart, cellEnd, par);
			HANDLE_CUDA_KERNEL_RUNTIME_ERROR("calcNormalWHA");
		}

		// Normal vector treshold calculation
		calcNormalThresholdWHA <<<NOB, TPB>>>(pSort, par);
		HANDLE_CUDA_KERNEL_RUNTIME_ERROR("calcNormalThresholdWHA");

		// Surface tension calculation
		if (parHost->T_SURFACE_TENSION == 1)
		{
			calcSurfaceTensionFromCurvatureWHA <<<NOB, TPB>>>(pSort, gridParticleIndex, cellStart, cellEnd, par);
			HANDLE_CUDA_KERNEL_RUNTIME_ERROR("calcSurfaceTensionFromCurvatureWHA");
		}
		else if (parHost->T_SURFACE_TENSION == 2)
		{
			calcCapillaryTensorWHA <<<NOB, TPB>>>(pSort, gridParticleIndex, cellStart, cellEnd, par);
			HANDLE_CUDA_KERNEL_RUNTIME_ERROR("calcCapillaryTensorWHA");

			calcSurfaceTensionFromCapillaryTensorWHA <<<NOB, TPB>>>(pSort, gridParticleIndex, cellStart, cellEnd, par);
			HANDLE_CUDA_KERNEL_RUNTIME_ERROR("calcSurfaceTensionFromCapillaryTensorWHA");
		}
	}

	if (parHost->T_SURFACTANTS != 0)
	{
		calcAreaWHA <<<NOB, TPB>>>(pSort, gridParticleIndex, cellStart, cellEnd, par);
		HANDLE_CUDA_KERNEL_RUNTIME_ERROR("calcAreaWHA");

		static bool isSurfactantMassSetUp = false;
		if (isSurfactantMassSetUp == false)
		{
			calcSurfactantsMassWHA <<<NOB, TPB>>>(pSort, gridParticleIndex, cellStart, cellEnd, par);
			HANDLE_CUDA_KERNEL_RUNTIME_ERROR("calcSurfactantsMassWHA");
			isSurfactantMassSetUp = true;
		}

		calcSurfactantsConcentrationWHA <<<NOB, TPB>>>(pSort, gridParticleIndex, cellStart, cellEnd, par);
		HANDLE_CUDA_KERNEL_RUNTIME_ERROR("calcSurfactantsConcentrationWHA");

		calcSurfactantsDiffusionBulkWHA <<<NOB, TPB>>>(pSort, gridParticleIndex, cellStart, cellEnd, par);
		HANDLE_CUDA_KERNEL_RUNTIME_ERROR("calcSurfactantsDiffusionBulkWHA");

		calcSurfactantsGradientWHA <<<NOB, TPB>>>(pSort, gridParticleIndex, cellStart, cellEnd, par);
		HANDLE_CUDA_KERNEL_RUNTIME_ERROR("calcSurfactantsGradientWHA");

		calcSurfactantsGradientNormWHA <<<NOB, TPB>>>(pSort, gridParticleIndex, cellStart, cellEnd, par);
		HANDLE_CUDA_KERNEL_RUNTIME_ERROR("calcSurfactantsGradientNormWHA");

		calcSurfactantsDiffusionWHA <<<NOB, TPB>>>(pSort, gridParticleIndex, cellStart, cellEnd, par);
		HANDLE_CUDA_KERNEL_RUNTIME_ERROR("calcSurfactantsDiffusionWHA");
	}

	if (parHost->T_DISPERSED_PHASE > 0)
	{
		calcDispersedPhaseField <<<(parHost->N_DISPERSED_PHASE + TPB - 1) / TPB, TPB>>>(pSort, gridParticleIndex, cellStart, cellEnd, pDispersedPhase, par);
		HANDLE_CUDA_KERNEL_RUNTIME_ERROR("calcDispersedPhaseField");

		calcDispersedPhaseAdvection <<<(parHost->N_DISPERSED_PHASE + TPB - 1) / TPB, TPB>>>(pDispersedPhase, par);
		HANDLE_CUDA_KERNEL_RUNTIME_ERROR("calcDispersedPhaseAdvection");
	}

	if (parHost->T_XSPH != 0)
	{
		calcXsphWHA <<<NOB, TPB>>>(pSort, gridParticleIndex, cellStart, cellEnd, par);
		HANDLE_CUDA_KERNEL_RUNTIME_ERROR("calcXsphWHA");
	}

	calcAdvectionWHA <<<NOB, TPB>>>(pSort, par);
	HANDLE_CUDA_KERNEL_RUNTIME_ERROR("calcAdvectionWHA");

	copyParticles <<<NOB, TPB>>>(p, pSort, gridParticleIndex, false, par, parHost->N);
}

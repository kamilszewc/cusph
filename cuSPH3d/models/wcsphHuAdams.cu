/*
*  wcsphHuAdams.cu
*
*  Author: Kamil Szewc (kamil.szewc@gmail.com)
*  Modified on: 27-09-2014
*
*/
#include <thrust/device_vector.h>
#include <iostream>
#include <cuda_runtime.h>
#include "../sph.h"
#include "../errlog.h"
#include "wcsphHuAdams.cuh"
#include "general/calcTimeStep/calcTimeStep.cuh"
#include "general/calcShearRate/calcShearRate.cuh"
#include "general/calcTurbulentViscosity/calcTurbulentViscosity.cuh"
#include "../methods/hashSortReorder.cuh"
#include "../methods/copyParticles.cuh"


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

	Particle* p = thrust::raw_pointer_cast(pVector.data());
	ParticleDispersedPhase* pDispersedPhase = thrust::raw_pointer_cast(pDispersedPhaseVector.data());

	calcTimeStep(pVector, par, parHost);

	hashSortReorder(NOB, TPB, p, par, pSort, gridParticleHash, gridParticleIndex, cellStart, cellEnd, parHost->N);
	copyParticles << <NOB, TPB >> >(pSort, p, gridParticleIndex, true, par);

	calcDensityWHA <<<NOB, TPB>>>(pSort, gridParticleIndex, cellStart, cellEnd, par);
	HANDLE_CUDA_KERNEL_RUNTIME_ERROR("calcDensityWHA");

	calcPressureWHA <<<NOB, TPB>>>(pSort, par);
	HANDLE_CUDA_KERNEL_RUNTIME_ERROR("calcPressureWHA");

	/*if (par.T_RENORMALIZE_PRESSURE != 0) {
		copyPressure<<<NOB,TPB>>>(helpArray, pSort, par);
		sortPressure(helpArray, par.N);
		renormalizePressure<<<NOB,TPB>>>(helpArray, pSort, par.N);
		//renormalizePressure( pSort, par, TPB );
		}*/

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

	calcInteractionWHA <<<NOB, TPB>>>(pSort, gridParticleIndex, cellStart, cellEnd, par);
	HANDLE_CUDA_KERNEL_RUNTIME_ERROR("calcInteractionWHA");

	if (parHost->T_SURFACE_TENSION != 0)
	{
		// Normal vector calculation method
		if ((parHost->T_NORMAL_VECTOR == 0) || (parHost->T_NORMAL_VECTOR == 1))
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
		calcNormalThresholdWHA << <NOB, TPB>>>(pSort, par);
		HANDLE_CUDA_KERNEL_RUNTIME_ERROR("calcNormalThresholdWHA");

		// Surface tension calculation
		if (parHost->T_SURFACE_TENSION == 1)
		{
			calcSurfaceTensionFromCurvatureWHA <<<NOB, TPB>>>(pSort, gridParticleIndex, cellStart, cellEnd, par);
			HANDLE_CUDA_KERNEL_RUNTIME_ERROR("calcSurfaceTensionFromCurvatureWHA");
		}
		else if (parHost->T_SURFACE_TENSION == 2)
		{
			//calcCapillaryTensorWHA << <NOB, TPB >> >(p, pSort, gridParticleIndex, cellStart, cellEnd, par);
			//HANDLE_CUDA_KERNEL_RUNTIME_ERROR("calcCapillaryTensorWHA");

			//calcSurfaceTensionFromCapillaryTensorWHA << <NOB, TPB >> >(p, pSort, gridParticleIndex, cellStart, cellEnd, par);
			//HANDLE_CUDA_KERNEL_RUNTIME_ERROR("calcSurfaceTensionFromCapillaryTensorWHA");
		}
	}

	if (parHost->T_XSPH != 0) {
		calcXsphWHA <<<NOB, TPB>>>(pSort, gridParticleIndex, cellStart, cellEnd, par);
		HANDLE_CUDA_KERNEL_RUNTIME_ERROR("calcXsphWHA");
	}

	if (parHost->T_DISPERSED_PHASE > 0)
	{
		calcDispersedPhaseField <<<(parHost->N_DISPERSED_PHASE + TPB - 1) / TPB, TPB>>>(pSort, gridParticleIndex, cellStart, cellEnd, pDispersedPhase, par);
		HANDLE_CUDA_KERNEL_RUNTIME_ERROR("calcDispersedPhaseField");

		calcDispersedPhaseAdvection <<<(parHost->N_DISPERSED_PHASE + TPB - 1) / TPB, TPB>>>(pDispersedPhase, par);
		HANDLE_CUDA_KERNEL_RUNTIME_ERROR("calcDispersedPhaseAdvection");
	}

	calcAdvectionWHA <<<NOB, TPB>>>(pSort, par);
	HANDLE_CUDA_KERNEL_RUNTIME_ERROR("calcAdvectionWHA");

	copyParticles << <NOB, TPB >> >(p, pSort, gridParticleIndex, false, par);
}

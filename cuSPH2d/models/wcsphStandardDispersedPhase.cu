/*
*  wcsphStandard.cuh
*
*  Author: Kamil Szewc (kamil.szewc@gmail.com)
*  Modified on: 26-09-2014
*
*/
#include <thrust/device_vector.h>
#include "../sph.h"
#include "wcsphStandardDispersedPhase/wcsphStandardDispersedPhase.h"
#include "general/calcTimeStep/calcTimeStep.cuh"
#include "general/calcNumberOfCells/calcNumberOfCells.cuh"
#include "general/calcDispersedPhase/calcDispersedPhase.cuh"
#include "general/renormalizePressure/renormalizePressure.cuh"
#include "general/smoothingDensity/smoothingDensity.cuh"
#include "general/calcStrainTensor/calcStrainTensor.cuh"
#include "general/calcChezyViscosity/calcChezyViscosity.cuh"
#include "general/dispersedPhaseFluidParticleManager/dispersedPhaseFluidParticleManager.h"
#include "../methods/hashSortReorder.cuh"
#include "../methods/copyParticles.cuh"
#include "../errlog.h"

struct is_position
{
	__host__ __device__
	bool operator()(const Particle p)
	{
		return p.pos.y < 0.5;
	}
};

void modelWcsphStandardDispersedPhase(int NOB, int TPB,
	thrust::device_vector<Particle>& pVector,
	Particle *pSort,
	uint *gridParticleHash,
	uint *gridParticleIndex,
	uint *cellStart,
	uint *cellEnd,
	thrust::device_vector<ParticleDispersedPhase>& pDispersedPhaseVector,
	thrust::device_vector<Particle>& pDispersedPhaseFluidVector,
	uint *cellStartPDPF,
	uint *cellEndPDPF,
	Parameters *par,
	Parameters *parHost,
	real time)
{
	STARTLOG("logs/models.log");
	static int step = 1;

	static Particle* p = thrust::raw_pointer_cast(pVector.data());
	static ParticleDispersedPhase* pDispersedPhase = thrust::raw_pointer_cast(pDispersedPhaseVector.data());

	Particle *pSortPDPF;
	uint *gridParticleHashPDPF;
	uint *gridParticleIndexPDPF;

	static real internalTime = 0.0;
	internalTime += parHost->DT;
	if (internalTime > 0.02) {
		DispersedPhaseFluidParticleManager dispersedPhaseFluidParticleManager(&pDispersedPhaseFluidVector, par, parHost);
		//dispersedPhaseFluidParticleManager.AddParticle(1.0, 0.8, 2500.0, 0.05);
		//dispersedPhaseFluidParticleManager.AddParticle(0.5+0.5*parHost->DR, 0.8, 1.5-0.5*parHost->DR, 0.8, 2500.0, 0.05);

		dispersedPhaseFluidParticleManager.DelParticle(1);

		internalTime = 0.0;
	}


	HANDLE_CUDA_ERROR(cudaMalloc((void**)&gridParticleHashPDPF, parHost->N_DISPERSED_PHASE_FLUID*sizeof(uint)));
	HANDLE_CUDA_ERROR(cudaMalloc((void**)&gridParticleIndexPDPF, parHost->N_DISPERSED_PHASE_FLUID*sizeof(uint)));
	HANDLE_CUDA_ERROR(cudaMalloc((void**)&pSortPDPF, parHost->N_DISPERSED_PHASE_FLUID*sizeof(Particle)));

	Particle* pPDPF = thrust::raw_pointer_cast(pDispersedPhaseFluidVector.data());

	calcNumberOfCells(pVector, par, parHost);
	calcTimeStep(pVector, par, parHost);

	hashSortReorder(NOB, TPB, p, par, pSort, gridParticleHash, gridParticleIndex, cellStart, cellEnd, parHost->N);
	copyParticles <<<NOB, TPB>>>(pSort, p, gridParticleIndex, true, par, parHost->N);

	if (parHost->T_DISPERSED_PHASE_FLUID != 0)
	{
		hashSortReorder( (parHost->N_DISPERSED_PHASE_FLUID + TPB - 1) / TPB, TPB, pPDPF, par, pSortPDPF, gridParticleHashPDPF, gridParticleIndexPDPF, cellStartPDPF, cellEndPDPF, parHost->N_DISPERSED_PHASE_FLUID);
		copyParticles <<< (parHost->N_DISPERSED_PHASE_FLUID + TPB - 1) / TPB, TPB>>>(pSortPDPF, pPDPF, gridParticleIndexPDPF, true, par, parHost->N_DISPERSED_PHASE_FLUID);
	}

	if ( (parHost->T_SMOOTHING_DENSITY != 0) && (step%parHost->T_SMOOTHING_DENSITY == 0) )
	{
		smoothingDensity <<<NOB, TPB>>>(pSort, gridParticleIndex, cellStart, cellEnd, par);
		HANDLE_CUDA_KERNEL_RUNTIME_ERROR("smoothingDensity");
	}

	calcPressureWSDP <<<NOB, TPB>>>(pSort, par);
	HANDLE_CUDA_KERNEL_RUNTIME_ERROR("calcPressureWS");

	if ( (parHost->T_STRAIN_TENSOR !=0) || (parHost->T_TURBULENCE != 0) || (parHost->T_SOIL != 0) )
	{
		calcStrainTensor <<<NOB, TPB>>>(pSort, gridParticleIndex, cellStart, cellEnd, par);
		HANDLE_CUDA_KERNEL_RUNTIME_ERROR("calcStrainTensor");
	}
	if (parHost->T_TURBULENCE != 0)
	{
		calcTurbulentViscosityWSDP <<<NOB, TPB>>>(pSort, par);
		HANDLE_CUDA_KERNEL_RUNTIME_ERROR("calcTurbulentViscositySO");
	}
	if (parHost->T_SOIL != 0)
	{
		calcSoilViscosityWSDP <<<NOB, TPB>>>(pSort, par);
		HANDLE_CUDA_KERNEL_RUNTIME_ERROR("calcSoilViscositySO");
	}
	if (parHost->T_SOIL == 2)
	{
		calcChezyViscosity <<<NOB, TPB>>>(pSort, gridParticleIndex, cellStart, cellEnd, par);
		HANDLE_CUDA_KERNEL_RUNTIME_ERROR("calcChezyViscosity");
	}

	if (parHost->T_DISPERSED_PHASE_FLUID != 0)
	{
		calcParticlesDensityAndVolumeFractionWSDP <<<(parHost->N_DISPERSED_PHASE_FLUID + TPB - 1) / TPB, TPB >>>(pSortPDPF, gridParticleIndexPDPF, cellStartPDPF, cellEndPDPF, par);
		HANDLE_CUDA_KERNEL_RUNTIME_ERROR("calcParticlesDensityAndVolumeFractionWS");

		calcInteractionFluidOnParticlesWSDP <<<(parHost->N_DISPERSED_PHASE_FLUID + TPB - 1) / TPB, TPB >>>(pSort, gridParticleIndex, cellStart, cellEnd, pSortPDPF, par);
		HANDLE_CUDA_KERNEL_RUNTIME_ERROR("calcInteractionFluidOnParticlesWS");

		calcInteractionParticlesOnFluidWSDP <<<NOB, TPB >>>(pSort, gridParticleIndexPDPF, cellStartPDPF, cellEndPDPF, pSortPDPF, par);
		HANDLE_CUDA_KERNEL_RUNTIME_ERROR("calcInteractionParticlesOnFluidWS");
	}

	calcInteractionWSDP <<<NOB, TPB>>>(pSort, gridParticleIndex, cellStart, cellEnd, par);
	HANDLE_CUDA_KERNEL_RUNTIME_ERROR("calcInteractionWS");

	calcAdvectionWSDP <<<NOB, TPB>>>(pSort, par, time);
	HANDLE_CUDA_KERNEL_RUNTIME_ERROR("calcAdvectionWS");

	copyParticles << <NOB, TPB >> >(p, pSort, gridParticleIndex, false, par, parHost->N);

	if (parHost->T_DISPERSED_PHASE_FLUID != 0)
	{
		calcAdvectionParticlesWSDP <<<(parHost->N_DISPERSED_PHASE_FLUID + TPB - 1) / TPB, TPB >>>(pSortPDPF, par);
		HANDLE_CUDA_KERNEL_RUNTIME_ERROR("calcAdvectionParticlesWS");

		copyParticles <<<(parHost->N_DISPERSED_PHASE_FLUID + TPB - 1) / TPB, TPB>>>(pPDPF, pSortPDPF, gridParticleIndexPDPF, false, par, parHost->N_DISPERSED_PHASE_FLUID);
	}

	cudaFree(gridParticleHashPDPF);
	cudaFree(gridParticleIndexPDPF);
	cudaFree(pSortPDPF);

	step++;
}

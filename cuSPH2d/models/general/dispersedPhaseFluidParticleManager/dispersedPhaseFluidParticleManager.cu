/*
 * dispersedParticleManager.cpp
 *
 *  Created on: 4 mar 2016
 *      Author: kamil
 */

#include "dispersedPhaseFluidParticleManager.h"

DispersedPhaseFluidParticleManager::DispersedPhaseFluidParticleManager(thrust::device_vector<Particle>* pDispersedPhaseFluidVector, Parameters* par, Parameters* parHost) {
	_pDispersedPhaseFluidVector = pDispersedPhaseFluidVector;
	_par = par;
	_parHost = parHost;
}

void DispersedPhaseFluidParticleManager::AddParticle(real x, real y, real di, real o) {
	std::vector<Particle> pDispersedPhaseFluidAdd = std::vector<Particle>();
	Particle particle;
	SetParticleProperties(particle, _parHost, _parHost->N_DISPERSED_PHASE_FLUID, x, y, di, o);
	pDispersedPhaseFluidAdd.push_back(particle);
	_parHost->N_DISPERSED_PHASE_FLUID += 1;

	thrust::device_vector<Particle> pDispersedPhaseFluidAddDev;
	pDispersedPhaseFluidAddDev = thrust::device_vector<Particle>(pDispersedPhaseFluidAdd);

	_pDispersedPhaseFluidVector->insert(_pDispersedPhaseFluidVector->end(), pDispersedPhaseFluidAddDev.begin(), pDispersedPhaseFluidAddDev.end());

	SetNumberOfParticlesOnGPU();
}

void DispersedPhaseFluidParticleManager::AddParticlesRow(real x0, real y0, real x1, real y1, real di, real o) {
	real dr = _parHost->H / _parHost->HDR;

	std::vector<Particle> pDispersedPhaseFluidAdd = std::vector<Particle>();

	for (real x = x0; x <= x1; x += dr) {
		for (real y = y0; y <= y1; y += dr) {
			Particle particle;
			SetParticleProperties(particle, _parHost, _parHost->N_DISPERSED_PHASE_FLUID, x, y, di, o);
			pDispersedPhaseFluidAdd.push_back(particle);
			_parHost->N_DISPERSED_PHASE_FLUID += 1;
		}
	}

	thrust::device_vector<Particle> pDispersedPhaseFluidAddDev;
	pDispersedPhaseFluidAddDev = thrust::device_vector<Particle>(pDispersedPhaseFluidAdd);

	_pDispersedPhaseFluidVector->insert(_pDispersedPhaseFluidVector->end(), pDispersedPhaseFluidAddDev.begin(), pDispersedPhaseFluidAddDev.end());

	SetNumberOfParticlesOnGPU();
}

struct IfIdCondition {
	IfIdCondition(int id) : id(id) {}

	__device__ __host__ bool operator()(const Particle p) {
		if (p.id == id) return true;
		else return false;
	};
private:
	int id;
};

void DispersedPhaseFluidParticleManager::DelParticle(int id) {
	_pDispersedPhaseFluidVector->erase(remove_if(_pDispersedPhaseFluidVector->begin(),
		_pDispersedPhaseFluidVector->end(),
		IfIdCondition(id)),
		_pDispersedPhaseFluidVector->end());
	_parHost->N_DISPERSED_PHASE_FLUID = _pDispersedPhaseFluidVector->size();
	SetNumberOfParticlesOnGPU();
}

struct Condition {
	__device__ __host__ bool operator()(const Particle p) {
		//Condition to remove particles
		return false;
	};
};

void DispersedPhaseFluidParticleManager::DelParticles() {
	_pDispersedPhaseFluidVector->erase(remove_if(_pDispersedPhaseFluidVector->begin(),
												 _pDispersedPhaseFluidVector->end(),
												 Condition()),
												 _pDispersedPhaseFluidVector->end());
	_parHost->N_DISPERSED_PHASE_FLUID = _pDispersedPhaseFluidVector->size();
	SetNumberOfParticlesOnGPU();
}




static __global__ void setNumberOfParticlesOnGPU(int n, Parameters* par)
{
	par->N_DISPERSED_PHASE_FLUID = n;
}

void DispersedPhaseFluidParticleManager::SetNumberOfParticlesOnGPU()
{
	setNumberOfParticlesOnGPU<<<1,1>>>(_parHost->N_DISPERSED_PHASE_FLUID, _par);
}

void DispersedPhaseFluidParticleManager::SetParticleProperties(Particle& p, Parameters* parHost, int id, real x, real y, real di, real o) {
	p.id = id; p.phaseId = 0;
	p.pos.x = x; p.rh_pos.x = 0.0;
	p.pos.y = y; p.rh_pos.y = 0.0;
	p.vel.x = 0.0; p.rh_vel.x = 0.0;
	p.vel.y = 0.0; p.rh_vel.y = 0.0;
	p.vel_s.x = 0.0; p.vel_s.y = 0.0;
	p.h = parHost->H; p.rh_h = 0.0;
	p.d = di * o; p.di = di; p.rh_d = 0.0; p.d_s = 0.0;
	p.m = parHost->XCV * parHost->YCV * p.d / (parHost->NX * parHost->NY);
	p.rh_m = 0.0;
	p.p = 0.0; p.ph = 0.0; p.phs = 0.0;
	p.nu = 0.01; p.mi = 0.01; p.nut = 0.0;
	p.str.x = 0.0; p.str.y = 0.0; p.str.z = 0.0; p.str.w = 0.0;
	p.tau.x = 0.0; p.tau.y = 0.0; p.tau.z = 0.0; p.tau.w = 0.0;
	p.phaseType = 0;
	p.gamma = 7.0; p.s = 10.0;
	p.b = p.s * p.s * p.di / p.gamma;
	p.o = o;// pow2(p[i].m / p[i].d);
	p.c = 0.0;
	p.n.x = 0.0; p.n.y = 0.0; p.n.z = 0.0;
	p.na = 0;
	p.cu = 0.0;
	p.st.x = 0.0; p.st.y = 0.0;
	p.cs = 0.0;
	p.cw = 0.0;
	p.ct.x = 0.0; p.ct.y = 0.0; p.ct.z = 0.0; p.ct.w = 0.0;

	// Surfactants
	p.mBulk = 0.0;
	p.cBulk = 0.0;
	p.dBulk = 0.0;
	p.mSurf = 0.0;
	p.cSurf = 0.0;
	p.dSurf = 0.0;
	p.cSurfGrad.x = 0.0;
	p.cSurfGrad.y = 0.0;
	p.a = 0.0;
}

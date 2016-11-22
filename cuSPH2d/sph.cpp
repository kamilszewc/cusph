/**
 * @file sph.cu
 * @brief The main cuSPH2d file.
 * @author Kamil Szewc (kamil.szewc@gmail.com)
 * @since 10-01-2015
 */

#if !defined(__SPH__)
#define __SPH__

#include <iostream>
#include <iomanip>
#include <sstream>
#include <ctime>
#include <vector>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include "sph.h"
#include "hlp.h"
#include "terminal.h"
#include "device.h"
#include "domain.h"
#include "errlog.h"
#include "license.h"

void modelWcsphStandard(int NOB, int TPB, thrust::device_vector<Particle>& pVector, Particle *dPSorted, uint *gridParticleHash, uint *gridParticleIndex, uint *cellStart, uint *cellEnd, Parameters *par, Parameters *parHost, real time);
void modelWcsphColagrossiLandrini(int NOB, int TPB, thrust::device_vector<Particle>& pVector, Particle *dPSorted, uint *gridParticleHash, uint *gridParticleIndex, uint *cellStart, uint *cellEnd, Parameters *par, Parameters *parHost, real time);
void modelWcsphHuAdams(int NOB, int TPB, thrust::device_vector<Particle>& pVector, Particle *dPSorted, uint *gridParticleHash, uint *gridParticleIndex, uint *cellStart, uint *cellEnd, thrust::device_vector<ParticleDispersedPhase>& pDispersedPhaseVector, Parameters *par, Parameters *parHost, real time);
void modelWcsphSzewcOlejnik(int NOB, int TPB, thrust::device_vector<Particle>& pVector, Particle *dPSorted, uint *gridParticleHash, uint *gridParticleIndex, uint *cellStart, uint *cellEnd, Parameters *par, Parameters *parHost, real time);
void modelSphTartakovskyMeakin(int NOB, int TPB, thrust::device_vector<Particle>& pVector, Particle *dPSorted, uint *gridParticleHash, uint *gridParticleIndex, uint *cellStart, uint *cellEnd, Parameters *par, Parameters *parHost, real time);
void modelTartakovskyEtAl(int NOB, int TPB, thrust::device_vector<Particle>& pVector, Particle *dPSorted, uint *gridParticleHash, uint *gridParticleIndex, uint *cellStart, uint *cellEnd, Parameters *par, Parameters *parHost, real time);
void modelWcsphStandardDispersedPhase(int NOB, int TPB, thrust::device_vector<Particle>& pVector, Particle *dPSorted, uint *gridParticleHash, uint *gridParticleIndex, uint *cellStart, uint *cellEnd, thrust::device_vector<ParticleDispersedPhase>& pDispersedPhaseVector, thrust::device_vector<Particle>& pDispersedPhaseFluidVector, uint* cellStartDevice, uint* cellEndDevice, Parameters *par, Parameters *parHost, real time);

int main(int argc, char *argv[])
{
	STARTLOG("logs/sph.log");
	License license;
	License::Version = "4.03.2016";
	LOG(License::GetShortInfo());
	Terminal terminal(argc, argv, license); // Listens for commends from terminal

	// Detecting and choosing GPUs
	LOG("Detecting GPUs...");
	LOG(Device::GetListOfDevices());
	int deviceId = 0;
	if (Device::GetNumberOfDevices() > 1)
	{
		std::ifstream file(".deviceId");
		if (file.is_open())
		{
			file >> deviceId;
			if (file.fail() || (deviceId < 0))
			{
				ERRLOG("Something wrong in config file .deviceId");
				exit(EXIT_FAILURE);
			}
			file.close();
		}
		else
		{
			ERRLOG("Cannot open .deviceId.\n To select GPU write its id into .deviceId config file.");
		}
	}

	std::stringstream ss; ss << deviceId;
	LOG( static_cast<std::string>("Choosing GPU no ") + ss.str() + "...");
	Device device(deviceId);

	// Choosing the case and the model
	int CASE = 102;
	int MODEL = 6;
	int OUT_PHASE = 0;
	FileFormat FILE_FORMAT = FileFormat::XML;

	// Declaring the CPU data variables
	Parameters* parHost;
	std::vector<Particle>* pHost;
	std::vector<ParticleDispersedPhase>* pDispersedPhaseHost;
	std::vector<Particle>* pDispersedPhaseFluidHost;

	// Data INPUT
	LOG("Setting up domain...");
	Domain *domain;
	double *t;
	switch (argc) {
	case 1:
		domain = new Domain(CASE, 2.4, 32);
		parHost = domain->GetParameters();
		pHost = domain->GetParticles();
		pDispersedPhaseHost = domain->GetParticlesDispersedPhase();
		pDispersedPhaseFluidHost = domain->GetParticlesDispersedPhaseFluid();
		t = domain->GetTime();
		parHost->T_MODEL = MODEL;
		domain->SetModel(MODEL);
		break;
	case 2:
		domain = new Domain(argv[1]);
		parHost = domain->GetParameters();
		pHost = domain->GetParticles();
		pDispersedPhaseHost = domain->GetParticlesDispersedPhase();
		pDispersedPhaseFluidHost = domain->GetParticlesDispersedPhaseFluid();
		t = domain->GetTime();
		MODEL = parHost->T_MODEL;
		break;
	case 3:
		domain = new Domain(argv[2]);
		parHost = domain->GetParameters();
		pHost = domain->GetParticles();
		pDispersedPhaseHost = domain->GetParticlesDispersedPhase();
		pDispersedPhaseFluidHost = domain->GetParticlesDispersedPhaseFluid();
		MODEL = parHost->T_MODEL;
		t = domain->GetTime();
		*t = atof(argv[1]);
		break;
	case 4:
		domain = new Domain(argv[2]);
		domain->SetOutputDirectory(argv[3]);
		parHost = domain->GetParameters();
		pHost = domain->GetParticles();
		pDispersedPhaseHost = domain->GetParticlesDispersedPhase();
		pDispersedPhaseFluidHost = domain->GetParticlesDispersedPhaseFluid();
		MODEL = parHost->T_MODEL;
		t = domain->GetTime();
		*t = atof(argv[1]);
		break;
	default:
		ERRLOG("-- Simulation did not start --");
		exit(EXIT_FAILURE);
	}
	
	LOG("Saving file...");
	domain->WriteToFile(FILE_FORMAT);

	double t0 = *t;

	// Declaring the GPU data variables
	thrust::device_vector<Particle> pDevice;
	thrust::device_vector<ParticleDispersedPhase> pDispersedPhaseDevice;
	thrust::device_vector<Particle> pDispersedPhaseFluidDevice;

	// Fluid particles
	Parameters *parDevice;
	uint *gridParticleHashDevice;
	uint *gridParticleIndexDevice;
	uint *cellStartDevice;
	uint *cellEndDevice;
	Particle *pSortDevice;

	// Dispersed (two-fluid approach) phase particles;
	uint *cellStartDevicePDPF;
	uint *cellEndDevicePDPF;

	// Allocating memory and transferring data to GPU
	LOG("Allocating memory and transferring data to GPU...");
	pDevice = thrust::device_vector<Particle>(*pHost);
	pDispersedPhaseDevice = thrust::device_vector<ParticleDispersedPhase>(*pDispersedPhaseHost);
	pDispersedPhaseFluidDevice = thrust::device_vector<Particle>(*pDispersedPhaseFluidHost);
	HANDLE_CUDA_ERROR( cudaMalloc((void**)&parDevice, sizeof(Parameters)) );
	HANDLE_CUDA_ERROR( cudaMemcpy(parDevice, parHost, sizeof(Parameters), cudaMemcpyHostToDevice) );
	HANDLE_CUDA_ERROR( cudaMalloc((void**)&gridParticleHashDevice, parHost->N*sizeof(uint)) );
	HANDLE_CUDA_ERROR( cudaMalloc((void**)&gridParticleIndexDevice, parHost->N*sizeof(uint)) );
	HANDLE_CUDA_ERROR( cudaMalloc((void**)&cellStartDevice, parHost->NC*sizeof(uint)) );
	HANDLE_CUDA_ERROR( cudaMemset(cellStartDevice, 0xffffffff, parHost->NC*sizeof(uint)) );
	HANDLE_CUDA_ERROR( cudaMalloc((void**)&cellEndDevice, parHost->NC*sizeof(uint)) );
	HANDLE_CUDA_ERROR( cudaMemset(cellEndDevice, 0xffffffff, parHost->NC*sizeof(uint)) );
	HANDLE_CUDA_ERROR( cudaMalloc((void**)&pSortDevice, parHost->N*sizeof(Particle)) );
	HANDLE_CUDA_ERROR(cudaMalloc((void**)&cellStartDevicePDPF, parHost->NC*sizeof(uint)));
	HANDLE_CUDA_ERROR(cudaMemset(cellStartDevicePDPF, 0xffffffff, parHost->NC*sizeof(uint)));
	HANDLE_CUDA_ERROR(cudaMalloc((void**)&cellEndDevicePDPF, parHost->NC*sizeof(uint)));
	HANDLE_CUDA_ERROR(cudaMemset(cellEndDevicePDPF, 0xffffffff, parHost->NC*sizeof(uint)));

	// Declaring threads per block and number of blocks
	int TPB = device.GetThreadsPerBlock(); // Threads Per Block
	int NOB = (parHost->N + TPB - 1) / TPB; // Number Of Blocks

	// Setting up the clock
	uint timestep = 0;
	time_t cpu_timer_0 = time(NULL);
	time_t cpu_timer_2 = time(NULL);
	double timeOfLastPrintOut = t0;

	// Starting the main loop
	LOG("Starting the main loop...");
	while (*t < parHost->END_TIME)
	{
		// Choosing the model
		switch (MODEL) {
		case 0:
			modelWcsphStandard(NOB, TPB, pDevice, pSortDevice, gridParticleHashDevice, gridParticleIndexDevice, cellStartDevice, cellEndDevice, parDevice, parHost, *t);
			break;
		case 1:
			modelWcsphColagrossiLandrini(NOB, TPB, pDevice, pSortDevice, gridParticleHashDevice, gridParticleIndexDevice, cellStartDevice, cellEndDevice, parDevice, parHost, *t);
			break;
		case 2:
			modelWcsphHuAdams(NOB, TPB, pDevice, pSortDevice, gridParticleHashDevice, gridParticleIndexDevice, cellStartDevice, cellEndDevice, pDispersedPhaseDevice, parDevice, parHost, *t);
			break;
		case 3:
			modelWcsphSzewcOlejnik(NOB, TPB, pDevice, pSortDevice, gridParticleHashDevice, gridParticleIndexDevice, cellStartDevice, cellEndDevice, parDevice, parHost, *t);
			break;
		case 4:
			modelSphTartakovskyMeakin(NOB, TPB, pDevice, pSortDevice, gridParticleHashDevice, gridParticleIndexDevice, cellStartDevice, cellEndDevice, parDevice, parHost, *t);
			break;
		case 5:
			modelTartakovskyEtAl(NOB, TPB, pDevice, pSortDevice, gridParticleHashDevice, gridParticleIndexDevice, cellStartDevice, cellEndDevice, parDevice, parHost, *t);
			break;
		case 6:
			modelWcsphStandardDispersedPhase(NOB, TPB, pDevice, pSortDevice, gridParticleHashDevice, gridParticleIndexDevice, cellStartDevice, cellEndDevice, pDispersedPhaseDevice, pDispersedPhaseFluidDevice, cellStartDevicePDPF, cellEndDevicePDPF, parDevice, parHost, *t);
			break;
		default:
			ERRLOG("Undefined model");
			exit(EXIT_FAILURE);
		}

		Terminal::ProgressBar((*t - timeOfLastPrintOut)/parHost->INTERVAL_TIME);

		// Data OUTPUT 
		if ( ((*t - timeOfLastPrintOut + 0.5*parHost->DT)/parHost->INTERVAL_TIME) >= 1.0 ) 
		{
			timeOfLastPrintOut = *t;

			try
			{
				LOG("\nTransfering data from GPU...");

				HANDLE_CUDA_ERROR( cudaMemcpy(parHost, parDevice, sizeof(Parameters), cudaMemcpyDeviceToHost) );

				thrust::copy(pDevice.begin(), pDevice.end(), pHost->begin());

				if (parHost->T_DISPERSED_PHASE != 0)
				{
					thrust::copy(pDispersedPhaseDevice.begin(), pDispersedPhaseDevice.end(), pDispersedPhaseHost->begin());
				}

				if (parHost->T_DISPERSED_PHASE_FLUID != 0)
				{
					pDispersedPhaseFluidHost->resize(parHost->N_DISPERSED_PHASE_FLUID);
					thrust::copy(pDispersedPhaseFluidDevice.begin(), pDispersedPhaseFluidDevice.end(), pDispersedPhaseFluidHost->begin());
				}

				double kinetic = domain->GetAndWriteKinetic();

				time_t cpu_timer_1 = time(NULL);
				double cpu_etime = difftime(cpu_timer_1, cpu_timer_0);
				double cpu_rtime = cpu_etime * ((parHost->END_TIME) - t0) / (*t - t0);
				double cpu_itime = difftime(cpu_timer_1, cpu_timer_2);

				{ // Printing info about the current timestep
					std::ostringstream stream;
					stream << "Time: " << std::fixed << *t << " ";
					stream << "Kinetic: " << std::fixed << kinetic;
					stream << std::endl;
					stream << "Elapsed time: " << static_cast<int>(cpu_etime) / 3600 << "h";
					stream << std::setfill('0') << std::setw(2) << (static_cast<int>(cpu_etime) % 3600) / 60 << "m";
					stream << std::setfill('0') << std::setw(2) << (static_cast<int>(cpu_etime) % 3600) % 60 << "s ";
					stream << "(" << static_cast<int>(100.0* *t / parHost->END_TIME) << "%) ";
					stream << "Total time: " << static_cast<int>(cpu_rtime) / 3600 << "h";
					stream << std::setfill('0') << std::setw(2) << (static_cast<int>(cpu_rtime) % 3600) / 60 << "m";
					stream << std::setfill('0') << std::setw(2) << (static_cast<int>(cpu_rtime) % 3600) % 60 << "s ";
					stream << std::endl;
					stream << "FPS: " << timestep / cpu_itime;
					LOG(stream.str().c_str());
					timestep = 0;
					cpu_timer_2 = cpu_timer_1;
				}

				LOG("Saving file...");
				domain->WriteToFile(FILE_FORMAT);
			}
			catch (std::exception& ex)
			{
				ERRLOG(ex.what());
				exit(EXIT_FAILURE);
			}
		}

		*t += static_cast<double>(parHost->DT);
		timestep += 1;
	}

	// Freeing GPU memory
	LOG("Freeing GPU memory...");
	cudaFree(parDevice);
	cudaFree(gridParticleHashDevice);
	cudaFree(gridParticleIndexDevice);
	cudaFree(cellStartDevice);
	cudaFree(cellEndDevice);
	cudaFree(pSortDevice);
	cudaFree(cellStartDevicePDPF);
	cudaFree(cellEndDevicePDPF);

	delete domain;

	return EXIT_SUCCESS;
}

#endif

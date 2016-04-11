/*
*  device.cpp
*
*  Author: Kamil Szewc (kamil.szewc@gmail.com)
*  Modified on: 14-12-2014
*
*/

#include <iostream>
#include <fstream>
#include <cstring>
#include <cstdlib>
#include <sstream>
#include "device.h"

Device::Device(int deviceId)
{
	if ( (deviceId < 0) || (deviceId > GetNumberOfDevices() - 1) ) 
	{
		std::cerr << "No deviceId=" << deviceId << std::endl;
		exit(EXIT_FAILURE);
	}
	cudaSetDevice(deviceId);
	cudaGetDeviceProperties(&_deviceProperties, deviceId);
}

int Device::GetMaxNumberOfThreadsPerBlock()
{
	return _deviceProperties.maxThreadsPerBlock;
}

int Device::GetSizeOfSharedMemoryPerBlock()
{
	return (int)_deviceProperties.sharedMemPerBlock;
}

const char* Device::GetName()
{
	return _deviceProperties.name;
}

int Device::GetClockRate()
{
	return _deviceProperties.clockRate;
}


int Device::GetThreadsPerBlock()
{
	int maxThreadsPerBlock = GetMaxNumberOfThreadsPerBlock();
	int sharedMemoryPerBlock = GetSizeOfSharedMemoryPerBlock();

	int threadsPerBlock = 0;

	std::ifstream file(".threadsPerBlock");

	if (file.is_open()) 
	{
		file >> threadsPerBlock;
		file.close();

		if (threadsPerBlock <= 0) 
		{
			std::cerr << "Something wrong in config file .threadsPerBlock" << std::endl;
			exit(EXIT_FAILURE);
		}
		if (threadsPerBlock > maxThreadsPerBlock) 
		{
			std::cerr << "Number of threads per block in .threadsPerBlock (" << threadsPerBlock << ") is higher than device limit (" << maxThreadsPerBlock << ")" << std::endl;
			exit(EXIT_FAILURE);
		}
	}
	else 
	{

		if (!strcmp(_deviceProperties.name, "GeForce GTX TITAN Black"))	threadsPerBlock = 128; // Not verified
		else if (!strcmp(_deviceProperties.name, "GeForce GTX 980Ti")) threadsPerBlock = 128; // Not verified
		else if (!strcmp(_deviceProperties.name, "GeForce GTX 980")) threadsPerBlock = 128; // Not verified
		else if (!strcmp(_deviceProperties.name, "GeForce GTX 970")) threadsPerBlock = 128; // Windows 10, VS 2013, CUDA SDK 7.0, -arch=sm_52
		else if (!strcmp(_deviceProperties.name, "GeForce GTX 780")) threadsPerBlock = 128; // Not verified
		else if (!strcmp(_deviceProperties.name, "GeForce GTX 770")) threadsPerBlock = 128; // Not verified
		else if (!strcmp(_deviceProperties.name, "GeForce GTX 650")) threadsPerBlock = 128; // Windows 8.1, VS2012, CUDA SDK 6.5, -arch=sm_30
		else if (!strcmp(_deviceProperties.name, "GeForce GTX 660Ti")) threadsPerBlock = 128; // Ubuntu Linux 12.04LTS, CUDA SDK 5.5, -arch=sm_30, sm_12
		else if (!strcmp(_deviceProperties.name, "GeForce GT 520")) threadsPerBlock = 64; // Not verified
		else if (!strcmp(_deviceProperties.name, "GeForce GT 240M")) threadsPerBlock = 64; // Debian Linux 6.0, CUDA SDK 3.0, -arch=sm_12
		else if (!strcmp(_deviceProperties.name, "GeForce GT 230M")) threadsPerBlock = 64; // Not verified
		else threadsPerBlock = 64;
	}

	if (sharedMemoryPerBlock < sizeof(uint)*(threadsPerBlock + 1)) 
	{
		std::cerr << "Device limit of shared memory per block (" << sharedMemoryPerBlock << "B) is lower than required (" << sizeof(uint)*(threadsPerBlock + 1) << ")" << std::endl;
		std::cerr << "Try to reduce number of threads per block in .threadsPerBlock." << std::endl;
		exit(EXIT_FAILURE);
	}

	return threadsPerBlock;
}

std::string Device::GetDescription()
{
	std::ostringstream stream;
	stream << _deviceProperties.name << " " << _deviceProperties.totalGlobalMem / 1048576 << " MB";
	return stream.str();
}

void Device::PrintDescription() 
{
	std::cout << GetDescription() << std::endl;
}

const char* Device::GetProperties() 
{
	std::ostringstream stream;
	stream << "  --- General information --- " << std::endl;
	stream << "Name: " << _deviceProperties.name << std::endl;
	stream << "Compute capability: " << _deviceProperties.major << "." << _deviceProperties.minor << std::endl;
	stream << "Clock rate: " << _deviceProperties.clockRate / 1000 << " MHz" << std::endl;
	stream << "Device copy overlap: ";
	if (_deviceProperties.deviceOverlap) 
		stream << "Enabled" << std::endl;
	else
		stream << "Disabled" << std::endl;
	stream << "Integrated card: ";
	if (_deviceProperties.integrated)
		stream << "Yes" << std::endl;
	else
		stream << "No" << std::endl;

	stream << "  --- Memory information ---" << std::endl;
	stream << "Total global memory: " << _deviceProperties.totalGlobalMem / 1048576 << " MB (" << _deviceProperties.totalGlobalMem << " B)" << std::endl;
	stream << "Total constant memory: " << _deviceProperties.totalConstMem << " B" << std::endl;
	stream << "Max mem pitch: " << _deviceProperties.memPitch / 1048576 << " MB (" << _deviceProperties.memPitch << " B)" << std::endl;
	stream << "Texture alignment: " << _deviceProperties.textureAlignment << std::endl;

	stream << "  --- MP information ---" << std::endl;
	stream << "Multiprocessor count: " << _deviceProperties.multiProcessorCount << std::endl;
	stream << "Shared memory per mp: " << _deviceProperties.sharedMemPerBlock << std::endl;
	stream << "Registers per mp: " << _deviceProperties.regsPerBlock << std::endl;
	stream << "Threads in warp: " << _deviceProperties.warpSize << std::endl;
	stream << "Max thread per block: " << _deviceProperties.maxThreadsPerBlock << std::endl;
	stream << "Max thread dimensions: (" << _deviceProperties.maxThreadsDim[0] << "," << _deviceProperties.maxThreadsDim[1] << "," << _deviceProperties.maxThreadsDim[2] << ")" << std::endl;
	stream << "Max grid dimensions: (" << _deviceProperties.maxGridSize[0] << "," << _deviceProperties.maxGridSize[1] << "," << _deviceProperties.maxGridSize[2] << ")" << std::endl;
	stream << "Concurrent kernels: ";
	if (_deviceProperties.concurrentKernels)
		stream << "Enabled" << std::endl;
	else
		stream << "Disabled" << std::endl;

	return stream.str().c_str();
}


int Device::GetNumberOfDevices()
{
	int count;
	cudaGetDeviceCount(&count);
	return count;
}

std::string Device::GetListOfDevices()
{
	std::ostringstream stream;

	for (int i = 0; i < GetNumberOfDevices(); i++) 
	{
		cudaDeviceProp deviceProperties;
		cudaGetDeviceProperties(&deviceProperties, i);
		stream << i << " ";
		stream << deviceProperties.name << " " << deviceProperties.totalGlobalMem / 1048576 << " MB";
		if (i != GetNumberOfDevices()-1) stream << std::endl;
	}
	std::string output = stream.str();

	return output;
}


/*
*  device.h
*
*  Author: Kamil Szewc (kamil.szewc@gmail.com)
*  Modified on: 27-09-2014
*
*/

#if !defined(__DEVICE_H__)
#define __DEVICE_H__

#include <string>
#include <cuda_runtime.h>

typedef unsigned int uint;

class Device {
private:
	cudaDeviceProp _deviceProperties;
public:
	Device(int deviceId);
	int GetMaxNumberOfThreadsPerBlock();
	int GetSizeOfSharedMemoryPerBlock();
	int GetThreadsPerBlock();
	const char* GetName();
	int GetClockRate();
	void PrintDescription();
	std::string GetDescription();
	const char* GetProperties();

	static int GetNumberOfDevices();
	static std::string GetListOfDevices();
};

#endif

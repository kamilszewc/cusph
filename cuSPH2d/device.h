/**
*  @file device.h
*  @author Kamil Szewc (kamil.szewc@gmail.com)
*  @since 26-09-2014
*/

#if !defined(__DEVICE_H__)
#define __DEVICE_H__

#include <string>
#include <cuda_runtime.h>

typedef unsigned int uint;

/**
 * Class for CUDA devices management.
 */
class Device {
private:
	cudaDeviceProp _deviceProperties;
public:
	/**
	 * @brief Constructor of Device
	 * @param[in] Device id (check GetListOfDevices)
	 */
	Device(int deviceId);
	int GetMaxNumberOfThreadsPerBlock(); ///< Returns maximal number of threads per block
	int GetSizeOfSharedMemoryPerBlock(); ///< Returns size of sharded memory per block
	int GetThreadsPerBlock(); ///< Returns number of threads per block
	const char* GetName(); ///< Returns GPU name
	int GetClockRate(); ///< Returns clock rate
	void PrintDescription(); ///< Print description to terminal
	std::string GetDescription(); ///< Returns description
	const char* GetProperties(); ///< Returns properties

	static int GetNumberOfDevices(); ///< Returns number of available CUDA devices
	static std::string GetListOfDevices(); ///< Returns list of CUDA devices
};

#endif

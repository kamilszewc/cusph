/*
*  @file errlog.h
*  @author Kamil Szewc (kamil.szewc@gmail.com)
*  @since 26-09-2014
*/

#if !defined(__ERRLOG_H__)
#define __ERRLOG_H__

#include <fstream>
#include <cuda_runtime.h>

/**
 * Class for management of reported errors.
 */
class ErrLog
{
private:
	const char* _filename; ///< Name of the log file
	std::ofstream _file;
public:
	ErrLog(const char*); ///< Constructor
	~ErrLog();           ///< Destructor

	void log(const char*); ///< Logs a message
	void log(std::string); ///< Logs a message
	void errLog(const char*, const char*, int); ///< Logs an error
	void errLog(std::string, const char*, int); ///< Logs an error
	void handleCudaError(cudaError_t, const char*, int); ///< Handle CUDA error
	void handleCudaKernelRuntimeError(const char* msg, const char*, int); ///< Handle CUDA kernel runtime error
	void handleCudaKernelRuntimeError(std::string, const char*, int); ///< Handle CUDA kernel runtime error
};

#define STARTLOG(filename) \
	static ErrLog errLog(filename);
//	static ErrLog *errLog = new ErrLog(filename);

#define LOG(msg) (errLog.log(msg))

#define ERRLOG(msg) (errLog.errLog(msg, __FILE__, __LINE__))

//(delete errLog)

#define HANDLE_CUDA_ERROR(err) (errLog.handleCudaError(err, __FILE__, __LINE__))

#define HANDLE_CUDA_KERNEL_RUNTIME_ERROR(msg) (errLog.handleCudaKernelRuntimeError(msg, __FILE__, __LINE__))

#endif

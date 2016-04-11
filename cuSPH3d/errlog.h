/*
*  errlog.h
*
*  Author: Kamil Szewc (kamil.szewc@gmail.com)
*  Modified on: 27-09-2014
*
*/

#if !defined(__ERRLOG_H__)
#define __ERRLOG_H__

#include <fstream>
#include <cuda_runtime.h>

class ErrLog
{
private:
	const char* _filename;
	std::ofstream _file;
public:
	ErrLog(const char*);
	~ErrLog();

	void log(const char*);
	void log(std::string);
	void errLog(const char*, const char*, int);
	void errLog(std::string, const char*, int);
	void handleCudaError(cudaError_t, const char*, int);
	void handleCudaKernelRuntimeError(const char* msg, const char*, int);
	void handleCudaKernelRuntimeError(std::string, const char*, int);
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

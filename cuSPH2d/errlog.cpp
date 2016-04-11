/*
*  errlog.cpp
*
*  Author: Kamil Szewc (kamil.szewc@gmail.com)
*  Modified on: 26-09-2014
*
*/

#include "errlog.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <ctime>

ErrLog::ErrLog(const char* filename) : _filename(filename)
{
	_file.open(_filename, std::ios::app);
	
	time_t t = time(0);
	struct tm* now = localtime(&t);
	
	if (_file.is_open())
	{
		_file << "CuSph2d log ";
		_file << (now->tm_year + 1900) << "-";
		_file << (now->tm_mon + 1) << "-";
		_file << (now->tm_mday) << " (";
		_file << (now->tm_hour) << ":";
		_file << (now->tm_min) << ":";
		_file << (now->tm_sec) << "):" << std::endl;
	}
}

ErrLog::~ErrLog()
{
	_file << std::endl;
	_file.close();
}

void ErrLog::log(const char* msg)
{
	std::cout << msg << std::endl;
	if (_file.is_open())
	{
		time_t t = time(0);
		struct tm* now = localtime(&t);
		_file << "(" << std::setfill('0') << std::setw(2) << (now->tm_hour);
		_file << ":" << std::setfill('0') << std::setw(2) << (now->tm_min);
		_file << ":" << std::setfill('0') << std::setw(2) << (now->tm_sec);
		_file << ") ";
		_file << msg << std::endl;
	}
}

void ErrLog::log(std::string msg)
{
	log(msg.c_str());
}

void ErrLog::errLog(const char* msg, const char* file, int line)
{
	std::cerr << "Error: " << msg << " in " << file << " in line " << line << std::endl;
	if (_file.is_open())
	{
		time_t t = time(0);
		struct tm* now = localtime(&t);
		_file << "(" << std::setfill('0') << std::setw(2) << (now->tm_hour);
		_file << ":" << std::setfill('0') << std::setw(2) << (now->tm_min);
		_file << ":" << std::setfill('0') << std::setw(2) << (now->tm_sec);
		_file << ") ";
		_file << "Error: " << msg << " in " << file << " in line " << line << std::endl;
	}
}

void ErrLog::errLog(std::string msg, const char* file, int line)
{
	errLog(msg.c_str(), file, line);
}

void ErrLog::handleCudaError(cudaError_t err, const char* file, int line)
{
	if (err != cudaSuccess)
	{
		errLog(cudaGetErrorString(err), file, line);
		exit(EXIT_FAILURE);
	}
}

void ErrLog::handleCudaKernelRuntimeError(const char* msg, const char* file, int line)
{
	cudaThreadSynchronize();
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		errLog(static_cast<std::string>("Kernel error: ") + cudaGetErrorString(err), file, line);
		exit(EXIT_FAILURE);
	}
}

void ErrLog::handleCudaKernelRuntimeError(std::string msg, const char* file, int line)
{
	handleCudaKernelRuntimeError(msg.c_str(), file, line);
}


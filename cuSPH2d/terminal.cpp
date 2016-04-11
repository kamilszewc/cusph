/*
*  terminal.cpp
*
*  Author: Kamil Szewc (kamil.szewc@gmail.com)
*  Modified on: 08-09-2015
*
*/
#include <iostream>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include "terminal.h"
#include "hlp.h"


Terminal::Terminal(int argc, char* argv[], License license)
{
	switch (argc) {
	case 1:
		break;
	case 2:
		if (!strcmp(argv[1], "-help") || !strcmp(argv[1], "--help") || !strcmp(argv[1], "-h")) { PrintHelp(); exit(EXIT_SUCCESS); }
		else if (!strcmp(argv[1], "-license")) { std::cout << License::GetLicense() << std::endl; exit(EXIT_SUCCESS); }
		else if (!strcmp(argv[1], "-short-license")) { std::cout << License::GetShortLicense() << std::endl; exit(EXIT_SUCCESS); }
		else if (!strcmp(argv[1], "-devices")) { std::cout << Device::GetListOfDevices() << std::endl; exit(EXIT_SUCCESS); }
		else if (!strcmp(argv[1], "-config-files")) { PrintConfigFiles(); exit(EXIT_SUCCESS); }
		else if (!strcmp(argv[1], "-authors")) { std::cout << License::GetAuthors() << std::endl; exit(EXIT_SUCCESS); }
		else { break; }
	case 3:
		break;
	case 4:
		break;
	default:
		std::cerr << "Incorrect number of arguments. ";
		std::cerr << "See 'cusph --help'." << std::endl;
		exit(EXIT_FAILURE);
	}
}

void Terminal::ProgressBar(real progress)
{
	int barWidth = 55;
	static int previousProgressPosition = 0;

	int progressPosition = static_cast<int>(barWidth * progress);

	if (progressPosition > previousProgressPosition)
	{
		std::cout << "[";
		for (int i = 0; i < barWidth; ++i)
		{
			if (i < progressPosition) std::cout << "=";
			else if (i == progressPosition) std::cout << ">";
			else std::cout << " ";
		}
		int percentage = static_cast<int>((progress + 0.005) * 100.0);
		std::cout << "] " << percentage << "%\r";
		std::cout.flush();

		previousProgressPosition = progressPosition;
		if ((percentage == 100) || (progressPosition == barWidth - 1))
		{
			previousProgressPosition = 0;
		}
	}
}

void Terminal::PrintConfigFiles()
{
	std::cout << ".deviceId - GPU id number" << std::endl;
	std::cout << ".threadsPerBlock - number of threads per block" << std::endl;
}

void Terminal::PrintHelp()
{
	std::cout << License::GetProgramFullName << std::endl;
	std::cout << License::GetShortLicense() << std::endl;
	std::cout << "Additional options:" << std::endl;
	std::cout << "  -license \t\t" << "Show license" << std::endl;
	std::cout << "  -short-license \t" << "Show shorten version of license" << std::endl;
	std::cout << "  -devices \t\t" << "Print list of CUDA devices" << std::endl;
	std::cout << "  -authors \t\t" << "Print list of authors" << std::endl;
	std::cout << "  -config-files \t" << "Configuration files" << std::endl;
}

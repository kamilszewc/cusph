/*
*  license.cpp
*
*  Author: Kamil Szewc (kamil.szewc@gmail.com)
*  Created on: 8-09-2015
*
*/
#include "license.h"

const char* License::Version;

License::License()
{
}

void License::SetVersion(char* version)
{
	Version = version;
}

const char* License::GetShortInfo()
{
	return "CuSPH2d by Kamil Szewc et al., all rights reserved.";
}

const char* License::GetShortLicense()
{
	return "Copyright (c) 2014-2016 Kamil Szewc (Institute of Fluid-Flow Machinery, PAS)\n"
		"All rights reserved.";
}

const char* License::GetLicense()
{
	return "Copyright (c) 2014-2016 Kamil Szewc (Institute of Fluid-Flow Machinery, PAS)\n"
		"All rights reserved.";
}

const char* License::GetProgramName()
{
	return "CuSph2d";
}

const char* License::GetProgramFullName()
{
	return "CuSph2d - CUDA Smoothed Particle Hydrodynamics solver";
}

const char* License::GetAuthors()
{
	return "Kamil Szewc, PhD; Michal Olejnik, MSc";
}


bool License::IsLicenseValid()
{
	return true;
}

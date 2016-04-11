/*
*  license.h
*
*  Author: Kamil Szewc (kamil.szewc@gmail.com)
*  Created on: 8-09-2015
*
*/

#if !defined(__LICENSE_H__)
#define __LICENSE_H__

/**
 * Class for license management.
 */
class License
{
public:
	static const char* Version;
	License(); ///< Constructor
	void SetVersion(char*); ///< Sets cuSPH2d version number
	static const char* GetShortInfo(); ///< Returns a short informations
	static const char* GetProgramFullName(); ///< Returns program full name
	static const char* GetProgramName(); ///< Returns program name
	static const char* GetLicense(); ///< Returns license
	static const char* GetShortLicense(); ///< Returns shorten verion of license
	static const char* GetAuthors(); ///< Returns authors
	bool IsLicenseValid(); ///< Check whether the license is valid
};


#endif

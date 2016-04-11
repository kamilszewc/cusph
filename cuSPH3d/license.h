/*
*  license.h
*
*  Author: Kamil Szewc (kamil.szewc@gmail.com)
*  Created on: 8-09-2015
*
*/

#if !defined(__LICENSE_H__)
#define __LICENSE_H__

class License
{
public:
	static char* Version;
	License();
	void SetVersion(char*);
	static const char* GetShortInfo();
	static const char* GetProgramFullName();
	static const char* GetProgramName();
	static const char* GetLicense();
	static const char* GetShortLicense();
	static const char* GetAuthors();
	bool IsLicenseValid();
};


#endif
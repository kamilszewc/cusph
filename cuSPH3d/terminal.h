/*
*  terminal.h
*
*  Author: Kamil Szewc (kamil.szewc@gmail.com)
*  Modified on: 27-09-2014
*
*/

#if !defined(__TERMINAL_H__)
#define __TERMINAL_H__

#include "device.h"
#include "license.h"
#include "hlp.h"

namespace terminal
{
	void terminal(int argc, char *argv[], License license);
	void progressBar(real progress);

}
#endif

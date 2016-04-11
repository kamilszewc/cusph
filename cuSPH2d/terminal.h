/**
 * @file terminal.h
*  @author Kamil Szewc (kamil.szewc@gmail.com)
*  @since 26-09-2014
*/

#if !defined(__TERMINAL_H__)
#define __TERMINAL_H__

#include "device.h"
#include "license.h"
#include "hlp.h"

class Terminal {
private:
	/**
	* @brief Prints help on terminal
	*/
	static void PrintHelp();

	/**
	* @brief Prints configuration files in terminal
	*/
	static void PrintConfigFiles();
public:
	/**
	 * @brief Terminal handling constructor
	 * @param[in] argc Number of arguments
	 * @param[in] argv Array of arguments
	 * @param[in] license License object
	 */
	Terminal(int argc, char *argv[], License license);

	/**
	 * @brief Prints a progress bar in terminal
	 * @param[in] progress A value that describes a level of completion (0-1)
	 */
	static void ProgressBar(real progress);
};
#endif

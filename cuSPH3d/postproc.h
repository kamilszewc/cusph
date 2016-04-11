/*
*  postproc.h
*
*  Author: Kamil Szewc (kamil.szewc@gmail.com)
*  Modified on: 4-01-2015
*
*/
#if !defined(__POSTPROC_H__)
#define __POSTPROC_H__

#include <iostream>
#include <vector>
#include "sph.h"
#include "domain.h"

class Postproc
{
private:
	int _NX, _NY, _NZ;
	Parameters* _par;
	std::vector<Particle>* _p;
	std::vector<std::vector<int>>* _particlesInCell;
public:
	Postproc(Domain*, int, int, int);
	Postproc(Domain*);

	void SetParticlesInCell();
};

#endif
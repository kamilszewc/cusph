/*
*  postproc.cpp
*
*  Author: Kamil Szewc (kamil.szewc@gmail.com)
*  Modified on: 4-01-2015
*
*/

#include <iostream>
#include "postproc.h"
#include "sph.h"
#include "hlp.h"

Postproc::Postproc(Domain* domain, int NX, int NY, int NZ) : _NX(NX), _NY(NY), _NZ(NZ)
{
	_par = domain->GetParameters();
	_p = domain->GetParticles();

	_particlesInCell = new std::vector<std::vector<int>>(_par->NC);
	SetParticlesInCell();
}

Postproc::Postproc(Domain* domain) 
{
}

void Postproc::SetParticlesInCell()
{

}
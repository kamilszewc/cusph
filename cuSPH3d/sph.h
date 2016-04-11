/*
*  sph.h
*
*  Author: Kamil Szewc (kamil.szewc@gmail.com)
*  Modified on: 27-09-2014
*
*/

#if !defined(__SPH_H__)
#define __SPH_H__

#include <cuda_runtime.h>
#include "hlp.h"

typedef unsigned int uint;

typedef struct Particle
{
	// Basic parameters
	int id;        // Particle index
	int phaseId;   // Phase index
	int phaseType;
	real3 pos;    // Position
	real3 rh_pos; // Warning: diff. implementations
	real3 vel;    // Velocity
	real3 rh_vel; // R.h.s. of N-S eq. (viscosity + grad. of pres.)
	real m;       // Mass
	real p, ph;       // Pressure
	real d;       // Density	
	real rh_d;    // R.h.s. of cont. eq.
	real di;      // Reference density / initial density
	real o;       // Warning: diff. implementations for diff. models
	real nu;      // Dynamical viscosity coef.
	real mi;      // Kinematic viscosity coef.

	// WCSPH parameters
	real gamma;   //
	real s;       // Sound speed
	real b;       // 

	real str;    // Strain-rate
	real nut;	  // Turb. viscosity

	// Surface Tension 
	real c;       //
	real cs;      //
	real cw;      //
	real4 n;      //
	real cu;      //
	real3 st;     //
	//real6 ct;     //
	int na;        //
} Particle;

typedef struct Parameters
{
	int T_MODEL;
	int NX, NY, NZ;     //
	int N;          //
	int NXC, NYC, NZC;   //
	int NC;         //
	real HDR;
	real XCV, YCV, ZCV; //
	real V_N, V_E, V_S, V_W, V_T, V_B; //
	real G_X, G_Y, G_Z;           //
	int T_BOUNDARY_PERIODICITY;
	real H, I_H, DH, DR;     //
	real KNORM;    //
	real GKNORM;   //
	real DT;       //
	real END_TIME;     //
	real INTERVAL_TIME; //
	int T_TIME_STEP; //
	int NOUT;       //
	int T_INTERFACE_CORRECTION;    //
	real INTERFACE_CORRECTION;    //
	int T_SURFACE_TENSION;// 
	real SURFACE_TENSION;  //
	int T_NORMAL_VECTOR; //
	int T_NORMAL_VECTOR_TRESHOLD; //
	int T_RENORMALIZE_PRESSURE; //
	int T_XSPH;     //
	real XSPH;     //
	int T_DISPERSED_PHASE; //
	int N_DISPERSED_PHASE; //
	int T_TURBULENCE;
	int T_SOIL;
	real SOIL_COHESION;          // Soil cohesion (only 1 phase)
	real SOIL_INTERNAL_ANGLE;    // Soil internal angle (only 1 phase)
	real SOIL_MINIMAL_VISCOSITY; // Soil minimal viscosity (only 1 phase)
	real SOIL_MAXIMAL_VISCOSITY; // Soil maximal viscosity (only 1 phase)
	int T_STRAIN_RATE;
	int T_HYDROSTATIC_PRESSURE;
	int T_SOLID_PARTICLE;
	int T_SMOOTHING_DENSITY;
} Parameters;

typedef struct ParticleBasic
{
	/* Used to store information only */
	real3 pos;     //
	real3 vel;     //
	real d;        //
} ParticleBasic;

typedef struct ParticleDispersedPhase
{
	real3 pos;
	real3 vel;
	real d;
	real dia;
	real3 velFl;
	real dFl;
	real miFl;
} ParticleDispersedPhase;

#endif

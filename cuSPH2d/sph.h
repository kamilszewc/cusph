/**
* @file sph.h
* @author Kamil Szewc (kamil.szewc@gmail.com)
* @since 26-09-2014
*/

#if !defined(__SPH_H__)
#define __SPH_H__

#include <cuda_runtime.h>
#include "hlp.h"

typedef unsigned int uint;


/**
 * Particle structure.
 */
typedef struct Particle 
{
	// Ids
	int id;        ///< Particle index
	int phaseId;   ///< Phase index

	// Newtonian/Non-newtonian fluid
	int phaseType; ///< Phase type (0-fluid, 1-soil)
                   ///< Present or not (0-no, 1-yes)

	// Basic parameters
	real2 pos;        ///< Position
	real2 rh_pos;     ///< Warning: diff. implementations
	real2 vel, vel_s; ///< Velocity
	real2 rh_vel;     ///< R.h.s. of N-S eq. (viscosity + grad. of pres.)
	real h, rh_h;     ///< Smoothing length
	real m, rh_m;     ///< Mass
	real p, ph, phs;  ///< Pressure
	real d, d_s;      ///< Density
	real rh_d;        ///< R.h.s. of the continuity equation
	real di;          ///< Reference density / initial density
	real o;           ///< Warning: diff. implementations for diff. models
	real nu;          ///< Dynamical viscosity coef.
	real mi;          ///< Kinematic viscosity coef.

	// Turbulence / soil rheological modelling
	real4 str;    ///< Strain-tensor
	real nut;	   ///< Turbulent viscosity / soil viscosity
	real4 tau;    ///< Stress-tensor
	
	// WCSPH parameters
	real gamma;   // Gamma parameter
	real s;       // Sound speed
	real b;       // Equation of state coefficient

	// Surface Tension 
	real c;       ///< Color function
	real cs;      ///< Smoothed color function (different implementations)
	real cw;      ///< Weighting function for s-t
	real3 n;      ///< Normal vector
	real cu;      ///< Curvature
	real2 st;     ///< Surface tension
	int na;       ///< Influence indicator
	real4 ct;     ///< Capilary tensor

	// Surfactants
	real mBulk;      ///< Surfactant mass in the bulk phase
	real cBulk;      ///< Surfactant concentration in the bulk phase
	real dBulk;      ///< Surfactant diffusion coefficient in the bulk phase
	real mSurf;      ///< Surfactant mass
	real cSurf;      ///< Surfactant concentration
	real dSurf;      ///< Surfactant diffusion coefficient
	real2 cSurfGrad; ///< Surfactant gradients
	real a;          ///< Area
} Particle;

/**
 * Parameters structure.
 */
typedef struct Parameters
{
	int CASE;		///< Additive if case from domain.cpp, negative for other cases
	int T_MODEL;	///< Type of model (0-standard, 1-Colagrossi & Landrini, 2-Hu & Adams, 3-Szewc & Olejnik)
	int NX, NY;     ///< Number of particles in given direction
	int N;          ///< Total number of particles
	int NXC, NYC;   ///< Number of cells (edge=2h) in given direction
	int NC;         ///< Total number of cells
	real HDR;       ///< h / dr
	real XCV, YCV;  ///< Domain sizes
	real V_N, V_E, V_S, V_W;   ///< Ghost particle boundart velocities
	real G_X, G_Y;             ///< Constant volume acceleration
	int T_BOUNDARY_PERIODICITY; ///< Type of boundary condition: (0-box, 1-full periodic, 2-left-right)
	real H, I_H, DH, DR;       ///< h, 1/h, 0.01 h, h / hdr
	real DT;            ///< Time step
	real END_TIME;      ///< End of simulation
	real INTERVAL_TIME; ///< Time between result outputs
	int T_TIME_STEP;     ///< Type of time step (0-fixed, 1-dynamic)
	int T_SURFACE_TENSION;        ///< Type of surface tension calculation (0-off, 1-CSF, 2-SSF)
	real SURFACE_TENSION;        ///< Value of surface tensions (only 2 phases)
	int T_NORMAL_VECTOR;          ///< Type of normal vector calculation (TODO)
	int T_NORMAL_VECTOR_TRESHOLD; ///< Normal vector threshold in s-t calculation (0-off, 1-on)
	int T_INTERFACE_CORRECTION;   ///< Interface correction (0-off, 1-on)
	real INTERFACE_CORRECTION;   ///< Interface correction coefficient
	int T_RENORMALIZE_PRESSURE;   ///< Pressure renormalization procedure (0-off, 1-on)
	int T_STRAIN_TENSOR; ///< Strain tensor output (0-off, 1-on)
	int T_XSPH;          ///< Xsph (0-off, 1-on)
	real XSPH;          ///< Xsph coefficient
	int T_DISPERSED_PHASE; ///< Dispersed phase (0-off, 1-on)
	int N_DISPERSED_PHASE; ///< Numer of dispresed phase particles
	int T_SURFACTANTS; ///< Surfactant calculation model (0-off, 1-on)
	int T_TURBULENCE;  ///< Turbulence model (0-off, 1-on)
	int T_SOIL;                   ///< Soil rheological model (0-off, 1-on, 2-on with Chezy model)
	real SOIL_COHESION;          ///< Soil cohesion (only 1 phase)
	real SOIL_INTERNAL_ANGLE;    ///< Soil internal angle (only 1 phase)
	real SOIL_MINIMAL_VISCOSITY; ///< Soil minimal viscosity (only 1 phase)
	real SOIL_MAXIMAL_VISCOSITY; ///< Soil maximal viscosity (only 1 phase)
	int T_HYDROSTATIC_PRESSURE;   ///< Hydrostatic pressure calculation (0-off, TODO)
	int T_SMOOTHING_DENSITY;      ///< Density smoothing for stabilization (0-off, int-time steps interval)
	int T_VARIABLE_H;             ///< Variable smoothing length (0-off, 1-on, 2-on TODO)
	int T_DISPERSED_PHASE_FLUID;  ///< Two-fluid model of dispresed phase (0-off, 1-on)
	int N_DISPERSED_PHASE_FLUID;  ///< Number of dispersed phase particles
} Parameters;

/**
 * Disperse phase particle structure.
 */
typedef struct ParticleDispersedPhase
{
	real2 pos;   ///< Position
	real2 vel;   ///< Velcoity
	real d;      ///< Density
	real dia;    ///< Diameter
	real2 velFl; ///< Velocity of fluid
	real dFl;    ///< Density of fluid
	real miFl;   ///< Kinematic viscosity of fluid
} ParticleDispersedPhase;


#endif

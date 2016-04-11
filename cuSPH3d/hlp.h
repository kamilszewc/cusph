/*
*  hlp.h
*
*  Author: Kamil Szewc (kamil.szewc@gmail.com)
*  Modified on: 27-09-2014
*
*/

#if !defined(__HLP_H__)
#define __HLP_H__

#include <math.h>

#define PRECISION_FLOAT

#if defined(PRECISION_FLOAT)
typedef float real;
typedef float2 real2;
typedef float3 real3;
typedef float4 real4;
#define MAKE_REAL2(x, y) make_float2(x, y)
#define MAKE_REAL3(x, y, z) make_float3(x, y, z)
#define MAKE_REAL4(x, y, z, w) make_float4(x, y, z, w)
#else
typedef double real;
typedef double2 real2;
typedef double3 real3;
typedef double4 real4;
#define MAKE_REAL2(x, y) make_double2(x, y)
#define MAKE_REAL3(x, y, z) make_double3(x, y, z)
#define MAKE_REAL4(x, y, z, w) make_double4(x, y, z, w)
#endif

// Mathematical constants
#ifndef M_E
#define M_E        2.71828182845904523536028747135      /* e */
#endif

#ifndef M_LOG2E
#define M_LOG2E    1.44269504088896340735992468100      /* log_2 (e) */
#endif

#ifndef M_LOG10E
#define M_LOG10E   0.43429448190325182765112891892      /* log_10 (e) */
#endif

#ifndef M_SQRT2
#define M_SQRT2    1.41421356237309504880168872421      /* sqrt(2) */
#endif

#ifndef M_SQRT1_2
#define M_SQRT1_2  0.70710678118654752440084436210      /* sqrt(1/2) */
#endif

#ifndef M_SQRT3
#define M_SQRT3    1.73205080756887729352744634151      /* sqrt(3) */
#endif

#ifndef M_PI
#define M_PI       3.14159265358979323846264338328      /* pi */
#endif

#ifndef M_PI_2
#define M_PI_2     1.57079632679489661923132169164      /* pi/2 */
#endif

#ifndef M_PI_4
#define M_PI_4     0.78539816339744830961566084582     /* pi/4 */
#endif

#ifndef M_SQRTPI
#define M_SQRTPI   1.77245385090551602729816748334      /* sqrt(pi) */
#endif

#ifndef M_2_SQRTPI
#define M_2_SQRTPI 1.12837916709551257389615890312      /* 2/sqrt(pi) */
#endif

#ifndef M_1_PI
#define M_1_PI     0.31830988618379067153776752675      /* 1/pi */
#endif

#ifndef M_2_PI
#define M_2_PI     0.63661977236758134307553505349      /* 2/pi */
#endif

#ifndef M_LN10
#define M_LN10     2.30258509299404568401799145468      /* ln(10) */
#endif

#ifndef M_LN2
#define M_LN2      0.69314718055994530941723212146      /* ln(2) */
#endif

#ifndef M_LNPI
#define M_LNPI     1.14472988584940017414342735135      /* ln(pi) */
#endif

#ifndef M_EULER
#define M_EULER    0.57721566490153286060651209008      /* Euler constant */
#endif

// Physical constants
#define C_BOLTZMANN 1.3806488E-23

// Zeros, infinities and not-a-numbers
#define ZERO 0.0000000001

// Small integer powers
#define pow2(x) ((x)*(x))
#define pow3(x) ((x)*(x)*(x))
#define pow4(x) (pow2(x)*pow2(x))
#define pow5(x) (pow2(x)*pow2(x)*(x))
#define pow6(x) (pow2(x)*pow2(x)*pow2(x))
#define pow7(x) (pow3(x)*pow3(x)*(x))
#define pow8(x) (pow2(pow2(x))*pow2(pow2(x)))
#define pow9(x) (pow3(x)*pow3(x)*pow3(x))

// Mathematical operations

#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define MIN(a,b) ((a) < (b) ? (a) : (b))

struct real6
{
	real x, y, z, u, v, w;
};

struct real8
{
	real s0, s1, s2, s3, s4, s5, s6, s7;
};

inline __host__ __device__ real6 MAKE_REAL6(real x, real y, real z, real u, real v, real w)
{
	real6 a;
	a.x = x; a.y = y; a.z = z; a.u = u; a.v = v; a.w = w;
	return a;
}

// int2
inline __host__ __device__ int2 operator-(int2 &a)
{
	return make_int2(-a.x, -a.y);
}

inline __host__ __device__ int2 operator+(int2 a, int2 b)
{
	return make_int2(a.x + b.x, a.y + b.y);
}

inline __host__ __device__ void operator+=(int2 &a, int2 b)
{
	a.x += b.x; a.y += b.y;
}

inline __host__ __device__ int2 operator-(int2 a, int2 b)
{
	return make_int2(a.x - b.x, a.y - b.y);
}

inline __host__ __device__ void operator-=(int2 &a, int2 b)
{
	a.x -= b.x; a.y -= b.y;
}

inline __host__ __device__ int2 operator*(int2 a, int2 b)
{
	return make_int2(a.x * b.x, a.y * b.y);
}

inline __host__ __device__ int2 operator*(int2 a, int s)
{
	return make_int2(a.x * s, a.y * s);
}

inline __host__ __device__ int2 operator*(int s, int2 a)
{
	return make_int2(a.x * s, a.y * s);
}

inline __host__ __device__ void operator*=(int2 &a, int s)
{
	a.x *= s; a.y *= s;
}

// real2
inline __host__ __device__ real2 operator-(real2 &a)
{
	return MAKE_REAL2(-a.x, -a.y);
}

inline __host__ __device__ real2 operator+(real2 a, real2 b)
{
	return MAKE_REAL2(a.x + b.x, a.y + b.y);
}

inline __host__ __device__ void operator+=(real2 &a, real2 b)
{
	a.x += b.x; a.y += b.y;
}

inline __host__ __device__ real2 operator-(real2 a, real2 b)
{
	return MAKE_REAL2(a.x - b.x, a.y - b.y);
}

inline __host__ __device__ void operator-=(real2 &a, real2 b)
{
	a.x -= b.x; a.y -= b.y;
}

inline __host__ __device__ real2 operator*(real2 a, real2 b)
{
	return MAKE_REAL2(a.x * b.x, a.y * b.y);
}

inline __host__ __device__ real2 operator*(real2 a, int s)
{
	return MAKE_REAL2(a.x * s, a.y * s);
}

inline __host__ __device__ real2 operator*(int s, real2 a)
{
	return MAKE_REAL2(a.x * s, a.y * s);
}

inline __host__ __device__ void operator*=(real2 &a, int s)
{
	a.x *= s; a.y *= s;
}

inline __host__ __device__ void operator*=(real2 &a, real s)
{
	a.x *= s; a.y *= s;
}

inline __host__ __device__ real2 operator/(real2 a, real2 b)
{
	return MAKE_REAL2(a.x / b.x, a.y / b.y);
}

inline __host__ __device__ real2 operator/(real2 a, real s)
{
	return a / s;
}

inline __host__ __device__ void operator/=(real2 &a, real s)
{
	real inv = 1.0 / s;
	a *= inv;
}

inline __host__ __device__ real dot(real2 a, real2 b)
{
	return a.x * b.x + a.y * b.y;
}

inline __host__ __device__ real length(real2 a)
{
	return sqrtf(dot(a,a));
}

inline __host__ __device__ real2 normalize(real2 a)
{
	return a / length(a);
}

// real3
inline __host__ __device__ real3 operator-(real3 &a)
{
	return MAKE_REAL3(-a.x, -a.y, -a.z);
}

inline __host__ __device__ real3 operator+(real3 a, real3 b)
{
	return MAKE_REAL3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline __host__ __device__ void operator+=(real3 &a, real3 b)
{
	a.x += b.x; a.y += b.y; a.z += b.z;
}

inline __host__ __device__ real3 operator-(real3 a, real3 b)
{
	return MAKE_REAL3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __host__ __device__ void operator-=(real3 &a, real3 b)
{
	a.x -= b.x; a.y -= b.y; a.z -= b.z;
}

inline __host__ __device__ real3 operator*(real3 a, real3 b)
{
	return MAKE_REAL3(a.x * b.x, a.y * b.y, a.z * b.z);
}

inline __host__ __device__ real3 operator*(real3 a, int s)
{
	return MAKE_REAL3(a.x * s, a.y * s, a.z * s);
}

inline __host__ __device__ real3 operator*(real3 a, real s)
{
	return MAKE_REAL3(a.x * s, a.y * s, a.z * s);
}

inline __host__ __device__ real3 operator*(int s, real3 a)
{
	return MAKE_REAL3(a.x * s, a.y * s, a.z * s);
}

inline __host__ __device__ void operator*=(real3 &a, int s)
{
	a.x *= s; a.y *= s; a.z *= s;
}

inline __host__ __device__ void operator*=(real3 &a, real s)
{
	a.x *= s; a.y *= s; a.z *= s;
}

inline __host__ __device__ real3 operator/(real3 a, real3 b)
{
	return MAKE_REAL3(a.x / b.x, a.y / b.y, a.z / b.z);
}

inline __host__ __device__ real3 operator/(real3 a, real s)
{
	return a / s;
}

inline __host__ __device__ void operator/=(real3 &a, real s)
{
	real inv = 1.0 / s;
	a *= inv;
}

inline __host__ __device__ real dot(real3 a, real3 b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __host__ __device__ real length(real3 a)
{
	return sqrtf(dot(a,a));
}

inline __host__ __device__ real3 normalize(real3 a)
{
	return a / length(a);
}

// real4
inline __host__ __device__ real4 operator-(real4 &a)
{
	return MAKE_REAL4(-a.x, -a.y, -a.z, -a.w);
}

inline __host__ __device__ real4 operator+(real4 a, real4 b)
{
	return MAKE_REAL4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

inline __host__ __device__ void operator+=(real4 &a, real4 b)
{
	a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
}

inline __host__ __device__ real4 operator-(real4 a, real4 b)
{
	return MAKE_REAL4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

inline __host__ __device__ void operator-=(real4 &a, real4 b)
{
	a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w;
}

inline __host__ __device__ real4 operator*(real4 a, real4 b)
{
	return MAKE_REAL4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}

inline __host__ __device__ real4 operator*(real4 a, int s)
{
	return MAKE_REAL4(a.x * s, a.y * s, a.z * s, a.w * s);
}

inline __host__ __device__ real4 operator*(int s, real4 a)
{
	return MAKE_REAL4(a.x * s, a.y * s, a.z * s, a.w * s);
}

inline __host__ __device__ void operator*=(real4 &a, int s)
{
	a.x *= s; a.y *= s; a.z *= s; a.w *= s;
}

inline __host__ __device__ void operator*=(real4 &a, real s)
{
	a.x *= s; a.y *= s; a.z *= s; a.w *= s;
}

inline __host__ __device__ real4 operator/(real4 a, real4 b)
{
	return MAKE_REAL4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}

inline __host__ __device__ real4 operator/(real4 a, real s)
{
	return a / s;
}

inline __host__ __device__ void operator/=(real4 &a, real s)
{
	real inv = 1.0 / s;
	a *= inv;
}

inline __host__ __device__ real dot(real4 a, real4 b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

inline __host__ __device__ real length(real4 a)
{
	return sqrtf(dot(a,a));
}

inline __host__ __device__ real4 normalize(real4 a)
{
	return a / length(a);
}

// real6
inline __host__ __device__ real6 operator+(real6 a, real6 b)
{
	real6 c;
	c.x = a.x + b.x; c.y = a.y + b.y; c.z = a.z + b.z; c.u = a.u + b.u; c.v = a.v + b.v; c.w = a.w + b.w;
	return c;
}

inline __host__ __device__ void operator+=(real6 &a, real6 b)
{
	a.x += b.x; a.y += b.y; a.z += b.z; a.u += b.u; a.v += b.v; a.w += b.w;
}

inline __host__ __device__ real6 operator-(real6 a, real6 b)
{
	real6 c;
	c.x = a.x - b.x; c.y = a.y - b.y; c.z = a.z - b.z; c.u = a.u - b.u; c.v = a.v - b.v; c.w = a.w - b.w;
	return c;
}

inline __host__ __device__ void operator-=(real6 &a, real6 b)
{
	a.x -= b.x; a.y -= b.y; a.z -= b.z; a.u -= b.u; a.v -= b.v; a.w -= b.w;
}

// real8
inline __host__ __device__ real8 operator+(real8 a, real8 b)
{
	real8 c;
	c.s0 = a.s0 + b.s0; c.s1 = a.s1 + b.s1; c.s2 = a.s2 + b.s2; c.s3 = a.s3 + b.s3; 
	c.s4 = a.s4 + b.s4; c.s5 = a.s5 + b.s5; c.s6 = a.s6 + b.s6; c.s7 = a.s7 + b.s7;
	return c;
}

inline __host__ __device__ void operator+=(real8 &a, real8 b)
{
	a.s0 += b.s0; a.s1 += b.s1; a.s2 += b.s2; a.s3 += b.s3; 
	a.s4 += b.s4; a.s5 += b.s5; a.s6 += b.s6; a.s7 += b.s7;
}

inline __host__ __device__ real8 operator-(real8 a, real8 b)
{
	real8 c;
	c.s0 = a.s0 - b.s0; c.s1 = a.s1 - b.s1; c.s2 = a.s2 - b.s2; c.s3 = a.s3 - b.s3; 
	c.s4 = a.s4 - b.s4; c.s5 = a.s5 - b.s5; c.s6 = a.s6 - b.s6; c.s7 = a.s7 - b.s7;
	return c;
}

inline __host__ __device__ void operator-=(real8 &a, real8 b)
{
	a.s0 -= b.s0; a.s1 -= b.s1; a.s2 -= b.s2; a.s3 -= b.s3; 
	a.s4 -= b.s4; a.s5 -= b.s5; a.s6 -= b.s6; a.s7 -= b.s7;
}


// MAKE_REAL
template <typename T>
inline __host__ __device__ T make_vector();

template <>
inline __host__ __device__ int make_vector()
{
	return 0;
}
template <>
inline __host__ __device__ int2 make_vector()
{
	return make_int2(0, 0);
}
template <>
inline __host__ __device__ int3 make_vector()
{
	return make_int3(0, 0, 0);
}
template<>
inline __host__ __device__ int4 make_vector()
{
	return make_int4(0, 0, 0, 0);
}
template <>
inline __host__ __device__ real make_vector()
{
	return 0.0;
}
template <>
inline __host__ __device__ real2 make_vector()
{
	return MAKE_REAL2(0.0, 0.0);
}
template <>
inline __host__ __device__ real3 make_vector()
{
	return MAKE_REAL3(0.0, 0.0, 0.0);
}
template<>
inline __host__ __device__ real4 make_vector()
{
	return MAKE_REAL4(0.0, 0.0, 0.0, 0.0);
}

#endif

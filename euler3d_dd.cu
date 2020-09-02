// Copyright 2009, Andrew Corrigan, acorriga@gmu.edu
// This code is from the AIAA-2009-4001 paper
// #include <cutil.h>
//#include <helper_cuda.h>
//#include <helper_timer.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <cuda_runtime_api.h>
#include <qd/dd_real.h>
//#include "gpuprec/gqd/gqd.cu"
//#if CUDART_VERSION < 3000
//struct double3
//{
//	double x, y, z;
//};
//#endif
/*
 * Options 
 * 
 */
using namespace std;
typedef struct gdd_real3
{
	gdd_real x, y, z;
} gdd_real3;

typedef struct dd_real3 
{
  dd_real x, y, z;
} dd_real3;

void qd2gqd(dd_real3* dd_data, gdd_real3* gdd_data, const unsigned int numElement) {
    for (unsigned int i = 0; i < numElement; i++) {
        gdd_data[i].x.x = dd_data[i].x.x[0];
        gdd_data[i].x.y = dd_data[i].x.x[1];
        gdd_data[i].y.x = dd_data[i].y.x[0];
        gdd_data[i].y.y = dd_data[i].y.x[1];
        gdd_data[i].z.x = dd_data[i].z.x[0];
        gdd_data[i].z.y = dd_data[i].z.x[1];
    }
}
void qd2gqd(dd_real* dd_data, gdd_real* gdd_data, const unsigned int numElement) {
    for (unsigned int i = 0; i < numElement; i++) {
        gdd_data[i].x = dd_data[i].x[0];
        gdd_data[i].y = dd_data[i].x[1];
    }
}
void gqd2qd(gdd_real* gdd_data, dd_real* dd_data, const unsigned int numElement) {
    for (unsigned int i = 0; i < numElement; i++) {
        dd_data[i].x[0] = gdd_data[i].x;
        dd_data[i].x[1] = gdd_data[i].y;
    }
}

#define GAMMA 1.4
#define iterations 2000
#ifndef block_length
	#define block_length 128
#endif
#define NDIM 3
#define NNB 4
/* (previously processed: ignoring self-referential macro declaration) macro name = RK */ 
#define ff_mach 1.2
#define deg_angle_of_attack 0.0
//template < typename T >
//void check ( T result, char const * const func, const char * const file,
//           int const line ) {
//  if ( result ) {
//    fprintf ( stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
//            static_cast < unsigned int > ( result ), _cudaGetErrorEnum ( result ), func );
//    exit ( 1 );
//  }
//}
////#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)
//
//static const char *_cudaGetErrorEnum(cudaError_t error)
//{
//  return cudaGetErrorName(error);
//}
/*
 * not options
 */
#if block_length > 128
#warning "the kernels may fail too launch on some systems if the block length is too large"
#endif
#define VAR_DENSITY 0
#define VAR_MOMENTUM  1
#define VAR_DENSITY_ENERGY (VAR_MOMENTUM+NDIM)
#define NVAR (VAR_DENSITY_ENERGY+1)
/*
 * Generic functions
 */
template < typename T >
T * alloc ( int N )
{
 T * t;
 check ( ( cudaMalloc ( ( void * * ) & t, sizeof ( T ) * N ) ), "cudaMalloc((void**)&t, sizeof(T)*N)", "euler3d_double.cu", 85 );
 return t;
}
template < typename T >
void dealloc ( T * array )
{
 check ( ( cudaFree ( ( void * ) array ) ), "cudaFree((void*)array)", "euler3d_double.cu", 92 );
}
template < typename T >
void copy ( T * dst, T * src, int N )
{
 check ( ( cudaMemcpy ( ( void * ) dst, ( void * ) src, N * sizeof ( T ), cudaMemcpyDeviceToDevice ) ), "cudaMemcpy((void*)dst, (void*)src, N*sizeof(T), cudaMemcpyDeviceToDevice)", "euler3d_double.cu", 98 );
}
template < typename T >
void upload ( T * dst, T * src, int N )
{
 check ( ( cudaMemcpy ( ( void * ) dst, ( void * ) src, N * sizeof ( T ), cudaMemcpyHostToDevice ) ), "cudaMemcpy((void*)dst, (void*)src, N*sizeof(T), cudaMemcpyHostToDevice)", "euler3d_double.cu", 104 );
}
template < typename T >
void download ( T * dst, T * src, int N )
{
 check ( ( cudaMemcpy ( ( void * ) dst, ( void * ) src, N * sizeof ( T ), cudaMemcpyDeviceToHost ) ), "cudaMemcpy((void*)dst, (void*)src, N*sizeof(T), cudaMemcpyDeviceToHost)", "euler3d_double.cu", 110 );
}

void dump(gdd_real *variables,int nel,int nelr)
{
  dd_real *h_variables = new dd_real [nelr * (1 + 3 + 1)];
  gdd_real *hh_variables = new gdd_real [nelr * (1 + 3 + 1)];
  download(hh_variables,variables,nelr * (1 + 3 + 1));
  gqd2qd(hh_variables, h_variables, (nelr * (1 + 3 + 1)));
  double output;
{
    std::ofstream file("density");
    (((file << nel)<<" ") << nelr) << endl;
    for (int i = 0; i < nel; i++) { 
      output = to_double(h_variables[i + 0 * nelr]);
      (file << output) << endl;
    }
  }
{
    std::ofstream file("momentum");
    (((file << nel)<<" ") << nelr) << endl;
    for (int i = 0; i < nel; i++) {
      for (int j = 0; j != 3; j++) { 
        output = to_double(h_variables[i + (1 + j) * nelr]);
        (file << output)<<" ";
      }
      file << endl;
    }
  }
{
    std::ofstream file("density_energy");
    (((file << nel)<<" ") << nelr) << endl;
    for (int i = 0; i < nel; i++)  {
      output = to_double(h_variables[i + (1 + 3) * nelr]);
      (file << output) << endl;
    }
  }
  delete []h_variables;
}
/*
 * Element-based Cell-centered FVM solver functions
 */
 __constant__ gdd_real ff_variable[5];
 __constant__ gdd_real3 ff_flux_contribution_momentum_x[1];
 __constant__ gdd_real3 ff_flux_contribution_momentum_y[1];
 __constant__ gdd_real3 ff_flux_contribution_momentum_z[1];
 __constant__ gdd_real3 ff_flux_contribution_density_energy[1];

__global__ void cuda_initialize_variables(int nelr,gdd_real *variables)
{
  const int i = (blockDim . x * blockIdx . x + threadIdx . x);
  for (int j = 0; j < 1 + 3 + 1; j++) 
    variables[i + j * nelr] = ff_variable[j];
}

void initialize_variables(int nelr,gdd_real *variables)
{
  ::dim3 Dg((nelr / 128));
  ::dim3 Db(128);
  cuda_initialize_variables<<<Dg,Db>>>(nelr,variables);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    fprintf(stderr,"GPUassert: %s Initializing variables\n",(cudaGetErrorString(error)));
    exit(- 1);
  }
}

inline __device__ void compute_flux_contribution(gdd_real &density,gdd_real3 &momentum,gdd_real &density_energy,gdd_real &pressure,gdd_real3 &velocity,gdd_real3 &fc_momentum_x,gdd_real3 &fc_momentum_y,gdd_real3 &fc_momentum_z,gdd_real3 &fc_density_energy)
{
  fc_momentum_x . x = velocity . x * momentum . x + pressure;
  fc_momentum_x . y = velocity . x * momentum . y;
  fc_momentum_x . z = velocity . x * momentum . z;
  fc_momentum_y . x = fc_momentum_x . y;
  fc_momentum_y . y = velocity . y * momentum . y + pressure;
  fc_momentum_y . z = velocity . y * momentum . z;
  fc_momentum_z . x = fc_momentum_x . z;
  fc_momentum_z . y = fc_momentum_y . z;
  fc_momentum_z . z = velocity . z * momentum . z + pressure;
  gdd_real de_p = density_energy + pressure;
  fc_density_energy . x = velocity . x * de_p;
  fc_density_energy . y = velocity . y * de_p;
  fc_density_energy . z = velocity . z * de_p;
}

void compute_flux_contribution(dd_real &density, dd_real3 &momentum, dd_real &density_energy, dd_real &pressure, dd_real3 &velocity, dd_real3 &fc_momentum_x, dd_real3 &fc_momentum_y, dd_real3 &fc_momentum_z, dd_real3 &fc_density_energy)
{
  fc_momentum_x . x = velocity . x * momentum . x + pressure;
  fc_momentum_x . y = velocity . x * momentum . y;
  fc_momentum_x . z = velocity . x * momentum . z;
  fc_momentum_y . x = fc_momentum_x . y;
  fc_momentum_y . y = velocity . y * momentum . y + pressure;
  fc_momentum_y . z = velocity . y * momentum . z;
  fc_momentum_z . x = fc_momentum_x . z;
  fc_momentum_z . y = fc_momentum_y . z;
  fc_momentum_z . z = velocity . z * momentum . z + pressure;
  dd_real de_p = density_energy + pressure;
  fc_density_energy . x = velocity . x * de_p;
  fc_density_energy . y = velocity . y * de_p;
  fc_density_energy . z = velocity . z * de_p;
}


inline __device__ void compute_velocity(gdd_real &density,gdd_real3 &momentum,gdd_real3 &velocity)
{
  velocity . x = momentum . x / density;
  velocity . y = momentum . y / density;
  velocity . z = momentum . z / density;
}

inline __device__ gdd_real compute_speed_sqd(gdd_real3 &velocity)
{
  return velocity . x * velocity . x + velocity . y * velocity . y + velocity . z * velocity . z;
}

inline __device__ gdd_real compute_pressure(gdd_real &density,gdd_real &density_energy,gdd_real &speed_sqd)
{
  return (((double )1.4) - ((double )1.0)) * (density_energy - ((double )0.5) * density * speed_sqd);
}

inline __device__ gdd_real compute_speed_of_sound(gdd_real &density,gdd_real &pressure)
{
  return sqrt(((double )1.4) * pressure / density);
}

__global__ void cuda_compute_step_factor(int nelr,gdd_real *variables,gdd_real *areas,gdd_real *step_factors)
{
  const int i = (blockDim . x * blockIdx . x + threadIdx . x);
  gdd_real density = variables[i + 0 * nelr];
  gdd_real3 momentum;
  momentum . x = variables[i + (1 + 0) * nelr];
  momentum . y = variables[i + (1 + 1) * nelr];
  momentum . z = variables[i + (1 + 2) * nelr];
  ::gdd_real density_energy = variables[i + (1 + 3) * nelr];
  ::gdd_real3 velocity;
  compute_velocity(density,momentum,velocity);
  ::gdd_real speed_sqd = compute_speed_sqd(velocity);
  ::gdd_real pressure = compute_pressure(density,density_energy,speed_sqd);
  ::gdd_real speed_of_sound = compute_speed_of_sound(density,pressure);
// dt = double(0.5) * sqrt(areas[i]) /  (||v|| + c).... but when we do time stepping, this later would need to be divided by the area, so we just do it all at once
  step_factors[i] = ((double )0.5) / (sqrt(areas[i]) * (sqrt(speed_sqd) + speed_of_sound));
}

void compute_step_factor(int nelr,gdd_real *variables,gdd_real *areas,gdd_real *step_factors)
{
  ::dim3 Dg((nelr / 128));
  ::dim3 Db(128);
  cuda_compute_step_factor<<<Dg,Db>>>(nelr,variables,areas,step_factors);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    fprintf(stderr,"GPUassert: %s compute_step_factor failed\n",(cudaGetErrorString(error)));
    exit(- 1);
  }
}
/*
 *
 *
*/

__global__ void cuda_compute_flux(int nelr,int *elements_surrounding_elements,gdd_real *normals,gdd_real *variables,gdd_real *fluxes)
{
  const double smoothing_coefficient = (double )0.2;
  const int i = (blockDim . x * blockIdx . x + threadIdx . x);
  int j;
  int nb;
  gdd_real3 normal;
  gdd_real normal_len;
  ::gdd_real factor;
  ::gdd_real density_i = variables[i + 0 * nelr];
  ::gdd_real3 momentum_i;
  momentum_i . x = variables[i + (1 + 0) * nelr];
  momentum_i . y = variables[i + (1 + 1) * nelr];
  momentum_i . z = variables[i + (1 + 2) * nelr];
  ::gdd_real density_energy_i = variables[i + (1 + 3) * nelr];
  ::gdd_real3 velocity_i;
  compute_velocity(density_i,momentum_i,velocity_i);
  ::gdd_real speed_sqd_i = compute_speed_sqd(velocity_i);
  ::gdd_real speed_i = sqrt(speed_sqd_i);
  ::gdd_real pressure_i = compute_pressure(density_i,density_energy_i,speed_sqd_i);
  ::gdd_real speed_of_sound_i = compute_speed_of_sound(density_i,pressure_i);
  ::gdd_real3 flux_contribution_i_momentum_x;
  ::gdd_real3 flux_contribution_i_momentum_y;
  ::gdd_real3 flux_contribution_i_momentum_z;
  ::gdd_real3 flux_contribution_i_density_energy;
  compute_flux_contribution(density_i,momentum_i,density_energy_i,pressure_i,velocity_i,flux_contribution_i_momentum_x,flux_contribution_i_momentum_y,flux_contribution_i_momentum_z,flux_contribution_i_density_energy);
  ::gdd_real flux_i_density;
  flux_i_density.x = (double )0.0;
  flux_i_density.y = (double )0.0;
  ::gdd_real3 flux_i_momentum;
  flux_i_momentum . x.x = ((double )0.0);
  flux_i_momentum . x.y = ((double )0.0);
  flux_i_momentum . y.x = ((double )0.0);
  flux_i_momentum . y.y = ((double )0.0);
  flux_i_momentum . z.x = ((double )0.0);
  flux_i_momentum . z.y = ((double )0.0);
  ::gdd_real flux_i_density_energy;
  flux_i_density_energy.x = (double )0.0;
  flux_i_density_energy.y = (double )0.0;
  ::gdd_real3 velocity_nb;
  ::gdd_real density_nb;
  ::gdd_real density_energy_nb;
  ::gdd_real3 momentum_nb;
  ::gdd_real3 flux_contribution_nb_momentum_x;
  ::gdd_real3 flux_contribution_nb_momentum_y;
  ::gdd_real3 flux_contribution_nb_momentum_z;
  ::gdd_real3 flux_contribution_nb_density_energy;
  ::gdd_real speed_sqd_nb;
  ::gdd_real speed_of_sound_nb;
  ::gdd_real pressure_nb;
  
#pragma unroll
  for (j = 0; j < 4; j++) {
    nb = elements_surrounding_elements[i + j * nelr];
    normal . x = normals[i + (j + 0 * 4) * nelr];
    normal . y = normals[i + (j + 1 * 4) * nelr];
    normal . z = normals[i + (j + 2 * 4) * nelr];
    normal_len = sqrt(normal . x * normal . x + normal . y * normal . y + normal . z * normal . z);
// a legitimate neighbor
    if (nb >= 0) {
      density_nb = variables[nb + 0 * nelr];
      momentum_nb . x = variables[nb + (1 + 0) * nelr];
      momentum_nb . y = variables[nb + (1 + 1) * nelr];
      momentum_nb . z = variables[nb + (1 + 2) * nelr];
      density_energy_nb = variables[nb + (1 + 3) * nelr];
      compute_velocity(density_nb,momentum_nb,velocity_nb);
      speed_sqd_nb = compute_speed_sqd(velocity_nb);
      pressure_nb = compute_pressure(density_nb,density_energy_nb,speed_sqd_nb);
      speed_of_sound_nb = compute_speed_of_sound(density_nb,pressure_nb);
      compute_flux_contribution(density_nb,momentum_nb,density_energy_nb,pressure_nb,velocity_nb,flux_contribution_nb_momentum_x,flux_contribution_nb_momentum_y,flux_contribution_nb_momentum_z,flux_contribution_nb_density_energy);
// artificial viscosity
      factor = -normal_len * smoothing_coefficient * ((double )0.5) * (speed_i + sqrt(speed_sqd_nb) + speed_of_sound_i + speed_of_sound_nb);
      flux_i_density = flux_i_density + factor * (density_i - density_nb);
      flux_i_density_energy = flux_i_density_energy + factor * (density_energy_i - density_energy_nb);
      flux_i_momentum . x = flux_i_momentum . x + factor * (momentum_i . x - momentum_nb . x);
      flux_i_momentum . y = flux_i_momentum . y + factor * (momentum_i . y - momentum_nb . y);
      flux_i_momentum . z = flux_i_momentum . z +factor * (momentum_i . z - momentum_nb . z);
// accumulate cell-centered fluxes
      factor = ((double )0.5) * normal . x;
      flux_i_density = flux_i_density +factor * (momentum_nb . x + momentum_i . x);
      flux_i_density_energy =flux_i_density_energy + factor * (flux_contribution_nb_density_energy . x + flux_contribution_i_density_energy . x);
      flux_i_momentum . x =flux_i_momentum . x + factor * (flux_contribution_nb_momentum_x . x + flux_contribution_i_momentum_x . x);
      flux_i_momentum . y =flux_i_momentum . y + factor * (flux_contribution_nb_momentum_y . x + flux_contribution_i_momentum_y . x);
      flux_i_momentum . z =flux_i_momentum . z + factor * (flux_contribution_nb_momentum_z . x + flux_contribution_i_momentum_z . x);
      factor = ((double )0.5) * normal . y;
      flux_i_density =flux_i_density + factor * (momentum_nb . y + momentum_i . y);
      flux_i_density_energy =flux_i_density_energy + factor * (flux_contribution_nb_density_energy . y + flux_contribution_i_density_energy . y);
      flux_i_momentum . x =flux_i_momentum . x + factor * (flux_contribution_nb_momentum_x . y + flux_contribution_i_momentum_x . y);
      flux_i_momentum . y =flux_i_momentum . y + factor * (flux_contribution_nb_momentum_y . y + flux_contribution_i_momentum_y . y);
      flux_i_momentum . z =flux_i_momentum . z + factor * (flux_contribution_nb_momentum_z . y + flux_contribution_i_momentum_z . y);
      factor = ((double )0.5) * normal . z;
      flux_i_density =flux_i_density + factor * (momentum_nb . z + momentum_i . z);
      flux_i_density_energy =flux_i_density_energy + factor * (flux_contribution_nb_density_energy . z + flux_contribution_i_density_energy . z);
      flux_i_momentum . x =flux_i_momentum . x + factor * (flux_contribution_nb_momentum_x . z + flux_contribution_i_momentum_x . z);
      flux_i_momentum . y =flux_i_momentum . y + factor * (flux_contribution_nb_momentum_y . z + flux_contribution_i_momentum_y . z);
      flux_i_momentum . z =flux_i_momentum . z + factor * (flux_contribution_nb_momentum_z . z + flux_contribution_i_momentum_z . z);
    }
     else 
// a wing boundary
if (nb == - 1) {
      flux_i_momentum . x =flux_i_momentum . x + normal . x * pressure_i;
      flux_i_momentum . y =flux_i_momentum . y + normal . y * pressure_i;
      flux_i_momentum . z =flux_i_momentum . z + normal . z * pressure_i;
    }
     else 
// a far field boundary
if (nb == - 2) {
      factor = ((double )0.5) * normal . x;
      flux_i_density =flux_i_density + factor * (ff_variable[1 + 0] + momentum_i . x);
      flux_i_density_energy =flux_i_density_energy + factor * (ff_flux_contribution_density_energy[0] . x + flux_contribution_i_density_energy . x);
      flux_i_momentum . x =flux_i_momentum . x + factor * (ff_flux_contribution_momentum_x[0] . x + flux_contribution_i_momentum_x . x);
      flux_i_momentum . y =flux_i_momentum . y + factor * (ff_flux_contribution_momentum_y[0] . x + flux_contribution_i_momentum_y . x);
      flux_i_momentum . z =flux_i_momentum . z + factor * (ff_flux_contribution_momentum_z[0] . x + flux_contribution_i_momentum_z . x);
      factor = ((double )0.5) * normal . y;
      flux_i_density =flux_i_density + factor * (ff_variable[1 + 1] + momentum_i . y);
      flux_i_density_energy =flux_i_density_energy + factor * (ff_flux_contribution_density_energy[0] . y + flux_contribution_i_density_energy . y);
      flux_i_momentum . x =flux_i_momentum . x + factor * (ff_flux_contribution_momentum_x[0] . y + flux_contribution_i_momentum_x . y);
      flux_i_momentum . y =flux_i_momentum . y + factor * (ff_flux_contribution_momentum_y[0] . y + flux_contribution_i_momentum_y . y);
      flux_i_momentum . z =flux_i_momentum . z + factor * (ff_flux_contribution_momentum_z[0] . y + flux_contribution_i_momentum_z . y);
      factor = ((double )0.5) * normal . z;
      flux_i_density =flux_i_density + factor * (ff_variable[1 + 2] + momentum_i . z);
      flux_i_density_energy =flux_i_density_energy + factor * (ff_flux_contribution_density_energy[0] . z + flux_contribution_i_density_energy . z);
      flux_i_momentum . x =flux_i_momentum . x + factor * (ff_flux_contribution_momentum_x[0] . z + flux_contribution_i_momentum_x . z);
      flux_i_momentum . y =flux_i_momentum . y + factor * (ff_flux_contribution_momentum_y[0] . z + flux_contribution_i_momentum_y . z);
      flux_i_momentum . z =flux_i_momentum . z + factor * (ff_flux_contribution_momentum_z[0] . z + flux_contribution_i_momentum_z . z);
    }
  }
  fluxes[i + 0 * nelr] = flux_i_density;
  fluxes[i + (1 + 0) * nelr] = flux_i_momentum . x;
  fluxes[i + (1 + 1) * nelr] = flux_i_momentum . y;
  fluxes[i + (1 + 2) * nelr] = flux_i_momentum . z;
  fluxes[i + (1 + 3) * nelr] = flux_i_density_energy;
}

void compute_flux(int nelr,int *elements_surrounding_elements,gdd_real *normals,gdd_real *variables,gdd_real *fluxes)
{
  ::dim3 Dg((nelr / 128));
  ::dim3 Db(128);
  cuda_compute_flux<<<Dg,Db>>>(nelr,elements_surrounding_elements,normals,variables,fluxes);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    fprintf(stderr,"GPUassert: %s compute_flux failed\n",(cudaGetErrorString(error)));
    exit(- 1);
  }
}

__global__ void cuda_time_step(int j,int nelr,gdd_real *old_variables,gdd_real *variables,gdd_real *step_factors,gdd_real *fluxes)
{
  const int i = (blockDim . x * blockIdx . x + threadIdx . x);
  gdd_real factor = step_factors[i] / ((double )(3 + 1 - j));
  variables[i + 0 * nelr] = old_variables[i + 0 * nelr] + factor * fluxes[i + 0 * nelr];
  variables[i + (1 + 3) * nelr] = old_variables[i + (1 + 3) * nelr] + factor * fluxes[i + (1 + 3) * nelr];
  variables[i + (1 + 0) * nelr] = old_variables[i + (1 + 0) * nelr] + factor * fluxes[i + (1 + 0) * nelr];
  variables[i + (1 + 1) * nelr] = old_variables[i + (1 + 1) * nelr] + factor * fluxes[i + (1 + 1) * nelr];
  variables[i + (1 + 2) * nelr] = old_variables[i + (1 + 2) * nelr] + factor * fluxes[i + (1 + 2) * nelr];
}

void time_step(int j,int nelr,gdd_real *old_variables,gdd_real *variables,gdd_real *step_factors,gdd_real *fluxes)
{
  ::dim3 Dg((nelr / 128));
  ::dim3 Db(128);
  cuda_time_step<<<Dg,Db>>>(j,nelr,old_variables,variables,step_factors,fluxes);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    fprintf(stderr,"GPUassert: %s update failed\n",(cudaGetErrorString(error)));
    exit(- 1);
  }
}
/*
 * Main function
 */

int main(int argc,char **argv)
{
  if (argc < 2) {
    (std::cout<<"specify data file name") << endl;
    return 0;
  }
  const char *data_file_name = argv[1];
  struct cudaDeviceProp prop;
  int dev;
// CUDA_SAFE_CALL(cudaSetDevice(0));
// CUDA_SAFE_CALL(cudaGetDevice(&dev));
// CUDA_SAFE_CALL(cudaGetDeviceProperties(&prop, dev));
  check((cudaSetDevice(0)),"cudaSetDevice(0)","euler3d_double.cu",438);
  check((cudaGetDevice(&dev)),"cudaGetDevice(&dev)","euler3d_double.cu",439);
  check((cudaGetDeviceProperties(&prop,dev)),"cudaGetDeviceProperties(&prop, dev)","euler3d_double.cu",440);
  printf("Name:                     %s\n",prop . name);
// set far field conditions and load them into constant memory on the gpu
  dd_real h_ff_variable[5];
  gdd_real hh_ff_variable[5];
  const double angle_of_attack = ((double )(3.1415926535897931 / 180.0)) * ((double )0.0);
  h_ff_variable[0] = ((double )1.4);
  dd_real ff_pressure = (double )1.0;
  dd_real ff_speed_of_sound = sqrt(1.4 * ff_pressure / h_ff_variable[0]);
  dd_real ff_speed = ((double )1.2) * ff_speed_of_sound;
  dd_real3 ff_velocity;
  ff_velocity . x = ff_speed * ((double )(cos((double )angle_of_attack)));
  ff_velocity . y = ff_speed * ((double )(sin((double )angle_of_attack)));
  ff_velocity . z = 0.0;
  h_ff_variable[1 + 0] = h_ff_variable[0] * ff_velocity . x;
  h_ff_variable[1 + 1] = h_ff_variable[0] * ff_velocity . y;
  h_ff_variable[1 + 2] = h_ff_variable[0] * ff_velocity . z;
  h_ff_variable[1 + 3] = h_ff_variable[0] * (((double )0.5) * (ff_speed * ff_speed)) + ff_pressure / ((double )(1.4 - 1.0));
  dd_real3 h_ff_momentum;
  h_ff_momentum . x =  *(h_ff_variable + 1 + 0);
  h_ff_momentum . y =  *(h_ff_variable + 1 + 1);
  h_ff_momentum . z =  *(h_ff_variable + 1 + 2);
  dd_real3 h_ff_flux_contribution_momentum_x;
  dd_real3 h_ff_flux_contribution_momentum_y;
  dd_real3 h_ff_flux_contribution_momentum_z;
  dd_real3 h_ff_flux_contribution_density_energy;
  gdd_real3 hh_ff_flux_contribution_momentum_x;
  gdd_real3 hh_ff_flux_contribution_momentum_y;
  gdd_real3 hh_ff_flux_contribution_momentum_z;
  gdd_real3 hh_ff_flux_contribution_density_energy;
  compute_flux_contribution(h_ff_variable[0],h_ff_momentum,h_ff_variable[1 + 3],ff_pressure,ff_velocity,h_ff_flux_contribution_momentum_x,h_ff_flux_contribution_momentum_y,h_ff_flux_contribution_momentum_z,h_ff_flux_contribution_density_energy);
// copy far field conditions to the gpu
  qd2gqd(h_ff_variable, hh_ff_variable, 5);
  qd2gqd(&h_ff_flux_contribution_momentum_x, &hh_ff_flux_contribution_momentum_x, 1);
  qd2gqd(&h_ff_flux_contribution_momentum_y, &hh_ff_flux_contribution_momentum_y, 1);
  qd2gqd(&h_ff_flux_contribution_momentum_z, &hh_ff_flux_contribution_momentum_z, 1);
  qd2gqd(&h_ff_flux_contribution_density_energy, &hh_ff_flux_contribution_density_energy, 1);
  check((cudaMemcpyToSymbol(ff_variable,hh_ff_variable,(1 + 3 + 1) * sizeof(gdd_real3 ))),"cudaMemcpyToSymbol(ff_variable, h_ff_variable, NVAR*sizeof(double))","euler3d_double.cu",477);
  check((cudaMemcpyToSymbol(ff_flux_contribution_momentum_x,(&h_ff_flux_contribution_momentum_x),sizeof(gdd_real3 ))),"cudaMemcpyToSymbol(ff_flux_contribution_momentum_x, &h_ff_flux_contribution_momentum_x, sizeof(double3))","euler3d_double.cu",478);
  check((cudaMemcpyToSymbol(ff_flux_contribution_momentum_y,(&h_ff_flux_contribution_momentum_y),sizeof(gdd_real3 ))),"cudaMemcpyToSymbol(ff_flux_contribution_momentum_y, &h_ff_flux_contribution_momentum_y, sizeof(double3))","euler3d_double.cu",479);
  check((cudaMemcpyToSymbol(ff_flux_contribution_momentum_z,(&h_ff_flux_contribution_momentum_z),sizeof(gdd_real3 ))),"cudaMemcpyToSymbol(ff_flux_contribution_momentum_z, &h_ff_flux_contribution_momentum_z, sizeof(double3))","euler3d_double.cu",480);
  check((cudaMemcpyToSymbol(ff_flux_contribution_density_energy,(&h_ff_flux_contribution_density_energy),sizeof(gdd_real3 ))),"cudaMemcpyToSymbol(ff_flux_contribution_density_energy, &h_ff_flux_contribution_density_energy, sizeof(double3))","euler3d_double.cu",482);
  int nel;
  int nelr;
// read in domain geometry
  gdd_real *areas;
  int *elements_surrounding_elements;
  gdd_real *normals;
  std::ifstream file(data_file_name);
  file >> nel;
  nelr = 128 * (nel / 128 + min(1,nel % 128));
  dd_real *h_areas = new dd_real [nelr];
  int *h_elements_surrounding_elements = new int [nelr * 4];
  dd_real *h_normals = new dd_real [nelr * 3 * 4];
  double input;
// read in data
  for (int i = 0; i < nel; i++) {
    file >> input;
    h_areas[i] = input;
    for (int j = 0; j < 4; j++) {
      file >> h_elements_surrounding_elements[i + j * nelr];
      if (h_elements_surrounding_elements[i + j * nelr] < 0) 
        h_elements_surrounding_elements[i + j * nelr] = - 1;
//it's coming in with Fortran numbering				
      h_elements_surrounding_elements[i + j * nelr]--;
      for (int k = 0; k < 3; k++) {
        file >> input;
        h_normals[i + (j + k * 4) * nelr] = input;
        h_normals[i + (j + k * 4) * nelr] = -h_normals[i + (j + k * 4) * nelr];
      }
    }
  }
// fill in remaining data
  int last = nel - 1;
  for (int i = nel; i < nelr; i++) {
    h_areas[i] = h_areas[last];
    for (int j = 0; j < 4; j++) {
// duplicate the last element
      h_elements_surrounding_elements[i + j * nelr] = h_elements_surrounding_elements[last + j * nelr];
      for (int k = 0; k < 3; k++) 
        h_normals[last + (j + k * 4) * nelr] = h_normals[last + (j + k * 4) * nelr];
    }
  }
  gdd_real *hh_areas = new gdd_real [nelr];
  gdd_real *hh_normals = new gdd_real [nelr * 3 * 4];
  qd2gqd(h_areas, hh_areas, nelr);
  qd2gqd(h_normals, hh_normals, (nelr * 3 * 4));
  areas = alloc< gdd_real  > (nelr);
  upload< gdd_real  > (areas,hh_areas,nelr);
  elements_surrounding_elements = alloc< int  > (nelr * 4);
  upload< int  > (elements_surrounding_elements,h_elements_surrounding_elements,nelr * 4);
  normals = alloc< gdd_real  > (nelr * 3 * 4);
  upload< gdd_real  > (normals,hh_normals,nelr * 3 * 4);
  delete []h_areas;
  delete []h_elements_surrounding_elements;
  delete []h_normals;
// Create arrays and set initial conditions
  gdd_real *variables = alloc< gdd_real  > (nelr * (1 + 3 + 1));
  initialize_variables(nelr,variables);
  gdd_real *old_variables = alloc< gdd_real  > (nelr * (1 + 3 + 1));
  gdd_real *fluxes = alloc< gdd_real  > (nelr * (1 + 3 + 1));
  gdd_real *step_factors = alloc< gdd_real  > (nelr);
// make sure all memory is doublely allocated before we start timing
  initialize_variables(nelr,old_variables);
  initialize_variables(nelr,fluxes);
  cudaMemset((void *)step_factors,0,sizeof(gdd_real) * nelr);
// make sure CUDA isn't still doing something before we start timing
  cudaThreadSynchronize();
// these need to be computed the first time in order to compute time step
  (std::cout<<"Starting...") << endl;
  cudaError_t error;
StopWatchInterface *timer = NULL;
sdkCreateTimer( &timer);
sdkStartTimer( &timer);
// Begin iterations
  for (int i = 0; i < 2000; i++) {
    copy< gdd_real  > (old_variables,variables,nelr * (1 + 3 + 1));
// for the first iteration we compute the time step
    compute_step_factor(nelr,variables,areas,step_factors);
    error = cudaGetLastError();
    if (error != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s compute_step_factor failed\n",(cudaGetErrorString(error)));
      exit(- 1);
    }
    for (int j = 0; j < 3; j++) {
      compute_flux(nelr,elements_surrounding_elements,normals,variables,fluxes);
      error = cudaGetLastError();
      if (error != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s compute_flux failed\n",(cudaGetErrorString(error)));
        exit(- 1);
      }
      time_step(j,nelr,old_variables,variables,step_factors,fluxes);
      error = cudaGetLastError();
      if (error != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s time_step failed\n",(cudaGetErrorString(error)));
        exit(- 1);
      }
    }
  }
  cudaThreadSynchronize();
sdkStopTimer(&timer);  
std::cout  << (sdkGetAverageTimerValue(&timer)/1000.0)  / iterations << " seconds per iteration" << std::endl;
  (std::cout<<"Saving solution...") << endl;
  dump(variables,nel,nelr);
  (std::cout<<"Saved solution...") << endl;
  (std::cout<<"Cleaning up...") << endl;
  dealloc< gdd_real  > (areas);
  dealloc< int  > (elements_surrounding_elements);
  dealloc< gdd_real  > (normals);
  dealloc< gdd_real  > (variables);
  dealloc< gdd_real  > (old_variables);
  dealloc< gdd_real  > (fluxes);
  dealloc< gdd_real  > (step_factors);
  (std::cout<<"Done...") << endl;
  return 0;
}

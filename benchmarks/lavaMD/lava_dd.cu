//#include <helper_cuda.h>

#include <helper_timer.h>

#include <mpfr.h>

#include <qd/dd_real.h>

#include "../../gpuprec/gqd/gqd.cu"

using namespace std;

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

void qd2gqd2(dd_real dd_data[][5], gdd_real gdd_data[][5], int d1, int d2, int numElement) {

    for (unsigned int i = 0; i < d1; i++) {

      for (unsigned int j = 0; j < d2; j++) {

        gdd_data[i][j].x = dd_data[i][j].x[0];

        gdd_data[i][j].y = dd_data[i][j].x[1];

      }

    }

}

#include <stdio.h>					

#include <stdlib.h>					

#include <stdbool.h>				

#include <iostream>

#include <cuda.h>

#include <cuda_runtime.h>

#include <sys/time.h>

#include <mpfr.h>

#include <math.h>

#include <string.h>

#define NUMBER_PAR_PER_BOX 100							

#ifdef RD_WG_SIZE_0_0

        #define NUMBER_THREADS RD_WG_SIZE_0_0

#elif defined(RD_WG_SIZE_0)

        #define NUMBER_THREADS RD_WG_SIZE_0

#elif defined(RD_WG_SIZE)

        #define NUMBER_THREADS RD_WG_SIZE

#else

        #define NUMBER_THREADS 128

#endif

#define DOT(A,B) ((A.x)*(B.x)+(A.y)*(B.y)+(A.z)*(B.z))	// STABLE

using namespace std;

typedef struct nei_str {

// neighbor box

int x;

int y;

int z;

int number;

long offset;}nei_str;

typedef struct box_str {

// home box

int x;

int y;

int z;

int number;

long offset;

// neighbor boxes

int nn;

::nei_str nei[26];}box_str;

typedef struct dim_str {

// input arguments

int cur_arg;

int arch_arg;

int cores_arg;

int boxes1d_arg;

// system memory

long number_boxes;

long box_mem;

long space_elem;}dim_str;

int isInteger(char *str)

{

  if (( *str) == '\0') {

    return 0;

  }

  for (; ( *str) != '\0'; str++) {

    if (( *str) < 48 || ( *str) > 57) {

      return 0;

    }

  }

  return 1;

}

void checkCUDAError(const char *msg)

{

  cudaError_t err = cudaGetLastError();

  if (cudaSuccess != err) {

    printf("Cuda error: %s: %s.\n",msg,(cudaGetErrorString(err)));

    fflush(0L);

    exit(1);

  }

}

__constant__ float dev_par;

__global__ void kernel_gpu_cuda(::dim_str d_dim_gpu,::box_str *d_box_gpu,float *d_rv_v,float *d_rv_x,float *d_rv_y,float *d_rv_z,float *d_qv_gpu,float *d_fv_v,float *d_fv_x,float *d_fv_y,float *d_fv_z)

{

  int bx = blockIdx . x;

  int tx = threadIdx . x;

  int wtx = tx;

  if (bx < d_dim_gpu . number_boxes) {

// parameters

    float a2 = 2.0 * dev_par * dev_par;

// home box

    int first_i;

    float *rA_v;

    float *rA_x;

    float *rA_y;

    float *rA_z;

    float *fA_v;

    float *fA_x;

    float *fA_y;

    float *fA_z;

    __shared__ float rA_shared_v[100];

    __shared__ float rA_shared_x[100];

    __shared__ float rA_shared_y[100];

    __shared__ float rA_shared_z[100];

// nei box

    int pointer;

    int k = 0;

    int first_j;

    float *rB_v;

    float *rB_x;

    float *rB_y;

    float *rB_z;

    float *qB;

    int j = 0;

    __shared__ float rB_shared_v[100];

    __shared__ float rB_shared_x[100];

    __shared__ float rB_shared_y[100];

    __shared__ float rB_shared_z[100];

    __shared__ float qb_shared[100];

// common

    float r2;

    float u2;

    float vij;

    float fs;

    float fxij;

    float fyij;

    float fzij;

    float s_x;

    float s_y;

    float s_z;

    first_i = d_box_gpu[bx] . offset;

    rA_v =( &d_rv_v[first_i]);

    rA_x =( &d_rv_x[first_i]);

    rA_y =( &d_rv_y[first_i]);

    rA_z =( &d_rv_z[first_i]);

    fA_v =( &d_fv_v[first_i]);

    fA_x =( &d_fv_x[first_i]);

    fA_y =( &d_fv_y[first_i]);

    fA_z =( &d_fv_z[first_i]);

    while(wtx < 100){

      rA_shared_v[wtx] =( rA_v[wtx]);

      rA_shared_x[wtx] =( rA_x[wtx]);

      rA_shared_y[wtx] =( rA_y[wtx]);

      rA_shared_z[wtx] =( rA_z[wtx]);

      wtx = wtx + 640;

    }

    wtx = tx;

    __syncthreads();

    for (k = 0; k < 1 + d_box_gpu[bx] . nn; k++) {

      if (k == 0) {

        pointer = bx;

      }

       else {

        pointer = d_box_gpu[bx] . nei[k - 1] . number;

      }

      first_j = d_box_gpu[pointer] . offset;

      rB_v =( &d_rv_v[first_j]);

      rB_x =( &d_rv_x[first_j]);

      rB_y =( &d_rv_y[first_j]);

      rB_z =( &d_rv_z[first_j]);

      qB =( &d_qv_gpu[first_j]);

      while(wtx < 100){

        rB_shared_v[wtx] =( rB_v[wtx]);

        rB_shared_x[wtx] =( rB_x[wtx]);

        rB_shared_y[wtx] =( rB_y[wtx]);

        rB_shared_z[wtx] =( rB_z[wtx]);

        qb_shared[wtx] =( qB[wtx]);

        wtx = wtx + 640;

      }

      wtx = tx;

      __syncthreads();

      while(wtx < 100){

        for (j = 0; j < 100; j++) {

          r2 = rA_shared_v[wtx] + rB_shared_v[j] - (rA_shared_x[wtx] * rB_shared_x[j] + rA_shared_y[wtx] * rB_shared_y[j] + rA_shared_z[wtx] * rB_shared_z[j]);

          u2 = a2 * r2;

          vij = exp(-u2);

          fs =( 2 * vij);

          s_x = rA_shared_x[wtx] - rB_shared_x[j];

          fxij =( fs * s_x);

          s_y = rA_shared_y[wtx] - rB_shared_y[j];

          fyij =( fs * s_y);

          s_z = rA_shared_z[wtx] - rB_shared_z[j];

          fzij =( fs * s_z);

          fA_v[wtx] =          fA_v[wtx]  +  qb_shared[j] * vij;

          fA_x[wtx] =          fA_x[wtx]  +  qb_shared[j] * fxij;

          fA_y[wtx] =          fA_y[wtx]  +  qb_shared[j] * fyij;

          fA_z[wtx] =          fA_z[wtx]  +  qb_shared[j] * fzij;

        }

        wtx = wtx + 640;

      }

      wtx = tx;

      __syncthreads();

    }

  }

}

int main(int argc,char *argv[])

{

  printf("thread block size of kernel = %d \n",640);

  int i;

  int j;

  int k;

  int l;

  int m;

  int n;

  float par_cpu;

  ::dim_str dim_cpu;

  ::box_str *box_cpu;

  int nh;

  dim_cpu . boxes1d_arg = 1;

  for (dim_cpu . cur_arg = 1; dim_cpu . cur_arg < argc; dim_cpu . cur_arg++) {

    if (strcmp(argv[dim_cpu . cur_arg],"-boxes1d") == 0) {

      if (argc >= dim_cpu . cur_arg + 1) {

        if (isInteger(argv[dim_cpu . cur_arg + 1]) == 1) {

          dim_cpu . boxes1d_arg = atoi(argv[dim_cpu . cur_arg + 1]);

          if (dim_cpu . boxes1d_arg < 0) {

            printf("ERROR: Wrong value to -boxes1d parameter, cannot be <=0\n");

            return 0;

          }

          dim_cpu . cur_arg = dim_cpu . cur_arg + 1;

        }

         else {

          printf("ERROR: Value to -boxes1d parameter in not a number\n");

          return 0;

        }

      }

       else {

        printf("ERROR: Missing value to -boxes1d parameter\n");

        return 0;

      }

    }

     else {

      printf("ERROR: Unknown parameter\n");

      return 0;

    }

  }

  printf("Configuration used: boxes1d = %d\n",dim_cpu . boxes1d_arg);

  par_cpu = 0.5;

  dim_cpu . number_boxes = (dim_cpu . boxes1d_arg * dim_cpu . boxes1d_arg * dim_cpu . boxes1d_arg);

  dim_cpu . space_elem = dim_cpu . number_boxes * 100;

  dim_cpu . box_mem = (dim_cpu . number_boxes * sizeof(::box_str ));

// allocate boxes

  box_cpu = ((::box_str *)(malloc(dim_cpu . box_mem)));

  nh = 0;

  for (i = 0; i < dim_cpu . boxes1d_arg; i++) {

// home boxes in y direction

    for (j = 0; j < dim_cpu . boxes1d_arg; j++) {

// home boxes in x direction

      for (k = 0; k < dim_cpu . boxes1d_arg; k++) {

// current home box

        box_cpu[nh] . x = k;

        box_cpu[nh] . y = j;

        box_cpu[nh] . z = i;

        box_cpu[nh] . number = nh;

        box_cpu[nh] . offset = (nh * 100);

// initialize number of neighbor boxes

        box_cpu[nh] . nn = 0;

// neighbor boxes in z direction

        for (l = - 1; l < 2; l++) {

// neighbor boxes in y direction

          for (m = - 1; m < 2; m++) {

// neighbor boxes in x direction

            for (n = - 1; n < 2; n++) {

              if ((i + l >= 0 && j + m >= 0 && k + n >= 0) == true && (i + l < dim_cpu . boxes1d_arg && j + m < dim_cpu . boxes1d_arg && k + n < dim_cpu . boxes1d_arg) == true && (l == 0 && m == 0 && n == 0) == false) {

                box_cpu[nh] . nei[box_cpu[nh] . nn] . x = k + n;

                box_cpu[nh] . nei[box_cpu[nh] . nn] . y = j + m;

                box_cpu[nh] . nei[box_cpu[nh] . nn] . z = i + l;

                box_cpu[nh] . nei[box_cpu[nh] . nn] . number = box_cpu[nh] . nei[box_cpu[nh] . nn] . z * dim_cpu . boxes1d_arg * dim_cpu . boxes1d_arg + box_cpu[nh] . nei[box_cpu[nh] . nn] . y * dim_cpu . boxes1d_arg + box_cpu[nh] . nei[box_cpu[nh] . nn] . x;

                box_cpu[nh] . nei[box_cpu[nh] . nn] . offset = (box_cpu[nh] . nei[box_cpu[nh] . nn] . number * 100);

                box_cpu[nh] . nn = box_cpu[nh] . nn + 1;

              }

// neighbor boxes in x direction

            }

// neighbor boxes in y direction

          }

// neighbor boxes in z direction

        }

        nh = nh + 1;

// home boxes in x direction

      }

// home boxes in y direction

    }

// home boxes in z direction

  }

  float *rv_cpu_v = new float [dim_cpu . space_elem];

  float *rv_cpu_x = new float [dim_cpu . space_elem];

  float *rv_cpu_y = new float [dim_cpu . space_elem];

  float *rv_cpu_z = new float [dim_cpu . space_elem];

  for (i = 0; i < dim_cpu . space_elem; i = i + 1) {

    rv_cpu_v[i] = (rand() % 10 + 1) / 10.0;

    rv_cpu_x[i] = (rand() % 10 + 1) / 10.0;

    rv_cpu_y[i] = (rand() % 10 + 1) / 10.0;

    rv_cpu_z[i] = (rand() % 10 + 1) / 10.0;

  }

  float *qv_cpu = new float [dim_cpu . space_elem];

  for (i = 0; i < dim_cpu . space_elem; i = i + 1) {

    qv_cpu[i] = (rand() % 10 + 1) / 10.0;

  }

  float *fv_cpu_v = new float [dim_cpu . space_elem];

  float *fv_cpu_x = new float [dim_cpu . space_elem];

  float *fv_cpu_y = new float [dim_cpu . space_elem];

  float *fv_cpu_z = new float [dim_cpu . space_elem];

  for (i = 0; i < dim_cpu . space_elem; i = i + 1) {

    fv_cpu_v[i] = ((float )0.0);

    fv_cpu_x[i] = ((float )0.0);

    fv_cpu_y[i] = ((float )0.0);

    fv_cpu_z[i] = ((float )0.0);

  }

  ::box_str *d_box_gpu;

  float *d_rv_v;

  float *d_rv_x;

  float *d_rv_y;

  float *d_rv_z;

  float *d_qv_gpu;

  float *d_fv_v;

  float *d_fv_x;

  float *d_fv_y;

  float *d_fv_z;

  cudaThreadSynchronize();

  ::dim3 threads;

  ::dim3 blocks;

  blocks . x = dim_cpu . number_boxes;

  blocks . y = 1;

  threads . x = 640;

  threads . y = 1;

  cudaMemcpyToSymbol((dev_par),(&par_cpu),sizeof(float ));

  cudaMalloc((void **)(&d_box_gpu),dim_cpu . box_mem);

  cudaMalloc((void **)(&d_rv_v),dim_cpu . space_elem * sizeof(float ));

  cudaMalloc((void **)(&d_rv_x),dim_cpu . space_elem * sizeof(float ));

  cudaMalloc((void **)(&d_rv_y),dim_cpu . space_elem * sizeof(float ));

  cudaMalloc((void **)(&d_rv_z),dim_cpu . space_elem * sizeof(float ));

  cudaMalloc((void **)(&d_qv_gpu),dim_cpu . space_elem * sizeof(float ));

  cudaMalloc((void **)(&d_fv_v),dim_cpu . space_elem * sizeof(float ));

  cudaMalloc((void **)(&d_fv_x),dim_cpu . space_elem * sizeof(float ));

  cudaMalloc((void **)(&d_fv_y),dim_cpu . space_elem * sizeof(float ));

  cudaMalloc((void **)(&d_fv_z),dim_cpu . space_elem * sizeof(float ));

  struct timeval start_t;

  struct timeval end_t;

  struct timeval skt_t;

  struct timeval ske_t;

  gettimeofday(&start_t,0L);

  cudaMemcpy(d_box_gpu,box_cpu,dim_cpu . box_mem,cudaMemcpyHostToDevice);

  cudaMemcpy(d_rv_v,rv_cpu_v,dim_cpu . space_elem * sizeof(float ),cudaMemcpyHostToDevice);

  cudaMemcpy(d_rv_x,rv_cpu_x,dim_cpu . space_elem * sizeof(float ),cudaMemcpyHostToDevice);

  cudaMemcpy(d_rv_y,rv_cpu_y,dim_cpu . space_elem * sizeof(float ),cudaMemcpyHostToDevice);

  cudaMemcpy(d_rv_z,rv_cpu_z,dim_cpu . space_elem * sizeof(float ),cudaMemcpyHostToDevice);

  cudaMemcpy(d_qv_gpu,qv_cpu,dim_cpu . space_elem * sizeof(float ),cudaMemcpyHostToDevice);

  cudaMemcpy(d_fv_v,fv_cpu_v,dim_cpu . space_elem * sizeof(float ),cudaMemcpyHostToDevice);

  cudaMemcpy(d_fv_x,fv_cpu_x,dim_cpu . space_elem * sizeof(float ),cudaMemcpyHostToDevice);

  cudaMemcpy(d_fv_y,fv_cpu_y,dim_cpu . space_elem * sizeof(float ),cudaMemcpyHostToDevice);

  cudaMemcpy(d_fv_z,fv_cpu_z,dim_cpu . space_elem * sizeof(float ),cudaMemcpyHostToDevice);

  gettimeofday(&skt_t,0L);

  kernel_gpu_cuda<<<blocks,threads>>>(dim_cpu,d_box_gpu,d_rv_v,d_rv_x,d_rv_y,d_rv_z,d_qv_gpu,d_fv_v,d_fv_x,d_fv_y,d_fv_z);

  checkCUDAError("Start");

  cudaThreadSynchronize();

  gettimeofday(&ske_t,0L);

  cudaMemcpy(fv_cpu_v,d_fv_v,dim_cpu . space_elem * sizeof(float ),cudaMemcpyDeviceToHost);

  cudaMemcpy(fv_cpu_x,d_fv_x,dim_cpu . space_elem * sizeof(float ),cudaMemcpyDeviceToHost);

  cudaMemcpy(fv_cpu_y,d_fv_y,dim_cpu . space_elem * sizeof(float ),cudaMemcpyDeviceToHost);

  cudaMemcpy(fv_cpu_z,d_fv_z,dim_cpu . space_elem * sizeof(float ),cudaMemcpyDeviceToHost);

  gettimeofday(&end_t,0L);
  mpf_t val_x, val_y, val_in, err;
  mpf_init2(val_x, 128);
  mpf_init2(val_y, 128);
  mpf_init2(val_in, 128);
  mpf_init2(err, 128);
  FILE* infile = fopen("fv_ref.txt", "r");
  for(int i = 0; i < dim_cpu.space_elem; i++) {
    gmp_fscanf(infile, "%Fe\n", val_in);
    mpf_set_d(val_x, fv_cpu_v[i]);
    mpf_sub(val_y, val_x, val_in);
    mpf_abs(val_x, val_y);
    mpf_div(val_x, val_x, val_in);
    if (i==0)
      mpf_set(err, val_x);
    else
      mpf_add(err, err, val_x);
    gmp_fscanf(infile, "%Fe\n", val_in);
    mpf_set_d(val_x, fv_cpu_x[i]);
    mpf_sub(val_y, val_x, val_in);
    mpf_abs(val_x, val_y);
    mpf_div(val_x, val_x, val_in);
    mpf_add(err, err, val_x);
    gmp_fscanf(infile, "%Fe\n", val_in);
    mpf_set_d(val_x, fv_cpu_y[i]);
    mpf_sub(val_y, val_x, val_in);
    mpf_abs(val_x, val_y);
    mpf_div(val_x, val_x, val_in);
    mpf_add(err, err, val_x);
    gmp_fscanf(infile, "%Fe\n", val_in);
    mpf_set_d(val_x, fv_cpu_z[i]);
    mpf_sub(val_y, val_x, val_in);
    mpf_abs(val_x, val_y);
    mpf_div(val_x, val_x, val_in);
    mpf_add(err, err, val_x);
  }
  mpf_div_ui(err, err, 4*dim_cpu.space_elem);
  fclose(infile);
  gmp_printf("error: %10.5Fe\n", err);
  int blockSize;
  int minGridSize = dim_cpu.number_boxes;
  cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernel_gpu_cuda, 0, 0);
  printf("block: %d\n", blockSize);

  ((std::cout<<"time: ") << (end_t . tv_sec + end_t . tv_usec * 1e-6 - (start_t . tv_sec + start_t . tv_usec * 1e-6)))<<"\n";

  ((std::cout<<"kernel: ") << ((ske_t . tv_sec - skt_t . tv_sec) + (ske_t . tv_usec - skt_t . tv_usec) * 1e-6)) << endl;

  cudaFree(d_rv_v);

  cudaFree(d_rv_x);

  cudaFree(d_rv_y);

  cudaFree(d_rv_z);

  cudaFree(d_qv_gpu);

  cudaFree(d_fv_v);

  cudaFree(d_fv_x);

  cudaFree(d_fv_y);

  cudaFree(d_fv_z);

  cudaFree(d_box_gpu);

  free(rv_cpu_v);

  free(rv_cpu_x);

  free(rv_cpu_y);

  free(rv_cpu_z);

  free(qv_cpu);

  free(fv_cpu_v);

  free(fv_cpu_x);

  free(fv_cpu_y);

  free(fv_cpu_z);

  free(box_cpu);

  return 0;

}


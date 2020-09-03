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

#include <sys/time.h>

#include "cuda.h"

#include <cuda_runtime.h>

#include <string.h>

#include <math.h>

#include <mpfr.h>

#include <iostream>

using namespace std;

#ifdef RD_WG_SIZE_0_0

        #define MAXBLOCKSIZE RD_WG_SIZE_0_0

#elif defined(RD_WG_SIZE_0)

        #define MAXBLOCKSIZE RD_WG_SIZE_0

#elif defined(RD_WG_SIZE)

        #define MAXBLOCKSIZE RD_WG_SIZE

#else

        #define MAXBLOCKSIZE 512

#endif

//2D defines. Go from specific to general                                                

#ifdef RD_WG_SIZE_1_0

        #define BLOCK_SIZE_XY RD_WG_SIZE_1_0

#elif defined(RD_WG_SIZE_1)

        #define BLOCK_SIZE_XY RD_WG_SIZE_1

#elif defined(RD_WG_SIZE)

        #define BLOCK_SIZE_XY RD_WG_SIZE

#else

        #define BLOCK_SIZE_XY 4

#endif

FILE *fp;

unsigned int totalKernelTime = 0;

// create both matrix and right hand side, Ke Wang 2013/08/12 11:51:06

void create_matrix(float *trix,int size)

{

  int i;

  int j;

  double lamda = - 0.01;

  double cof[2 * size - 1];

  double coe_i = 0.0;

  for (i = 0; i < size; i++) {

    coe_i = 10 * exp(lamda * i);

    j = size - 1 + i;

    cof[j] = coe_i;

    j = size - 1 - i;

    cof[j] = coe_i;

  }

  for (i = 0; i < size; i++) {

    for (j = 0; j < size; j++) {

      trix[i * size + j] = cof[size - 1 - i + j];

    }

  }

}

void checkCUDAError(const char *msg)

{

  cudaError_t err = cudaGetLastError();

  if (cudaSuccess != err) {

    fprintf(stderr,"Cuda error: %s: %s.\n",msg,(cudaGetErrorString(err)));

    exit(1);

  }

}

/*-------------------------------------------------------

 ** Pay attention to the index.  Index i give the range

 ** which starts from 0 to range-1.  The real values of

 ** the index should be adjust and related with the value

 **-------------------------------------------------------

 */

__global__ void Fan1(float *c_m,float *c_a,int Size,int t)

{

  if (threadIdx . x + blockIdx . x * blockDim . x >= (Size - 1 - t)) 

    return ;

  c_m[Size * (blockDim . x * blockIdx . x + threadIdx . x + t + 1) + t] = c_a[Size * (blockDim . x * blockIdx . x + threadIdx . x + t + 1) + t] / c_a[Size * t + t];

}

/*-------------------------------------------------------

 **-------------------------------------------------------

 */

__global__ void Fan2(float *c_m,float *c_a,float *c_b,int Size,int j1,int t)

{

  if (threadIdx . x + blockIdx . x * blockDim . x >= (Size - 1 - t)) 

    return ;

  if (threadIdx . y + blockIdx . y * blockDim . y >= (Size - t)) 

    return ;

  int xidx = (blockIdx . x * blockDim . x + threadIdx . x);

  int yidx = (blockIdx . y * blockDim . y + threadIdx . y);

  c_a[Size * (xidx + 1 + t) + (yidx + t)] =(  c_a[Size * (xidx + 1 + t) + (yidx + t)]  -  c_m[Size * (xidx + 1 + t) + t] * c_a[Size * t + (yidx + t)]);

  if (yidx == 0) {

    c_b[xidx + 1 + t] =(    c_b[xidx + 1 + t]  -  c_m[Size * (xidx + 1 + t) + (yidx + t)] * c_b[t]);

  }

}

/*------------------------------------------------------

 ** ForwardSub() -- Forward substitution of Gaussian

 ** elimination.

 **------------------------------------------------------

 */

/*------------------------------------------------------

 ** BackSub() -- Backward substitution

 **------------------------------------------------------

 */

int main(int argc,char *argv[])

{

  printf("WG size of kernel 1 = %d, WG size of kernel 2= %d X %d\n",512,4,4);

  int verbose = 0;

  int i;

  int j;

  int t;

  char flag;

  if (argc < 2) {

    printf("Usage: gaussian -f filename / -s size [-q]\n\n");

    printf("-q (quiet) suppresses printing the matrix and result values.\n");

    printf("-f (filename) path of input file\n");

    printf("-s (size) size of matrix. Create matrix and rhs in this program \n");

    printf("The first line of the file contains the dimension of the matrix, n.");

    printf("The second line of the file is a newline.\n");

    printf("The next n lines contain n tab separated values for the matrix.");

    printf("The next line of the file is a newline.\n");

    printf("The next line of the file is a 1xn vector with tab separated values.\n");

    printf("The next line of the file is a newline. (optional)\n");

    printf("The final line of the file is the pre-computed solution. (optional)\n");

    printf("Example: matrix4.txt:\n");

    printf("4\n");

    printf("\n");

    printf("-0.6\t-0.5\t0.7\t0.3\n");

    printf("-0.3\t-0.9\t0.3\t0.7\n");

    printf("-0.4\t-0.5\t-0.3\t-0.8\n");

    printf("0.0\t-0.1\t0.2\t0.9\n");

    printf("\n");

    printf("-0.85\t-0.68\t0.24\t-0.53\n");

    printf("\n");

    printf("0.7\t0.0\t-0.4\t-0.5\n");

    exit(0);

  }

  int Size;

  for (i = 1; i < argc; i++) {

// flag

    if (argv[i][0] == '-') {

      flag = argv[i][1];

      switch(flag){

// platform

        case 's':

        i++;

        Size = atoi(argv[i]);

        printf("Create matrix internally in parse, size = %d \n",Size);

        break; 

      }

    }

  }

  float *ha = new float [Size * Size];

  create_matrix(ha,Size);

  float *hb = new float [Size];

  for (j = 0; j < Size; j++) 

    hb[j] = 1.0;

  float *hm = new float [Size * Size];

  float *finalVec = new float [Size];

//InitProblemOnce(filename);

  for (i = 0; i < Size * Size; i++) 

    hm[i] = ((float )0.0);

//begin timing

  struct timeval start_t;

  struct timeval end_t;

  struct timeval skt_t;

  struct timeval ske_t;

  struct timeval sht_t;

  struct timeval she_t;

  gettimeofday(&start_t,0L);

  float *cuda_m;

  float *cuda_a;

  float *cuda_b;

// allocate memory on GPU

  cudaMalloc((void **)(&cuda_m),(Size * Size) * sizeof(float ));

  cudaMalloc((void **)(&cuda_a),(Size * Size) * sizeof(float ));

  cudaMalloc((void **)(&cuda_b),Size * sizeof(float ));

// copy memory to GPU

  cudaMemcpy(cuda_m,hm,(Size * Size) * sizeof(float ),cudaMemcpyHostToDevice);

  cudaMemcpy(cuda_a,ha,(Size * Size) * sizeof(float ),cudaMemcpyHostToDevice);

  cudaMemcpy(cuda_b,hb,Size * sizeof(float ),cudaMemcpyHostToDevice);

  int block_size;

  int grid_size;

  block_size = 512;

  grid_size = Size / block_size + ((!(Size % block_size)?0 : 1));

//printf("1d grid size: %d\n",grid_size);

//dim3 dimGrid( (N/dimBlock.x) + (!(N%dimBlock.x)?0:1) );

  int blockSize2d;

  int gridSize2d;

  blockSize2d = 4;

  gridSize2d = Size / blockSize2d + (!(((Size % blockSize2d)?0 : 1)));

  ::dim3 dimBlockXY(blockSize2d,blockSize2d);

  ::dim3 dimGridXY(gridSize2d,gridSize2d);

  ::dim3 dimBlock(block_size);

  ::dim3 dimGrid(grid_size);

  gettimeofday(&skt_t,0L);

  for (t = 0; t < Size - 1; t++) {

    Fan1<<<dimGrid,dimBlock>>>(cuda_m,cuda_a,Size,t);

    cudaThreadSynchronize();

    Fan2<<<dimGridXY,dimBlockXY>>>(cuda_m,cuda_a,cuda_b,Size,Size - t,t);

    cudaThreadSynchronize();

    checkCUDAError("Fan2");

  }

  gettimeofday(&ske_t,0L);

// copy memory back to CPU

  cudaMemcpy(hm,cuda_m,(Size * Size) * sizeof(float ),cudaMemcpyDeviceToHost);

  cudaMemcpy(ha,cuda_a,(Size * Size) * sizeof(float ),cudaMemcpyDeviceToHost);

  cudaMemcpy(hb,cuda_b,Size * sizeof(float ),cudaMemcpyDeviceToHost);

//BackSub();

// create a new vector to hold the final answer

// solve "bottom up"

  gettimeofday(&sht_t,0L);

  for (i = 0; i < Size; i++) {

    finalVec[Size - i - 1] = hb[Size - i - 1];

    for (j = 0; j < i; j++) {

      finalVec[Size - i - 1] =      finalVec[Size - i - 1]  -  ha[Size * (Size - i - 1) + (Size - j - 1)] * finalVec[Size - j - 1];

    }

    finalVec[Size - i - 1] = finalVec[Size - i - 1] / ha[Size * (Size - i - 1) + (Size - i - 1)];

  }

  gettimeofday(&she_t,0L);

  gettimeofday(&end_t,0L);

  mpf_t val_x, val_y, val_in, err;
  mpf_init2(val_x, 128);
  mpf_init2(val_y, 128);
  mpf_init2(val_in, 128);
  mpf_init2(err, 128);
  FILE* infile = fopen("fv_ref.txt", "r");
  for(int i = 0; i < Size; i++) {
    gmp_fscanf(infile, "%Fe\n", val_in);
    mpf_set_d(val_x, finalVec[i]);
    mpf_sub(val_x, val_x, val_in);
    mpf_abs(val_y, val_x);
    mpf_div(val_x, val_y, val_in);
    if (i==0)
      mpf_set(err, val_x);
    else
      mpf_add(err, err, val_x);

  }
  mpf_div_ui(err, err, Size);
  fclose(infile);
  gmp_printf("error: %10.5Fe\n", err);

  ((std::cout<<"time: ") << ((end_t . tv_sec - start_t . tv_sec) + (end_t . tv_usec - start_t . tv_usec) * 1e-6)) << endl;

  ((std::cout<<"kernel: ") << ((ske_t . tv_sec - skt_t . tv_sec) + (ske_t . tv_usec - skt_t . tv_usec) * 1e-6 + (she_t . tv_sec - sht_t . tv_sec) + (she_t . tv_usec - sht_t . tv_usec) * 1e-6)) << endl;

//if (verbose) {

//    printf("The final solution is: \n");

//    PrintAry(finalVec,Size);

//}

  cudaFree(cuda_m);

  cudaFree(cuda_a);

  cudaFree(cuda_b);

  free(hm);

  free(ha);

  free(hb);

}


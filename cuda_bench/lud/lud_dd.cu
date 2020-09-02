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

#include <cuda.h>

#include <cuda_runtime.h>

#include <math.h>

#include <stdio.h>

#include <unistd.h>

#include <getopt.h>

#include <stdlib.h>

#include <assert.h>

#include <sys/time.h>

#include <mpfr.h>

#include <iostream>

using namespace std;

#ifdef RD_WG_SIZE_0_0

        #define BLOCK_SIZE RD_WG_SIZE_0_0

#elif defined(RD_WG_SIZE_0)

        #define BLOCK_SIZE RD_WG_SIZE_0

#elif defined(RD_WG_SIZE)

        #define BLOCK_SIZE RD_WG_SIZE

#else

        #define BLOCK_SIZE 16

#endif

#define MIN(i,j) ((i)<(j) ? (i) : (j))

static struct option long_options[] = {/* Need explicit braces: is this where we insert the class name? */ {("input"), (1), (0L), ('i')}, /* Need explicit braces: is this where we insert the class name? */ {("size"), (1), (0L), ('s')}, /* Need explicit braces: is this where we insert the class name? */ {("verify"), (0), (0L), ('v')}, /* Need explicit braces: is this where we insert the class name? */ {(0), (0), (0), (0)}};

__global__ void lud_diagonal(float *m,int matrix_dim,int offset)

{

  int i;

  int j;

  __shared__ float shadow[27][27];

  int array_offset = offset * matrix_dim + offset;

  for (i = 0; i < 27; i++) {

    shadow[i][threadIdx . x] =( m[array_offset + threadIdx . x]);

    array_offset =    array_offset  +  matrix_dim;

  }

  __syncthreads();

  for (i = 0; i < 27 - 1; i++) {

    if (threadIdx . x > i) {

      for (j = 0; j < i; j++) 

        shadow[threadIdx . x][i] =        shadow[threadIdx . x][i]  -  shadow[threadIdx . x][j] * shadow[j][i];

      shadow[threadIdx . x][i] =      shadow[threadIdx . x][i]  /  shadow[i][i];

    }

    __syncthreads();

    if (threadIdx . x > i) {

      for (j = 0; j < i + 1; j++) 

        shadow[i + 1][threadIdx . x] =        shadow[i + 1][threadIdx . x]  -  shadow[i + 1][j] * shadow[j][threadIdx . x];

    }

    __syncthreads();

  }

  array_offset = (offset + 1) * matrix_dim + offset;

  for (i = 1; i < 27; i++) {

    m[array_offset + threadIdx . x] = shadow[i][threadIdx . x];

    array_offset =    array_offset  +  matrix_dim;

  }

}

__global__ void lud_perimeter(float *m,int matrix_dim,int offset)

{

  __shared__ float dia[27][27];

  __shared__ float peri_row[27][27];

  __shared__ float peri_col[27][27];

  int i;

  int j;

  int array_offset;

  int idx;

  if (threadIdx . x < 27) {

    idx = threadIdx . x;

    array_offset = offset * matrix_dim + offset;

    for (i = 0; i < 27 / 2; i++) {

      dia[i][idx] =( m[array_offset + idx]);

      array_offset =      array_offset  +  matrix_dim;

    }

    array_offset = offset * matrix_dim + offset;

    for (i = 0; i < 27; i++) {

      peri_row[i][idx] =( m[array_offset + (blockIdx . x + 1) * 27 + idx]);

      array_offset =      array_offset  +  matrix_dim;

    }

  }

   else {

    idx = (threadIdx . x - 27);

    array_offset = (offset + 27 / 2) * matrix_dim + offset;

    for (i = 27 / 2; i < 27; i++) {

      dia[i][idx] =( m[array_offset + idx]);

      array_offset =      array_offset  +  matrix_dim;

    }

    array_offset = ((offset + (blockIdx . x + 1) * 27) * matrix_dim + offset);

    for (i = 0; i < 27; i++) {

      peri_col[i][idx] =( m[array_offset + idx]);

      array_offset =      array_offset  +  matrix_dim;

    }

  }

  __syncthreads();

  if (threadIdx . x < 27) {

    idx = threadIdx . x;

    for (i = 1; i < 27; i++) {

      for (j = 0; j < i; j++) 

        peri_row[i][idx] =        peri_row[i][idx]  -  dia[i][j] * peri_row[j][idx];

    }

  }

   else {

    idx = (threadIdx . x - 27);

    for (i = 0; i < 27; i++) {

      for (j = 0; j < i; j++) 

        peri_col[idx][i] =        peri_col[idx][i]  -  peri_col[idx][j] * dia[j][i];

      peri_col[idx][i] =      peri_col[idx][i]  /  dia[i][i];

    }

  }

  __syncthreads();

//peri-row

  if (threadIdx . x < 27) {

    idx = threadIdx . x;

    array_offset = (offset + 1) * matrix_dim + offset;

    for (i = 1; i < 27; i++) {

      m[array_offset + (blockIdx . x + 1) * 27 + idx] = peri_row[i][idx];

      array_offset =      array_offset  +  matrix_dim;

    }

//peri-col

  }

   else {

    idx = (threadIdx . x - 27);

    array_offset = ((offset + (blockIdx . x + 1) * 27) * matrix_dim + offset);

    for (i = 0; i < 27; i++) {

      m[array_offset + idx] = peri_col[i][idx];

      array_offset =      array_offset  +  matrix_dim;

    }

  }

}

__global__ void lud_internal(float *m,int matrix_dim,int offset)

{

  __shared__ float peri_row[27][27];

  __shared__ float peri_col[27][27];

  int i;

  float sum;

  int global_row_id = (offset + (blockIdx . y + 1) * 27);

  int global_col_id = (offset + (blockIdx . x + 1) * 27);

  peri_row[threadIdx . y][threadIdx . x] =( m[(offset + threadIdx . y) * matrix_dim + global_col_id + threadIdx . x]);

  peri_col[threadIdx . y][threadIdx . x] =( m[(global_row_id + threadIdx . y) * matrix_dim + offset + threadIdx . x]);

  __syncthreads();

  sum = ((float )0.0);

  for (i = 0; i < 27; i++) 

    sum =    sum  +  peri_col[threadIdx . y][i] * peri_row[i][threadIdx . x];

  m[(global_row_id + threadIdx . y) * matrix_dim + global_col_id + threadIdx . x] =  m[(global_row_id + threadIdx . y) * matrix_dim + global_col_id + threadIdx . x]  -  sum;

}

void create_matrix(float *finalv,int size)

{

  int i;

  int j;

  double lamda = - 0.001;

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

      finalv[i * size + j] = cof[size - 1 - i + j];

    }

  }

}

int main(int argc,char **argv)

{

  printf("WG size of kernel = %d X %d\n",27,27);

  int matrix_dim = 32;

  int opt;

  int option_index = 0;

  int i;

  int j;

  const char *input_file = 0L;

  float *dtrix;

  struct timeval start_t;

  struct timeval end_t;

  struct timeval skt_t;

  struct timeval ske_t;

  while((opt = getopt_long(argc,argv,"::vs:i:",long_options,&option_index)) != - 1){

    switch(opt){

      case 'i':

      input_file = optarg;

      break; 

      case 's':

      matrix_dim = atoi(optarg);

      printf("Generate input matrix internally, size =%d\n",matrix_dim);

      break; 

      case '?':

      fprintf(stderr,"invalid option\n");

      break; 

      case ':':

      fprintf(stderr,"missing argument\n");

      break; 

      default:

      fprintf(stderr,"Usage: %s [-v] [-s matrix_size|-i input_file]\n",argv[0]);

      exit(1);

    }

  }

  if (optind < argc || optind == 1) {

    fprintf(stderr,"Usage: %s [-v] [-s matrix_size|-i input_file]\n",argv[0]);

    exit(1);

  }

  float *fv = new float [matrix_dim * matrix_dim];

  if (input_file) {

    printf("Reading matrix from file %s\n",input_file);

    FILE *fp = 0L;

    float input;

    fp = fopen(input_file,"rb");

    fscanf(fp,"%d\n",&matrix_dim);

    for (i = 0; i < matrix_dim; i++) {

      for (j = 0; j < matrix_dim; j++) {

        fscanf(fp,"%f ",&input);

        fv[i * matrix_dim + j] = ((float )input);

      }

    }

    fclose(fp);

    printf("matrix dim: %d\n",matrix_dim);

  }

   else if (matrix_dim) {

    printf("Creating matrix internally size=%d\n",matrix_dim);

    create_matrix(fv,matrix_dim);

  }

   else {

    printf("No input file specified!\n");

    exit(1);

  }

  cudaMalloc((void **)(&dtrix),(matrix_dim * matrix_dim) * sizeof(float ));

  gettimeofday(&start_t,0L);

  cudaMemcpy(dtrix,fv,(matrix_dim * matrix_dim) * sizeof(float ),cudaMemcpyHostToDevice);

  gettimeofday(&skt_t,0L);

  i = 0;

  ::dim3 dimBlock(27,27);

  for (i = 0; i < matrix_dim - 27; i = i + 27) {

    lud_diagonal<<<1,27>>>(dtrix,matrix_dim,i);

    lud_perimeter<<<((matrix_dim - i) / 27 - 1),(27 * 2)>>>(dtrix,matrix_dim,i);

    ::dim3 dimGrid(((matrix_dim - i) / 27 - 1),((matrix_dim - i) / 27 - 1));

    lud_internal<<<dimGrid,dimBlock>>>(dtrix,matrix_dim,i);

  }

  lud_diagonal<<<1,27>>>(dtrix,matrix_dim,i);

  cudaThreadSynchronize();

  gettimeofday(&ske_t,0L);

  cudaMemcpy(fv,dtrix,(matrix_dim * matrix_dim) * sizeof(float ),cudaMemcpyDeviceToHost);

  gettimeofday(&end_t,0L);

  mpf_t val_x, val_y, val_in, err;
  mpf_init2(val_x, 128);
  mpf_init2(val_y, 128);
  mpf_init2(val_in, 128);
  mpf_init2(err, 128);
  FILE* infile = fopen("m_ref.txt", "r");
  for(int i = 0; i < matrix_dim*matrix_dim; i++) {
    gmp_fscanf(infile, "%Fe\n", val_in);
    mpf_set_d(val_x, fv[i]);
    mpf_sub(val_x, val_x, val_in);
    mpf_abs(val_y, val_x);
    mpf_div(val_x, val_y, val_in);
    if (i==0)
      mpf_set(err, val_x);
    else
      mpf_add(err, err, val_x);

  }
  mpf_div_ui(err, err, matrix_dim*matrix_dim);
  fclose(infile);
  gmp_printf("error: %10.5Fe\n", err);
  int blockSize;
  int minGridSize;
  int gridSize;
  cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, lud_diagonal, 0, 0);
  blockSize = sqrt(blockSize);
  printf("block: %d\n", blockSize);


  ((std::cout<<"time: ") << (end_t . tv_sec + end_t . tv_usec * 1e-6 - (start_t . tv_sec + start_t . tv_usec * 1e-6)))<<"\n";

  ((std::cout<<"kernel: ") << ((ske_t . tv_sec - skt_t . tv_sec) + (ske_t . tv_usec - skt_t . tv_usec) * 1e-6)) << endl;

  cudaFree(dtrix);

  free(fv);

  return 0;

}


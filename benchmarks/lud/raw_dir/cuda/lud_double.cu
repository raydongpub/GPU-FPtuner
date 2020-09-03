/*
 * =====================================================================================
 *
 *       Filename:  lud.cu
 *
 *    Description:  The main wrapper for the suite
 *
 *        Version:  1.0
 *        Created:  10/22/2009 08:40:34 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Liang Wang (lw2aw), lw2aw@virginia.edu
 *        Company:  CS@UVa
 *
 * =====================================================================================
 */

#include <cuda.h>
#include <stdio.h>
#include <unistd.h>
#include <getopt.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/time.h>
#include <mpfr.h>

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

#define GET_RAND_FP ( (double)rand() /   \
                     ((double)(RAND_MAX)+(double)(1)) )

static int do_verify = 0;

static struct option long_options[] = {
  /* name, has_arg, flag, val */
  {"input", 1, NULL, 'i'},
  {"size", 1, NULL, 's'},
  {"verify", 0, NULL, 'v'},
  {0,0,0,0}
};

__global__ void 
lud_diagonal(double *m, int matrix_dim, int offset)
{
  int i,j;
  __shared__ double shadow[BLOCK_SIZE][BLOCK_SIZE];

  int array_offset = offset*matrix_dim+offset;
  for(i=0; i < BLOCK_SIZE; i++){
    shadow[i][threadIdx.x]=m[array_offset+threadIdx.x];
    array_offset += matrix_dim;
  }
  __syncthreads();
  for(i=0; i < BLOCK_SIZE-1; i++) {

    if (threadIdx.x>i){
      for(j=0; j < i; j++)
        shadow[threadIdx.x][i] -= shadow[threadIdx.x][j]*shadow[j][i];
      shadow[threadIdx.x][i] /= shadow[i][i];
    }

    __syncthreads();
    if (threadIdx.x>i){

      for(j=0; j < i+1; j++)
        shadow[i+1][threadIdx.x] -= shadow[i+1][j]*shadow[j][threadIdx.x];
    }
    __syncthreads();
  }

  /* 
     The first row is not modified, it
     is no need to write it back to the
     global memory

   */
  array_offset = (offset+1)*matrix_dim+offset;
  for(i=1; i < BLOCK_SIZE; i++){
    m[array_offset+threadIdx.x]=shadow[i][threadIdx.x];
    array_offset += matrix_dim;
  }
}

__global__ void
lud_perimeter(double *m, int matrix_dim, int offset)
{
  __shared__ double dia[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ double peri_row[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ double peri_col[BLOCK_SIZE][BLOCK_SIZE];

  int i,j, array_offset;
  int idx;

  if (threadIdx.x < BLOCK_SIZE) {
    idx = threadIdx.x;
    
    array_offset = offset*matrix_dim+offset;
    for (i=0; i < BLOCK_SIZE/2; i++){
      dia[i][idx]=m[array_offset+idx];
      array_offset += matrix_dim;
    }
    
    array_offset = offset*matrix_dim+offset;
    for (i=0; i < BLOCK_SIZE; i++) {
      peri_row[i][idx]=m[array_offset+(blockIdx.x+1)*BLOCK_SIZE+idx];
      array_offset += matrix_dim;
    }

  } else {
    idx = threadIdx.x-BLOCK_SIZE;
    
    array_offset = (offset+BLOCK_SIZE/2)*matrix_dim+offset;
    for (i=BLOCK_SIZE/2; i < BLOCK_SIZE; i++){
      dia[i][idx]=m[array_offset+idx];
      array_offset += matrix_dim;
    }
    
    array_offset = (offset+(blockIdx.x+1)*BLOCK_SIZE)*matrix_dim+offset;
    for (i=0; i < BLOCK_SIZE; i++) {
      peri_col[i][idx] = m[array_offset+idx];
      array_offset += matrix_dim;
    }
  
  }
  __syncthreads();

  if (threadIdx.x < BLOCK_SIZE) { //peri-row
    idx=threadIdx.x;
    for(i=1; i < BLOCK_SIZE; i++){
      for (j=0; j < i; j++)
        peri_row[i][idx]-=dia[i][j]*peri_row[j][idx];
    }
  } else { //peri-col
    idx=threadIdx.x - BLOCK_SIZE;
    for(i=0; i < BLOCK_SIZE; i++){
      for(j=0; j < i; j++)
        peri_col[idx][i]-=peri_col[idx][j]*dia[j][i];
      peri_col[idx][i] /= dia[i][i];
    }
  }

  __syncthreads();
    
  if (threadIdx.x < BLOCK_SIZE) { //peri-row
    idx=threadIdx.x;
    array_offset = (offset+1)*matrix_dim+offset;
    for(i=1; i < BLOCK_SIZE; i++){
      m[array_offset+(blockIdx.x+1)*BLOCK_SIZE+idx] = peri_row[i][idx];
      array_offset += matrix_dim;
    }
  } else { //peri-col
    idx=threadIdx.x - BLOCK_SIZE;
    array_offset = (offset+(blockIdx.x+1)*BLOCK_SIZE)*matrix_dim+offset;
    for(i=0; i < BLOCK_SIZE; i++){
      m[array_offset+idx] =  peri_col[i][idx];
      array_offset += matrix_dim;
    }
  }

}

__global__ void
lud_internal(double *m, int matrix_dim, int offset)
{
  __shared__ double peri_row[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ double peri_col[BLOCK_SIZE][BLOCK_SIZE];

  int i;
  double sum;

  int global_row_id = offset + (blockIdx.y+1)*BLOCK_SIZE;
  int global_col_id = offset + (blockIdx.x+1)*BLOCK_SIZE;

  peri_row[threadIdx.y][threadIdx.x] = m[(offset+threadIdx.y)*matrix_dim+global_col_id+threadIdx.x];
  peri_col[threadIdx.y][threadIdx.x] = m[(global_row_id+threadIdx.y)*matrix_dim+offset+threadIdx.x];

  __syncthreads();

  sum = 0;
  for (i=0; i < BLOCK_SIZE; i++)
    sum += peri_col[threadIdx.y][i] * peri_row[i][threadIdx.x];
  m[(global_row_id+threadIdx.y)*matrix_dim+global_col_id+threadIdx.x] -= sum;


}

void create_matrix_from_random(double **mp, int size){
  double *l, *u, *m;
  int i,j,k;

  srand(time(NULL));

  l = (double*)malloc(size*size*sizeof(double));

  u = (double*)malloc(size*size*sizeof(double));

  for (i = 0; i < size; i++) {
      for (j=0; j < size; j++) {
          if (i>j) {
              l[i*size+j] = GET_RAND_FP;
          } else if (i == j) {
              l[i*size+j] = 1;
          } else {
              l[i*size+j] = 0;
          }
      }
  }

  for (j=0; j < size; j++) {
      for (i=0; i < size; i++) {
          if (i>j) {
              u[j*size+i] = 0;
          }else {
              u[j*size+i] = GET_RAND_FP; 
          }
      }
  }

  for (i=0; i < size; i++) {
      for (j=0; j < size; j++) {
          for (k=0; k <= MIN(i,j); k++)
            m[i*size+j] = l[i*size+k] * u[j*size+k];
      }
  }

  free(l);
  free(u);

  *mp = m;
}

void create_matrix(double **mp, int size){
  double *m;
  int i,j;
  double lamda = -0.001;
  double coe[2*size-1];
  double coe_i =0.0;

  for (i=0; i < size; i++)
    {
      coe_i = 10*exp(lamda*i); 
      j=size-1+i;     
      coe[j]=coe_i;
      j=size-1-i;     
      coe[j]=coe_i;
    }

  m = (double*) malloc(sizeof(double)*size*size);

  for (i=0; i < size; i++) {
      for (j=0; j < size; j++) {
	m[i*size+j]=coe[size-1-i+j];
      }
  }

  *mp = m;
}

int main ( int argc, char *argv[] ) {
  printf("WG size of kernel = %d X %d\n", BLOCK_SIZE, BLOCK_SIZE);

  int matrix_dim = 32; /* default matrix_dim */
  int opt, option_index=0;
  int i, j;
  const char *input_file = NULL;
  double *m, *d_m;

  while ((opt = getopt_long(argc, argv, "::vs:i:", 
                            long_options, &option_index)) != -1 ) {
    switch(opt){
    case 'i':
      input_file = optarg;
      break;
    case 'v':
      do_verify = 1;
      break;
    case 's':
      matrix_dim = atoi(optarg);
      printf("Generate input matrix internally, size =%d\n", matrix_dim);
      // fprintf(stderr, "Currently not supported, use -i instead\n");
      // fprintf(stderr, "Usage: %s [-v] [-s matrix_size|-i input_file]\n", argv[0]);
      // exit(EXIT_FAILURE);
      break;
    case '?':
      fprintf(stderr, "invalid option\n");
      break;
    case ':':
      fprintf(stderr, "missing argument\n");
      break;
    default:
      fprintf(stderr, "Usage: %s [-v] [-s matrix_size|-i input_file]\n",
	      argv[0]);
      exit(EXIT_FAILURE);
    }
  }
  
  if ( (optind < argc) || (optind == 1)) {
    fprintf(stderr, "Usage: %s [-v] [-s matrix_size|-i input_file]\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  if (input_file) {
    printf("Reading matrix from file %s\n", input_file);
    FILE *fp = NULL;
    float input;
     fp = fopen(input_file, "rb");
     fscanf(fp, "%d\n", &matrix_dim);
     m = new double[matrix_dim*matrix_dim];
     //m = (double*) malloc(sizeof(double)*size*size);
     for (i=0; i < matrix_dim; i++) {
         for (j=0; j < matrix_dim; j++) {
             fscanf(fp, "%f ", &input);
             *(m+i*matrix_dim+j) = (double)input; 
         }
     }
     fclose(fp);
     printf("matrix dim: %d\n", matrix_dim);
    //create_matrix_from_file(m, input_file, &matrix_dim);
  } 
  else if (matrix_dim) {
    printf("Creating matrix internally size=%d\n", matrix_dim);
    create_matrix(&m, matrix_dim);
  }
  else {
    printf("No input file specified!\n");
    exit(EXIT_FAILURE);
  }

  cudaMalloc((void**)&d_m, 
             matrix_dim*matrix_dim*sizeof(double));

  struct timeval time_start;
  struct timeval time_end;
  gettimeofday(&time_start, NULL);	
  /* beginning of timing point */
  cudaMemcpy(d_m, m, matrix_dim*matrix_dim*sizeof(double), 
	     cudaMemcpyHostToDevice);

  //lud_cuda(d_m, matrix_dim);
    i = 0;
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    //double *m_debug = (double*)malloc(matrix_dim*matrix_dim*sizeof(double));
    for (i=0; i < matrix_dim-BLOCK_SIZE; i += BLOCK_SIZE) {
        lud_diagonal<<<1, BLOCK_SIZE>>>(d_m, matrix_dim, i);
        lud_perimeter<<<(matrix_dim-i)/BLOCK_SIZE-1, BLOCK_SIZE*2>>>(d_m, matrix_dim, i);
        dim3 dimGrid((matrix_dim-i)/BLOCK_SIZE-1, (matrix_dim-i)/BLOCK_SIZE-1);
        lud_internal<<<dimGrid, dimBlock>>>(d_m, matrix_dim, i); 
    }
    lud_diagonal<<<1,BLOCK_SIZE>>>(d_m, matrix_dim, i);

  cudaMemcpy(m, d_m, matrix_dim*matrix_dim*sizeof(double), 
	     cudaMemcpyDeviceToHost);

  /* end of timing point */

  gettimeofday(&time_end, NULL);

  mpf_t val_x, val_y, val_in, err;
  mpf_init2(val_x, 128);
  mpf_init2(val_y, 128);
  mpf_init2(val_in, 128);
  mpf_init2(err, 128);
  FILE *infile = fopen("m_ref.txt", "r");
  for (i=0; i<matrix_dim*matrix_dim; i++) {
    gmp_fscanf(infile, "%Fe\n", val_in);
    mpf_set_d(val_x, m[i]);
    mpf_sub(val_y, val_x, val_in);
    mpf_abs(val_x, val_y);
    mpf_div(val_x, val_x, val_in);
    if (i==0)
      mpf_set(err, val_x);
    else
      mpf_add(err, err, val_x);
  }
  mpf_div_ui(err, err, matrix_dim*matrix_dim);
  gmp_printf("error: %.80Ff\n", err);

  double time_total = (time_end.tv_sec + time_end.tv_usec*1e-6) - (time_start.tv_sec + time_start.tv_usec*1e-6);
  printf("time: %f\n", time_total);
  //if (do_verify){
  //  printf("After LUD\n");
  //  // print_matrix(m, matrix_dim);
  //  printf(">>>Verify<<<<\n");
  //  lud_verify(mm, m, matrix_dim); 
  //  free(mm);
  //}

  cudaFree(d_m);
  free(m);

  return EXIT_SUCCESS;
}				/* ----------  end of function main  ---------- */

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "cuda.h"
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
void
create_matrix(double *m, int size){
  int i,j;
  double lamda = -0.01;
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


  for (i=0; i < size; i++) {
      for (j=0; j < size; j++) {
	m[i*size+j]=coe[size-1-i+j];
      }
  }


}

void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) 
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, 
                                  cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }                         
}

/*-------------------------------------------------------
 ** Pay attention to the index.  Index i give the range
 ** which starts from 0 to range-1.  The real values of
 ** the index should be adjust and related with the value
 **-------------------------------------------------------
 */
__global__ void Fan1(double *m_cuda, double *a_cuda, int Size, int t)
{   

	if(threadIdx.x + blockIdx.x * blockDim.x >= Size-1-t) return;
	m_cuda[Size*(blockDim.x*blockIdx.x+threadIdx.x+t+1)+t] = a_cuda[Size*(blockDim.x*blockIdx.x+threadIdx.x+t+1)+t] / a_cuda[Size*t+t];
	//*(m_cuda+Size*(blockDim.x*blockIdx.x+threadIdx.x+t+1)+t) = *(a_cuda+Size*(blockDim.x*blockIdx.x+threadIdx.x+t+1)+t) / *(a_cuda+Size*t+t);
}

/*-------------------------------------------------------
 **-------------------------------------------------------
 */ 

__global__ void Fan2(double *m_cuda, double *a_cuda, double *b_cuda,int Size, int j1, int t)
{
	if(threadIdx.x + blockIdx.x * blockDim.x >= Size-1-t) return;
	if(threadIdx.y + blockIdx.y * blockDim.y >= Size-t) return;
	
	int xidx = blockIdx.x * blockDim.x + threadIdx.x;
	int yidx = blockIdx.y * blockDim.y + threadIdx.y;
	
	a_cuda[Size*(xidx+1+t)+(yidx+t)] -= m_cuda[Size*(xidx+1+t)+t] * a_cuda[Size*t+(yidx+t)];
	if(yidx == 0){
		b_cuda[xidx+1+t] -= m_cuda[Size*(xidx+1+t)+(yidx+t)] * b_cuda[t];
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

int main(int argc, char *argv[])
{
  printf("WG size of kernel 1 = %d, WG size of kernel 2= %d X %d\n", MAXBLOCKSIZE, BLOCK_SIZE_XY, BLOCK_SIZE_XY);
    int verbose = 0;
    int i, j, t;
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
        printf("-0.6	-0.5	0.7	0.3\n");
        printf("-0.3	-0.9	0.3	0.7\n");
        printf("-0.4	-0.5	-0.3	-0.8\n");	
        printf("0.0	-0.1	0.2	0.9\n");
        printf("\n");
        printf("-0.85	-0.68	0.24	-0.53\n");	
        printf("\n");
        printf("0.7	0.0	-0.4	-0.5\n");
        exit(0);
    }
    

    int Size;
    double *a, *b, *finalVec;
    double *m;
    for(i=1;i<argc;i++) {
      if (argv[i][0]=='-') {// flag
        flag = argv[i][1];
          switch (flag) {
            case 's': // platform
              i++;
              Size = atoi(argv[i]);
	      printf("Create matrix internally in parse, size = %d \n", Size);

	      a = new double[Size * Size];
	      create_matrix(a, Size);

	      b = new double[Size];
	      for (j =0; j< Size; j++)
	    	b[j]=1.0;

	      m = new double[Size * Size];
              break;
	  }
      }
    }
	  finalVec = new double[Size];

    //InitProblemOnce(filename);
  	for (i=0; i<Size*Size; i++)
  			*(m+i) = (double)0.0;

    //begin timing
    struct timeval start_t;
    struct timeval end_t;
    struct timeval skt_t;
    struct timeval ske_t;
    struct timeval sht_t;
    struct timeval she_t;
    gettimeofday(&start_t,0L);
    
        double *m_cuda,*a_cuda,*b_cuda;

	// allocate memory on GPU
	cudaMalloc((void **) &m_cuda, Size * Size * sizeof(double));
	cudaMalloc((void **) &a_cuda, Size * Size * sizeof(double));
	cudaMalloc((void **) &b_cuda, Size * sizeof(double));	
	// copy memory to GPU
	cudaMemcpy(m_cuda, m, Size * Size * sizeof(double),cudaMemcpyHostToDevice );
	cudaMemcpy(a_cuda, a, Size * Size * sizeof(double),cudaMemcpyHostToDevice );
	cudaMemcpy(b_cuda, b, Size * sizeof(double),cudaMemcpyHostToDevice );
	
	int block_size,grid_size;
	block_size = MAXBLOCKSIZE;
	grid_size = (Size/block_size) + (!(Size%block_size)? 0:1);
	//printf("1d grid size: %d\n",grid_size);
	dim3 dimBlock(block_size);
	dim3 dimGrid(grid_size);
	//dim3 dimGrid( (N/dimBlock.x) + (!(N%dimBlock.x)?0:1) );
	int blockSize2d, gridSize2d;
	blockSize2d = BLOCK_SIZE_XY;
	gridSize2d = (Size/blockSize2d) + (!(Size%blockSize2d?0:1)); 
	dim3 dimBlockXY(blockSize2d,blockSize2d);
	dim3 dimGridXY(gridSize2d,gridSize2d);


  gettimeofday(&skt_t,0L);
	for (t=0; t<(Size-1); t++) {
		Fan1<<<dimGrid,dimBlock>>>(m_cuda,a_cuda,Size,t);
		cudaThreadSynchronize();
		Fan2<<<dimGridXY,dimBlockXY>>>(m_cuda,a_cuda,b_cuda,Size,Size-t,t);
		cudaThreadSynchronize();
		checkCUDAError("Fan2");
	}
  gettimeofday(&ske_t,0L);
	// copy memory back to CPU
	cudaMemcpy(m, m_cuda, Size * Size * sizeof(double),cudaMemcpyDeviceToHost );
	cudaMemcpy(a, a_cuda, Size * Size * sizeof(double),cudaMemcpyDeviceToHost );
	cudaMemcpy(b, b_cuda, Size * sizeof(double),cudaMemcpyDeviceToHost );

    //BackSub();
	// create a new vector to hold the final answer
	// solve "bottom up"
  gettimeofday(&sht_t,0L);
	for(i=0;i<Size;i++){
		finalVec[Size-i-1]=b[Size-i-1];
		for(j=0;j<i;j++)
		{
			finalVec[Size-i-1]-=a[Size*(Size-i-1)+(Size-j-1)] * finalVec[Size-j-1];
		}
		finalVec[Size-i-1]=finalVec[Size-i-1]/ a[Size*(Size-i-1)+(Size-i-1)];
	}
  gettimeofday(&she_t,0L);

  gettimeofday(&end_t,0L);


    ((std::cout<<"time: ") << ((end_t . tv_sec - start_t . tv_sec) + (end_t . tv_usec - start_t . tv_usec) * 1e-6)) << endl;
    ((std::cout<<"kernel: ") << ((ske_t . tv_sec - skt_t . tv_sec) + (ske_t . tv_usec - skt_t . tv_usec) * 1e-6 + (she_t . tv_sec - sht_t . tv_sec) + (she_t . tv_usec - sht_t . tv_usec) * 1e-6)) << endl;
    //if (verbose) {
    //    printf("The final solution is: \n");
    //    PrintAry(finalVec,Size);
    //}
    
	cudaFree(m_cuda);
	cudaFree(a_cuda);
	cudaFree(b_cuda);
  free(m);
  free(a);
  free(b);
}

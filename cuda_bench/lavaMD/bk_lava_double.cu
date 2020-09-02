//======================================================================================================================================================150
//	LIBRARIES
//======================================================================================================================================================150

#include <stdio.h>					// (in path known to compiler)			needed by printf
#include <stdlib.h>					// (in path known to compiler)			needed by malloc
#include <stdbool.h>				// (in path known to compiler)			needed by true/false
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <mpfr.h>

//======================================================================================================================================================150
//	MAIN FUNCTION HEADER
//======================================================================================================================================================150
#define NUMBER_PAR_PER_BOX 100							// keep this low to allow more blocks that share shared memory to run concurrently, code does not work for larger than 110, more speedup can be achieved with larger number and no shared memory used

/* #define NUMBER_THREADS 128								// this should be roughly equal to NUMBER_PAR_PER_BOX for best performance */

// Parameterized work group size
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

typedef struct nei_str
{

	// neighbor box
	int x, y, z;
	int number;
	long offset;

} nei_str;
typedef struct box_str
{

	// home box
	int x, y, z;
	int number;
	long offset;

	// neighbor boxes
	int nn;
	nei_str nei[26];

} box_str;
typedef struct dim_str
{

	// input arguments
	int cur_arg;
	int arch_arg;
	int cores_arg;
	int boxes1d_arg;

	// system memory
	long number_boxes;
	long box_mem;
	long space_elem;

} dim_str;


int isInteger(char *str){

	if (*str == '\0'){
		return 0;
	}

	for(; *str != '\0'; str++){
		if (*str < 48 || *str > 57){	
			return 0;
		}
	}
	return 1;
}

void checkCUDAError(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if( cudaSuccess != err) {
		// fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
		printf("Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
		fflush(NULL);
		exit(EXIT_FAILURE);
	}
}
//======================================================================================================================================================150
//	KERNEL
//======================================================================================================================================================150

__constant__ double dev_par;

//#include "./kernel/kernel_gpu_cuda_wrapper.h"	// (in library path specified here)
__global__ void kernel_gpu_cuda(dim_str d_dim_gpu, box_str* d_box_gpu,
				double* d_rv_v, double* d_rv_x, double* d_rv_y, double* d_rv_z, double* d_qv_gpu, 
				double* d_fv_v, double* d_fv_x, double* d_fv_y, double* d_fv_z)
{

	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------180
	//	THREAD PARAMETERS
	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------180

	int bx = blockIdx.x;																// get current horizontal block index (0-n)
	int tx = threadIdx.x;															// get current horizontal thread index (0-n)
	// int ax = bx*NUMBER_THREADS+tx;
	// int wbx = bx;
	int wtx = tx;

	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------180
	//	DO FOR THE NUMBER OF BOXES
	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------180

	if(bx<d_dim_gpu.number_boxes){
	// while(wbx<box_indexes_counter){

		//------------------------------------------------------------------------------------------------------------------------------------------------------160
		//	Extract input parameters
		//------------------------------------------------------------------------------------------------------------------------------------------------------160

		// parameters
		double a2 = 2.0*dev_par*dev_par;

		// home box
		int first_i;
		double* rA_v;
		double* rA_x;
		double* rA_y;
		double* rA_z;

		double* fA_v;
		double* fA_x;
		double* fA_y;
		double* fA_z;
		__shared__ double rA_shared_v[100];
		__shared__ double rA_shared_x[100];
		__shared__ double rA_shared_y[100];
		__shared__ double rA_shared_z[100];
		// nei box
		int pointer;
		int k = 0;
		int first_j;
		double* rB_v;
		double* rB_x;
		double* rB_y;
		double* rB_z;
		double* qB;
		int j = 0;
		__shared__ double rB_shared_v[100];
		__shared__ double rB_shared_x[100];
		__shared__ double rB_shared_y[100];
		__shared__ double rB_shared_z[100];
		__shared__ double qB_shared[100];
		// common
		double r2;
		double u2;
		double vij;
		double fs;
		double fxij;
		double fyij;
		double fzij;

		double d_x;
		double d_y;
		double d_z;

		//----------------------------------------------------------------------------------------------------------------------------------140
		//	Setup parameters
		//----------------------------------------------------------------------------------------------------------------------------------140

		// home box - box parameters
		first_i = d_box_gpu[bx].offset;

		// home box - distance, force, charge and type parameters
		//rA = &d_rv_gpu[first_i];
		rA_v = &d_rv_v[first_i];
		rA_x = &d_rv_x[first_i];
		rA_y = &d_rv_y[first_i];
		rA_z = &d_rv_z[first_i];

		//fA = &d_fv_gpu[first_i];
		fA_v = &d_fv_v[first_i];
		fA_x = &d_fv_x[first_i];
		fA_y = &d_fv_y[first_i];
		fA_z = &d_fv_z[first_i];

		//----------------------------------------------------------------------------------------------------------------------------------140
		//	Copy to shared memory
		//----------------------------------------------------------------------------------------------------------------------------------140

		// home box - shared memory
		while(wtx<NUMBER_PAR_PER_BOX){
			//rA_shared[wtx] = rA[wtx];
			rA_shared_v[wtx] = rA_v[wtx];
			rA_shared_x[wtx] = rA_x[wtx];
			rA_shared_y[wtx] = rA_y[wtx];
			rA_shared_z[wtx] = rA_z[wtx];
			wtx = wtx + NUMBER_THREADS;
		}
		wtx = tx;

		// synchronize threads  - not needed, but just to be safe
		__syncthreads();

		//------------------------------------------------------------------------------------------------------------------------------------------------------160
		//	nei box loop
		//------------------------------------------------------------------------------------------------------------------------------------------------------160

		// loop over neiing boxes of home box
		for (k=0; k<(1+d_box_gpu[bx].nn); k++){

			if(k==0){
				pointer = bx;													// set first box to be processed to home box
			}
			else{
				pointer = d_box_gpu[bx].nei[k-1].number;							// remaining boxes are nei boxes
			}

			//----------------------------------------------------------------------------------------------------------------------------------140
			//	Setup parameters
			//----------------------------------------------------------------------------------------------------------------------------------140

			first_j = d_box_gpu[pointer].offset;

			rB_v = &d_rv_v[first_j];
			rB_x = &d_rv_x[first_j];
			rB_y = &d_rv_y[first_j];
			rB_z = &d_rv_z[first_j];
			qB = &d_qv_gpu[first_j];

			//----------------------------------------------------------------------------------------------------------------------------------140
			//	Setup parameters
			//----------------------------------------------------------------------------------------------------------------------------------140

			while(wtx<NUMBER_PAR_PER_BOX){
				rB_shared_v[wtx] = rB_v[wtx];
				rB_shared_x[wtx] = rB_x[wtx];
				rB_shared_y[wtx] = rB_y[wtx];
				rB_shared_z[wtx] = rB_z[wtx];
				qB_shared[wtx] = qB[wtx];
				wtx = wtx + NUMBER_THREADS;
			}
			wtx = tx;

			__syncthreads();

			while(wtx<NUMBER_PAR_PER_BOX){

				for (j=0; j<NUMBER_PAR_PER_BOX; j++){
          #if 1
					r2 = (double)rA_shared_v[wtx] + (double)rB_shared_v[j] - 
					     (rA_shared_x[wtx]*rB_shared_x[j] + rA_shared_y[wtx]*rB_shared_y[j] + rA_shared_z[wtx]*rB_shared_z[j]); 
					u2 = a2*r2;
					vij= exp(-u2);
					fs = 2*vij;

					d_x = rA_shared_x[wtx]  - rB_shared_x[j];
					fxij=fs*d_x;
					d_y = rA_shared_y[wtx]  - rB_shared_y[j];
					fyij=fs*d_y;
					d_z = rA_shared_z[wtx]  - rB_shared_z[j];
					fzij=fs*d_z;

#endif
					fA_v[wtx] +=  (double)((double)qB_shared[j]*vij);
					fA_x[wtx] +=  (double)((double)qB_shared[j]*fxij);
					fA_y[wtx] +=  (double)((double)qB_shared[j]*fyij);
					fA_z[wtx] +=  (double)((double)qB_shared[j]*fzij);

				}
				wtx = wtx + NUMBER_THREADS;
			}
			wtx = tx;
			__syncthreads();
		}
	}

}

//========================================================================================================================================================================================================200
//	MAIN FUNCTION
//========================================================================================================================================================================================================200
int main(	int argc, 
		char *argv [])
{

	printf("thread block size of kernel = %d \n", NUMBER_THREADS);
	// counters
	int i, j, k, l, m, n;

	// system memory
	double par_cpu;
	dim_str dim_cpu;
	box_str* box_cpu;
	int nh;

	// assing default values
	dim_cpu.boxes1d_arg = 1;

	// go through arguments
	for(dim_cpu.cur_arg=1; dim_cpu.cur_arg<argc; dim_cpu.cur_arg++){
	  // check if -boxes1d
	  if(strcmp(argv[dim_cpu.cur_arg], "-boxes1d")==0){
	    // check if value provided
	    if(argc>=dim_cpu.cur_arg+1){
	      // check if value is a number
	      if(isInteger(argv[dim_cpu.cur_arg+1])==1){
	      	dim_cpu.boxes1d_arg = atoi(argv[dim_cpu.cur_arg+1]);
	      	if(dim_cpu.boxes1d_arg<0){
	      		printf("ERROR: Wrong value to -boxes1d parameter, cannot be <=0\n");
	      		return 0;
	      	}
	      	dim_cpu.cur_arg = dim_cpu.cur_arg+1;
	      }
	      // value is not a number
	      else{
	      	printf("ERROR: Value to -boxes1d parameter in not a number\n");
	      	return 0;
	      }
	    }
	    // value not provided
	    else{
	    	printf("ERROR: Missing value to -boxes1d parameter\n");
	    	return 0;
	    }
	  }
	  // unknown
	  else{
	  	printf("ERROR: Unknown parameter\n");
	  	return 0;
	  }
	}

	// Print configuration
	printf("Configuration used: boxes1d = %d\n", dim_cpu.boxes1d_arg);

	//======================================================================================================================================================150
	//	INPUTS
	//======================================================================================================================================================150
	par_cpu = 0.5;
	//======================================================================================================================================================150
	//	DIMENSIONS
	//======================================================================================================================================================150
	// total number of boxes
	dim_cpu.number_boxes = dim_cpu.boxes1d_arg * dim_cpu.boxes1d_arg * dim_cpu.boxes1d_arg;
	// how many particles space has in each direction
	dim_cpu.space_elem = dim_cpu.number_boxes * NUMBER_PAR_PER_BOX;
	// box array
	dim_cpu.box_mem = dim_cpu.number_boxes * sizeof(box_str);

	//====================================================================================================100
	//	BOX
	//====================================================================================================100

	// allocate boxes
	box_cpu = (box_str*)malloc(dim_cpu.box_mem);
	// initialize number of home boxes
	nh = 0;
	// home boxes in z direction
	for(i=0; i<dim_cpu.boxes1d_arg; i++){
		// home boxes in y direction
		for(j=0; j<dim_cpu.boxes1d_arg; j++){
			// home boxes in x direction
			for(k=0; k<dim_cpu.boxes1d_arg; k++){

				// current home box
				box_cpu[nh].x = k;
				box_cpu[nh].y = j;
				box_cpu[nh].z = i;
				box_cpu[nh].number = nh;
				box_cpu[nh].offset = nh * NUMBER_PAR_PER_BOX;

				// initialize number of neighbor boxes
				box_cpu[nh].nn = 0;

				// neighbor boxes in z direction
				for(l=-1; l<2; l++){
					// neighbor boxes in y direction
					for(m=-1; m<2; m++){
						// neighbor boxes in x direction
						for(n=-1; n<2; n++){

							if(		(((i+l)>=0 && (j+m)>=0 && (k+n)>=0)==true && ((i+l)<dim_cpu.boxes1d_arg && (j+m)<dim_cpu.boxes1d_arg && (k+n)<dim_cpu.boxes1d_arg)==true)	&&
									(l==0 && m==0 && n==0)==false	){

								box_cpu[nh].nei[box_cpu[nh].nn].x = (k+n);
								box_cpu[nh].nei[box_cpu[nh].nn].y = (j+m);
								box_cpu[nh].nei[box_cpu[nh].nn].z = (i+l);
								box_cpu[nh].nei[box_cpu[nh].nn].number =	(box_cpu[nh].nei[box_cpu[nh].nn].z * dim_cpu.boxes1d_arg * dim_cpu.boxes1d_arg) + 
																			(box_cpu[nh].nei[box_cpu[nh].nn].y * dim_cpu.boxes1d_arg) + 
																			 box_cpu[nh].nei[box_cpu[nh].nn].x;
								box_cpu[nh].nei[box_cpu[nh].nn].offset = box_cpu[nh].nei[box_cpu[nh].nn].number * NUMBER_PAR_PER_BOX;

								box_cpu[nh].nn = box_cpu[nh].nn + 1;

							}

						} // neighbor boxes in x direction
					} // neighbor boxes in y direction
				} // neighbor boxes in z direction

				// increment home box
				nh = nh + 1;

			} // home boxes in x direction
		} // home boxes in y direction
	} // home boxes in z direction

	//====================================================================================================100
	//	PARAMETERS, DISTANCE, CHARGE AND FORCE
	//====================================================================================================100


	double* rv_cpu_v = new double[dim_cpu.space_elem];
	double* rv_cpu_x = new double[dim_cpu.space_elem];
	double* rv_cpu_y = new double[dim_cpu.space_elem];
	double* rv_cpu_z = new double[dim_cpu.space_elem];
	for(i=0; i<dim_cpu.space_elem; i=i+1){
		rv_cpu_v[i] = (rand()%10 + 1) / 10.0;			
		rv_cpu_x[i] = (rand()%10 + 1) / 10.0;			
		rv_cpu_y[i] = (rand()%10 + 1) / 10.0;			
		rv_cpu_z[i] = (rand()%10 + 1) / 10.0;			
	}

	double* qv_cpu = new double[dim_cpu.space_elem];
	for(i=0; i<dim_cpu.space_elem; i=i+1){
		qv_cpu[i] = (rand()%10 + 1) / 10.0;			
	}

	double* fv_cpu_v = new double[dim_cpu.space_elem];
	double* fv_cpu_x = new double[dim_cpu.space_elem];
	double* fv_cpu_y = new double[dim_cpu.space_elem];
	double* fv_cpu_z = new double[dim_cpu.space_elem];
	for(i=0; i<dim_cpu.space_elem; i=i+1){
		fv_cpu_v[i] = 0;								
		fv_cpu_x[i] = 0;								
		fv_cpu_y[i] = 0;								
		fv_cpu_z[i] = 0;								
	}

	box_str* d_box_gpu;
	double* d_rv_v;
	double* d_rv_x;
	double* d_rv_y;
	double* d_rv_z;
	double* d_qv_gpu;
	double* d_fv_v;
	double* d_fv_x;
	double* d_fv_y;
	double* d_fv_z;

	//====================================================================================================100
	//	GPU_CUDA
	//====================================================================================================100
	  cudaThreadSynchronize();

	  dim3 threads;
	  dim3 blocks;
	  blocks.x = dim_cpu.number_boxes;
	  blocks.y = 1;
	  threads.x = NUMBER_THREADS;											
	  threads.y = 1;


	  cudaMemcpyToSymbol (dev_par, &par_cpu, sizeof(double));

	  cudaMalloc(	(void **)&d_box_gpu, dim_cpu.box_mem);

	  cudaMalloc(	(void **)&d_rv_v, dim_cpu.space_elem*sizeof(double));
	  cudaMalloc(	(void **)&d_rv_x, dim_cpu.space_elem*sizeof(double));
	  cudaMalloc(	(void **)&d_rv_y, dim_cpu.space_elem*sizeof(double));
	  cudaMalloc(	(void **)&d_rv_z, dim_cpu.space_elem*sizeof(double));

	  cudaMalloc(	(void **)&d_qv_gpu, dim_cpu.space_elem*sizeof(double));

	  cudaMalloc(	(void **)&d_fv_v, dim_cpu.space_elem*sizeof(double));
	  cudaMalloc(	(void **)&d_fv_x, dim_cpu.space_elem*sizeof(double));
	  cudaMalloc(	(void **)&d_fv_y, dim_cpu.space_elem*sizeof(double));
	  cudaMalloc(	(void **)&d_fv_z, dim_cpu.space_elem*sizeof(double));

    struct timeval time_start;
    struct timeval time_end;
    gettimeofday(&time_start, NULL);	

	  cudaMemcpy(	d_box_gpu, box_cpu, dim_cpu.box_mem, cudaMemcpyHostToDevice);

	  cudaMemcpy(	d_rv_v, rv_cpu_v, dim_cpu.space_elem*sizeof(double), cudaMemcpyHostToDevice);
	  cudaMemcpy(	d_rv_x, rv_cpu_x, dim_cpu.space_elem*sizeof(double), cudaMemcpyHostToDevice);
	  cudaMemcpy(	d_rv_y, rv_cpu_y, dim_cpu.space_elem*sizeof(double), cudaMemcpyHostToDevice);
	  cudaMemcpy(	d_rv_z, rv_cpu_z, dim_cpu.space_elem*sizeof(double), cudaMemcpyHostToDevice);

	  cudaMemcpy(	d_qv_gpu, qv_cpu, dim_cpu.space_elem*sizeof(double), cudaMemcpyHostToDevice);

	  cudaMemcpy(	d_fv_v, fv_cpu_v, dim_cpu.space_elem*sizeof(double), cudaMemcpyHostToDevice);
	  cudaMemcpy(	d_fv_x, fv_cpu_x, dim_cpu.space_elem*sizeof(double), cudaMemcpyHostToDevice);
	  cudaMemcpy(	d_fv_y, fv_cpu_y, dim_cpu.space_elem*sizeof(double), cudaMemcpyHostToDevice);
	  cudaMemcpy(	d_fv_z, fv_cpu_z, dim_cpu.space_elem*sizeof(double), cudaMemcpyHostToDevice);

	  kernel_gpu_cuda<<<blocks, threads>>>(dim_cpu, d_box_gpu, d_rv_v, d_rv_x, d_rv_y, d_rv_z, d_qv_gpu, d_fv_v, d_fv_x, d_fv_y, d_fv_z);

	  checkCUDAError("Start");
	  cudaThreadSynchronize();

	  cudaMemcpy(	fv_cpu_v, d_fv_v, dim_cpu.space_elem*sizeof(double), cudaMemcpyDeviceToHost);
	  cudaMemcpy(	fv_cpu_x, d_fv_x, dim_cpu.space_elem*sizeof(double), cudaMemcpyDeviceToHost);
	  cudaMemcpy(	fv_cpu_y, d_fv_y, dim_cpu.space_elem*sizeof(double), cudaMemcpyDeviceToHost);
	  cudaMemcpy(	fv_cpu_z, d_fv_z, dim_cpu.space_elem*sizeof(double), cudaMemcpyDeviceToHost);

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
	//======================================================================================================================================================150
	//	SYSTEM MEMORY DEALLOCATION
	//======================================================================================================================================================150

  gettimeofday(&time_end, NULL);

  mpf_t val_x, val_y, val_in, err;
  mpf_init2(val_x, 128);
  mpf_init2(val_y, 128);
  mpf_init2(val_in, 128);
  mpf_init2(err, 128);
  FILE *infile = fopen("fv_ref.txt", "r");
  for (int i=0; i<dim_cpu.space_elem; i++) {
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
  gmp_printf("error: %.80Ff\n", err);

	// dump results
//#ifdef OUTPUT
//        FILE *fptr;
//	fptr = fopen("result.txt", "w");	
//	for(i=0; i<dim_cpu.space_elem; i=i+1){
//        	fprintf(fptr, "%f, %f, %f, %f\n", fv_cpu[i].v, fv_cpu[i].x, fv_cpu[i].y, fv_cpu[i].z);
//	}
//	fclose(fptr);
//#endif       	
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

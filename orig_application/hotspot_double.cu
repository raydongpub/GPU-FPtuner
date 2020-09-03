#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <assert.h>
#include <mpfr.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

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

#define STR_SIZE 256

/* maximum power density possible (say 300W for a 10mm x 10mm chip)	*/
#define MAX_PD	(3.0e6)
/* required precision in degrees	*/
#define PRECISION	0.001
#define SPEC_HEAT_SI 1.75e6
#define K_SI 100
/* capacitance fitting factor	*/
#define FACTOR_CHIP	0.5

#define pin_stats_reset()   startCycle()
#define pin_stats_pause(cycles)   stopCycle(cycles)
#define pin_stats_dump(cycles)    printf("timer: %Lu\n", cycles)



__constant__  double dcap, dxr, dyr, dzr, dsp;

void readtemp(double *vtp, int grid_rows, int grid_cols, char *file){
	int i,j;
	FILE *fp;
	char str[STR_SIZE];
	float tpv;
	if( (fp  = fopen(file, "r" )) ==0 )
            printf( "The file was not opened\n" );
	for (i=0; i <= grid_rows-1; i++) 
	 for (j=0; j <= grid_cols-1; j++)
	 {
		fgets(str, STR_SIZE, fp);
		if (feof(fp))
	    fprintf(stderr, "error: not enough lines in file\n");
		if ((sscanf(str, "%f", &tpv) != 1))
	    fprintf(stderr, "error: invalid file format\n");
		vtp[i*grid_cols+j] = (double)tpv;
	}
	fclose(fp);	
}
void readpower(double *vpr, int grid_rows, int grid_cols, char *file){
	int i,j;
	FILE *fp;
	char str[STR_SIZE];
	float pwv;
	if( (fp  = fopen(file, "r" )) ==0 )
            printf( "The file was not opened\n" );
	for (i=0; i <= grid_rows-1; i++) 
	 for (j=0; j <= grid_cols-1; j++)
	 {
		fgets(str, STR_SIZE, fp);
		if (feof(fp))
	    fprintf(stderr, "error: not enough lines in file\n");
		if ((sscanf(str, "%f", &pwv) != 1))
	    fprintf(stderr, "error: invalid file format\n");
		vpr[i*grid_cols+j] = (double)pwv;
	}
	fclose(fp);	
}

#define IN_RANGE(x, min, max)   ((x)>=(min) && (x)<=(max))
#define CLAMP_RANGE(x, min, max) x = (x<(min)) ? min : ((x>(max)) ? max : x )
#define MIN(a, b) ((a)<=(b) ? (a) : (b))

__global__ void calculate_temp(int iteration,  
                               double *pwer,   
                               double *temp_src,    
                               double *temp_dst,    
                               int grid_cols,  
                               int grid_rows,  
							                 int border_cols,  
							                 int border_rows){  
	
        __shared__ double temp_on_cuda[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ double power_on_cuda[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ double temp_t[BLOCK_SIZE][BLOCK_SIZE]; 

	double amb_temp = (double)80.0;
        double step_div_Cap;
        double Rx_1,Ry_1,Rz_1;
        
	int bx = blockIdx.x;
        int by = blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;
	
	step_div_Cap=dsp/dcap;
	
	Rx_1=1/dxr;
	Ry_1=1/dyr;
	Rz_1=1/dzr;
	
 
        // calculate the small block size
	int small_block_rows = BLOCK_SIZE-iteration*2;
	int small_block_cols = BLOCK_SIZE-iteration*2;

        // calculate the boundary for the block according to 
        // the boundary of its small block
        int blkY = small_block_rows*by-border_rows;
        int blkX = small_block_cols*bx-border_cols;
        int blkYmax = blkY+BLOCK_SIZE-1;
        int blkXmax = blkX+BLOCK_SIZE-1;

        // calculate the global thread coordination
	int yidx = blkY+ty;
	int xidx = blkX+tx;

        // load data if it is within the valid input range
	int loadYidx=yidx, loadXidx=xidx;
        int index = grid_cols*loadYidx+loadXidx;
       
	if(IN_RANGE(loadYidx, 0, grid_rows-1) && IN_RANGE(loadXidx, 0, grid_cols-1)){
            temp_on_cuda[ty][tx] = temp_src[index];  
            power_on_cuda[ty][tx] = pwer[index];
	}
	__syncthreads();

        // effective range within this block that falls within 
        // the valid range of the input data
        // used to rule out computation outside the boundary.
        int validYmin = (blkY < 0) ? -blkY : 0;
        int validYmax = (blkYmax > grid_rows-1) ? BLOCK_SIZE-1-(blkYmax-grid_rows+1) : BLOCK_SIZE-1;
        int validXmin = (blkX < 0) ? -blkX : 0;
        int validXmax = (blkXmax > grid_cols-1) ? BLOCK_SIZE-1-(blkXmax-grid_cols+1) : BLOCK_SIZE-1;

        int N = ty-1;
        int S = ty+1;
        int W = tx-1;
        int E = tx+1;
        
        N = (N < validYmin) ? validYmin : N;
        S = (S > validYmax) ? validYmax : S;
        W = (W < validXmin) ? validXmin : W;
        E = (E > validXmax) ? validXmax : E;

        bool computed;
        for (int i=0; i<iteration ; i++){ 
            computed = false;
            if( IN_RANGE(tx, i+1, BLOCK_SIZE-i-2) &&  \
                  IN_RANGE(ty, i+1, BLOCK_SIZE-i-2) &&  \
                  IN_RANGE(tx, validXmin, validXmax) && \
                  IN_RANGE(ty, validYmin, validYmax) ) {
                  computed = true;
                  temp_t[ty][tx] =   temp_on_cuda[ty][tx] + step_div_Cap * (power_on_cuda[ty][tx] + 
	       	         (temp_on_cuda[S][tx] + temp_on_cuda[N][tx] - 2.0*temp_on_cuda[ty][tx]) * Ry_1 + 
		             (temp_on_cuda[ty][E] + temp_on_cuda[ty][W] - 2.0*temp_on_cuda[ty][tx]) * Rx_1 + 
		             (amb_temp - temp_on_cuda[ty][tx]) * Rz_1);
	
            }
            __syncthreads();
            if(i==iteration-1)
                break;
            if(computed)	 
                temp_on_cuda[ty][tx]= temp_t[ty][tx];
            __syncthreads();
          }

      // update the global memory
      // after the last iteration, only threads coordinated within the 
      // small block perform the calculation and switch on ``computed''
      if (computed){
          temp_dst[index]= temp_t[ty][tx];		
      }
}


void usage(int argc, char **argv)
{
	fprintf(stderr, "Usage: %s <grid_rows/grid_cols> <pyramid_height> <sim_time> <temp_file> <power_file> <output_file>\n", argv[0]);
	fprintf(stderr, "\t<grid_rows/grid_cols>  - number of rows/cols in the grid (positive integer)\n");
	fprintf(stderr, "\t<pyramid_height> - pyramid heigh(positive integer)\n");
	fprintf(stderr, "\t<sim_time>   - number of iterations\n");
	fprintf(stderr, "\t<temp_file>  - name of the file containing the initial temperature values of each cell\n");
	fprintf(stderr, "\t<power_file> - name of the file containing the dissipated power values of each cell\n");
	fprintf(stderr, "\t<output_file> - name of the output file\n");
	exit(1);
}

int main(int argc, char** argv)
{
  printf("WG size of kernel = %d X %d\n", BLOCK_SIZE, BLOCK_SIZE);

  int size;
  int grid_rows,grid_cols;
  //char *tfile, *pfile, *ofile;
  char *tfile, *pfile;
  struct timeval start_t;
  struct timeval end_t;
  struct timeval skt_t;
  struct timeval ske_t;

  
  int total_iterations = 60;
  int pyramid_height = 1; // number of iterations
  int ret;
  if (argc != 6)
  	usage(argc, argv);
  if((grid_rows = atoi(argv[1]))<=0||(grid_cols = atoi(argv[1]))<=0||(pyramid_height = atoi(argv[2]))<=0||(total_iterations = atoi(argv[3]))<=0)
  	usage(argc, argv);
  tfile=argv[4];
  pfile=argv[5];
  size=grid_rows*grid_cols;

  /* --------------- pyramid parameters --------------- */
  # define EXPAND_RATE 2
  int borderCols = (pyramid_height)*EXPAND_RATE/2;
  int borderRows = (pyramid_height)*EXPAND_RATE/2;
  int smallBlockCol = BLOCK_SIZE-(pyramid_height)*EXPAND_RATE;
  int smallBlockRow = BLOCK_SIZE-(pyramid_height)*EXPAND_RATE;
  int blockCols = grid_cols/smallBlockCol+((grid_cols%smallBlockCol==0)?0:1);
  int blockRows = grid_rows/smallBlockRow+((grid_rows%smallBlockRow==0)?0:1);

  double *FilesavingTemp = new double[size];
  double *FilesavingPower =  new double[size];
  double *MatrixOut = new double[size];
  for (int i=0; i<size; i++)
    MatrixOut[i] = (double)0.0;

  readtemp(FilesavingTemp, grid_rows, grid_cols, tfile);
  readpower(FilesavingPower, grid_rows, grid_cols, pfile);
	double t_chip = 0.0005;
  double chip_height = 0.016;
  double chip_width = 0.016;

	double grid_height = chip_height / grid_rows;
	double grid_width = chip_width / grid_cols;
	double max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);

	double Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height;
	double Rx = grid_width / (2.0 * K_SI * t_chip * grid_height);
	double Ry = grid_height / (2.0 * K_SI * t_chip * grid_width);
	double Rz = t_chip / (K_SI * grid_height * grid_width);

	double step = PRECISION / max_slope;
	int ts;
  int src = 1, dst = 0;
  int dviter;
  int temp;
  gettimeofday(&start_t,0L);

	cudaMemcpyToSymbol (&dcap, &Cap, sizeof(double));
	cudaMemcpyToSymbol (&dxr, &Rx, sizeof(double));
	cudaMemcpyToSymbol (&dyr, &Ry, sizeof(double));
	cudaMemcpyToSymbol (&dzr, &Rz, sizeof(double));
	cudaMemcpyToSymbol (&dsp, &step, sizeof(double));

  double *MatrixTemp[2], *MatrixPower;
  cudaMalloc((void**)&MatrixTemp[0], sizeof(double)*size);
  cudaMalloc((void**)&MatrixTemp[1], sizeof(double)*size);

  cudaMemcpy(MatrixTemp[0], FilesavingTemp, sizeof(double)*size, cudaMemcpyHostToDevice);
  cudaMemcpy(MatrixTemp[1], MatrixOut, sizeof(double)*size, cudaMemcpyHostToDevice);

  cudaMalloc((void**)&MatrixPower, sizeof(double)*size);
  cudaMemcpy(MatrixPower, FilesavingPower, sizeof(double)*size, cudaMemcpyHostToDevice);
  printf("Start computing the transient temperature\n");

	
  gettimeofday(&skt_t,0L);
	for (ts = 0; ts < total_iterations; ts=ts+pyramid_height) {
     temp = src;
     src = dst;
     dst = temp;
     dviter = MIN(pyramid_height, total_iterations-ts);
     dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
     dim3 dimGrid(blockCols, blockRows);  

     calculate_temp<<<dimGrid, dimBlock>>>(dviter, MatrixPower,MatrixTemp[src],MatrixTemp[dst],\
		grid_cols, grid_rows,borderCols, borderRows);
	}
	cudaThreadSynchronize();
  ret = dst;
  gettimeofday(&ske_t,0L);

  printf("Ending simulation\n");

  cudaMemcpy(MatrixOut, MatrixTemp[ret], sizeof(double)*size, cudaMemcpyDeviceToHost);

  gettimeofday(&end_t,0L);

  std::cout << "time: " << ((end_t.tv_sec + end_t.tv_usec*1e-6) - (start_t.tv_sec + start_t.tv_usec*1e-6)) << "\n";
  std::cout <<"kernel: " << ((ske_t . tv_sec - skt_t . tv_sec) + (ske_t . tv_usec - skt_t . tv_usec) * 1e-6) << endl;

  cudaFree(MatrixPower);
  cudaFree(MatrixTemp[0]);
  cudaFree(MatrixTemp[1]);
  free(MatrixOut);

  return EXIT_SUCCESS;
}

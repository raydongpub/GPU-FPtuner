
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#define NPB_VERSION "3.3.1"

using namespace std;

#define min(x,y) (x) <= (y) ? (x) : (y)
#define max(x,y) (x) >= (y) ? (x) : (y)

// block sizes for CUDA kernels
#define NORM_BLOCK 32
#define SOLVE_BLOCK 32
#define ERHS_BLOCK 32

// timer constants
#define t_total 0
#define t_rhsx 1
#define t_rhsy 2
#define t_rhsz 3
#define t_rhs 4
#define t_xsolve 5
#define t_ysolve 6
#define t_zsolve 7
#define t_rdis1 8
#define t_rdis2 9
#define t_txinvr 10
#define t_pinvr 11
#define t_ninvr 12
#define t_tzetar 13
#define t_add 14
#define t_last 15


//---------------------------------------------------------------------
// diffusion coefficients
//---------------------------------------------------------------------
#define dx1 0.75
#define dx2 0.75
#define dx3 0.75
#define dx4 0.75
#define dx5 0.75
#define dy1 0.75
#define dy2 0.75
#define dy3 0.75
#define	dy4 0.75
#define dy5 0.75
#define dz1 1.0
#define dz2 1.0
#define dz3 1.0
#define dz4 1.0
#define dz5 1.0
//#define dxmax max(dx3,dx4)
//#define dymax max(dy2,dy4)
//#define dzmax max(dz2,dz3)
#define dxmax dx3
#define dymax dy2
#define dzmax dz2
//---------------------------------------------------------------------
//   fourth difference dissipation
//---------------------------------------------------------------------
#define dssp (max(max(dx1,dy1),dz1)*.25)
#define c4dssp (4.0*dssp)
#define c5dssp (5.0*dssp)

#define c1 1.4
#define c2 0.4
#define	c3 0.1
#define c4 1.0
#define c5 1.4
#define c1c2 (c1*c2)
#define c1c5 (c1*c5)
#define c3c4 (c3*c4)
#define c1345 (c1c5*c3c4)
#define conz1 (1.0-c1c5)
#define c2iv 2.5
#define con43 (4.0/3.0)
#define con16 (1.0/6.0)

// macros to linearize multidimensional array accesses 
#define fu(m,i,j,k) fu[(i)+nx*((j)+ny*((k)+nz*(m)))]
#define forcing(m,i,j,k) forcing[(i)+nx*((j)+ny*((k)+nz*(m)))]
#define rhs(m,i,j,k) rhs[m+(i)*5+(j)*5*nx+(k)*5*nx*ny]
#define rho_i(i,j,k) rho_i[i+(j)*nx+(k)*nx*ny]
#define us(i,j,k) us[i+(j)*nx+(k)*nx*ny]
#define vs(i,j,k) vs[i+(j)*nx+(k)*nx*ny]
#define ws(i,j,k) ws[i+(j)*nx+(k)*nx*ny]
#define square(i,j,k) square[i+(j)*nx+(k)*nx*ny]
#define qs(i,j,k) qs[i+(j)*nx+(k)*nx*ny]
#define speed(i,j,k) speed[i+(j)*nx+(k)*nx*ny]

static void inline HandleError( cudaError_t err, const char *file, int line ) {
	if (err != cudaSuccess) {
		printf( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
		exit( EXIT_FAILURE );
	}
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

__constant__ double tx1, tx2, tx3, ty1, ty2, ty3, tz1, tz2, tz3;
__constant__ double bt, dt, dtdssp;
__constant__ double dnxm1, dnym1, dnzm1;
__constant__ double dtx1, dttx2, dty1, dtty2, dtz1, dttz2, c2dttx1, c2dtty1, c2dttz1;
__constant__ double comz1, comz4, comz5, comz6, c3c4tx3, c3c4ty3, c3c4tz3;
__constant__ double xxcon1, xxcon2, xxcon3, xxcon4, xxcon5, dx1tx1, dx2tx1, dx3tx1, dx4tx1, dx5tx1;
__constant__ double yycon1, yycon2, yycon3, yycon4, yycon5, dy1ty1, dy2ty1, dy3ty1, dy4ty1, dy5ty1;
__constant__ double zzcon1, zzcon2, zzcon3, zzcon4, zzcon5, dz1tz1, dz2tz1, dz3tz1, dz4tz1, dz5tz1;
__constant__ double ce[13][5];


//---------------------------------------------------------------------
// exact_rhs computation
//---------------------------------------------------------------------
__device__ static void exact_solution_kernel (const double xi, const double eta, const double zta, double *dtemp) {
	for (int m = 0; m < 5; m++)
		dtemp[m] = ce[0][m] + xi*(ce[1][m] + xi*(ce[4][m] + xi*(ce[7][m] + xi*ce[10][m]))) +
				eta*(ce[2][m] + eta*(ce[5][m] + eta*(ce[8][m] + eta*ce[11][m])))+
				zta*(ce[3][m] + zta*(ce[6][m] + zta*(ce[9][m] + zta*ce[12][m])));
}

__global__ static void exact_rhs_kernel_init (double *forcing, const int nx, const int ny, const int nz) {
	int i, j, k, m;
	k = blockIdx.y;
	j = blockIdx.x;
	i = threadIdx.x;
	for (m = 0; m < 5; m++) forcing(m,i,j,k) = (double)0.0;
}

__global__ static void exact_rhs_kernel_x (double *forcing, const int nx, const int ny, const int nz) {
	int i, j, k, m;
	double xi, eta, zta, dtemp[5], dtpp;
	double ue[5][5], buf[3][5], cuf[3], q[3];

	k = blockIdx.x*blockDim.x+threadIdx.x+1;
	j = blockIdx.y*blockDim.y+threadIdx.y+1;
	if (k >= nz-1 || j >= ny-1) return;

	zta = (double)k * dnzm1;
	eta = (double)j * dnym1;
	//---------------------------------------------------------------------
	//      xi-direction flux differences                      
	//---------------------------------------------------------------------
	for (i = 0; i < 3; i++) {
		xi = (double)i * dnxm1;
		exact_solution_kernel(xi, eta, zta, dtemp);
		for (m = 0; m < 5; m++) ue[i+1][m] = dtemp[m];
		dtpp = 1.0/dtemp[0];
		for (m = 1; m < 5; m++) buf[i][m] = dtpp*dtemp[m];
		cuf[i] = buf[i][1] * buf[i][1];
		buf[i][0] = cuf[i] + buf[i][2] * buf[i][2] + buf[i][3] * buf[i][3];
		q[i] = 0.5 * (buf[i][1]*ue[i+1][1] + buf[i][2]*ue[i+1][2] + buf[i][3]*ue[i+1][3]);
	}
	for (i = 1; i < nx-1; i++) {
		if (i+2 < nx) {
			xi = (double)(i+2) * dnxm1;
			exact_solution_kernel(xi, eta, zta, dtemp);
			for (m = 0; m < 5; m++) ue[4][m] = dtemp[m];
		}
		
		dtemp[0] = 0.0 - tx2*(ue[3][1]-ue[1][1])+ dx1tx1*(ue[3][0]-2.0*ue[2][0]+ue[1][0]);
		dtemp[1] = 0.0 - tx2*((ue[3][1]*buf[2][1]+c2*(ue[3][4]-q[2]))-(ue[1][1]*buf[0][1]+c2*(ue[1][4]-q[0])))+xxcon1*(buf[2][1]-2.0*buf[1][1]+buf[0][1])+dx2tx1*(ue[3][1]-2.0*ue[2][1]+ue[1][1]);
		dtemp[2] = 0.0 - tx2*(ue[3][2]*buf[2][1]-ue[1][2]*buf[0][1])+xxcon2*(buf[2][2]-2.0*buf[1][2]+buf[0][2])+dx3tx1*(ue[3][2]-2.0*ue[2][2]+ue[1][2]);
		dtemp[3] = 0.0 - tx2*(ue[3][3]*buf[2][1]-ue[1][3]*buf[0][1])+xxcon2*(buf[2][3]-2.0*buf[1][3]+buf[0][3])+dx4tx1*(ue[3][3]-2.0*ue[2][3]+ue[1][3]);
		dtemp[4] = 0.0 - tx2*(buf[2][1]*(c1*ue[3][4]-c2*q[2])-buf[0][1]*(c1*ue[1][4]-c2*q[0]))+0.5*xxcon3*(buf[2][0]-2.0*buf[1][0]+buf[0][0])+xxcon4*(cuf[2]-2.0*cuf[1]+cuf[0])+
					xxcon5*(buf[2][4]-2.0*buf[1][4]+buf[0][4])+dx5tx1*(ue[3][4]-2.0*ue[2][4]+ ue[1][4]);
		//---------------------------------------------------------------------
		//            Fourth-order dissipation                         
		//---------------------------------------------------------------------
		if (i == 1) {
			for (m = 0; m < 5; m++) forcing(m,i,j,k) = dtemp[m] - dssp*(5.0*ue[2][m] - 4.0*ue[3][m] + ue[4][m]);
		} else if (i == 2) {
			for (m = 0; m < 5; m++) forcing(m,i,j,k) = dtemp[m] - dssp*(-4.0*ue[1][m] + 6.0*ue[2][m] - 4.0*ue[3][m] + ue[4][m]);
		} else if (i >= 3 && i < nx-3) {
			for (m = 0; m < 5; m++) forcing(m,i,j,k) = dtemp[m] - dssp*(ue[0][m] - 4.0*ue[1][m]+6.0*ue[2][m] - 4.0*ue[3][m] + ue[4][m]);
		} else if (i == nx-3) {
			for (m = 0; m < 5; m++) forcing(m,i,j,k) = dtemp[m] - dssp*(ue[0][m] - 4.0*ue[1][m] +6.0*ue[2][m] - 4.0*ue[3][m]);
		} else if (i == nx-2) {
			for (m = 0; m < 5; m++) forcing(m,i,j,k) = dtemp[m] - dssp*(ue[0][m] - 4.0*ue[1][m] + 5.0*ue[2][m]);
		}

		for (m = 0; m < 5; m++) {
			ue[0][m] = ue[1][m]; 
			ue[1][m] = ue[2][m];
			ue[2][m] = ue[3][m];
			ue[3][m] = ue[4][m];
			buf[0][m] = buf[1][m];
			buf[1][m] = buf[2][m];
		}
		cuf[0] = cuf[1]; cuf[1] = cuf[2];
		q[0] = q[1]; q[1] = q[2];

		if (i < nx-2) {
			dtpp = 1.0/ue[3][0];
			for (m = 1; m < 5; m++) buf[2][m] = dtpp*ue[3][m];
			cuf[2] = buf[2][1] * buf[2][1];
			buf[2][0] = cuf[2] + buf[2][2] * buf[2][2] + buf[2][3] * buf[2][3];
			q[2] = 0.5 * (buf[2][1]*ue[3][1] + buf[2][2]*ue[3][2] + buf[2][3]*ue[3][3]);
		}
	}
}

__global__ static void exact_rhs_kernel_y (double *forcing, const int nx, const int ny, const int nz) {
	int i, j, k, m;
	double xi, eta, zta, dtemp[5], dtpp;
	double ue[5][5], buf[3][5], cuf[3], q[3];

	k = blockIdx.x*blockDim.x+threadIdx.x+1;
	i = blockIdx.y*blockDim.y+threadIdx.y+1;
	if (k >= nz-1 || i >= nx-1) return;

	zta = (double)k * dnzm1;
	xi = (double)i * dnxm1;
	//---------------------------------------------------------------------
	//  eta-direction flux differences             
	//---------------------------------------------------------------------
	for (j = 0; j < 3; j++) {
		eta = (double)j * dnym1;
		exact_solution_kernel(xi, eta, zta, dtemp);
		for (m = 0; m < 5; m++) ue[j+1][m] = dtemp[m];
		dtpp = 1.0/dtemp[0];
		for (m = 1; m < 5; m++) buf[j][m] = dtpp * dtemp[m];
		cuf[j] = buf[j][2] * buf[j][2];
		buf[j][0] = cuf[j] + buf[j][1] * buf[j][1] + buf[j][3] * buf[j][3];
		q[j] = 0.5*(buf[j][1]*ue[j+1][1] + buf[j][2]*ue[j+1][2] + buf[j][3]*ue[j+1][3]);
	}

	for (j = 1; j < ny-1; j++) {
		if (j+2 < ny) {
			eta = (double)(j+2) * dnym1;
			exact_solution_kernel(xi, eta, zta, dtemp);
			for (m = 0; m < 5; m++) ue[4][m] = dtemp[m];
		}

		dtemp[0] = forcing(0,i,j,k) - ty2*(ue[3][2]-ue[1][2])+ dy1ty1*(ue[3][0]-2.0*ue[2][0]+ue[1][0]);
		dtemp[1] = forcing(1,i,j,k) - ty2*(ue[3][1]*buf[2][2]-ue[1][1]*buf[0][2])+yycon2*(buf[2][1]-2.0*buf[1][1]+buf[0][1])+dy2ty1*(ue[3][1]-2.0*ue[2][1]+ ue[1][1]);
		dtemp[2] = forcing(2,i,j,k) - ty2*((ue[3][2]*buf[2][2]+c2*(ue[3][4]-q[2]))-(ue[1][2]*buf[0][2]+c2*(ue[1][4]-q[0])))+yycon1*(buf[2][2]-2.0*buf[1][2]+buf[0][2])+dy3ty1*( ue[3][2]-2.0*ue[2][2] +ue[1][2]);
		dtemp[3] = forcing(3,i,j,k) - ty2*(ue[3][3]*buf[2][2]-ue[1][3]*buf[0][2])+yycon2*(buf[2][3]-2.0*buf[1][3]+buf[0][3])+dy4ty1*( ue[3][3]-2.0*ue[2][3]+ ue[1][3]);
		dtemp[4] = forcing(4,i,j,k) - ty2*(buf[2][2]*(c1*ue[3][4]-c2*q[2])-buf[0][2]*(c1*ue[1][4]-c2*q[0]))+0.5*yycon3*(buf[2][0]-2.0*buf[1][0]+buf[0][0])+yycon4*(cuf[2]-2.0*cuf[1]+cuf[0])+
					yycon5*(buf[2][4]-2.0*buf[1][4]+buf[0][4])+dy5ty1*(ue[3][4]-2.0*ue[2][4]+ue[1][4]);
		//---------------------------------------------------------------------
		//            Fourth-order dissipation                      
		//---------------------------------------------------------------------
		if (j == 1) {
			for (m = 0; m < 5; m++) forcing(m,i,j,k) = dtemp[m] - dssp * (5.0*ue[2][m] - 4.0*ue[3][m] +ue[4][m]);
		} else if (j == 2) {
			for (m = 0; m < 5; m++) forcing(m,i,j,k) = dtemp[m] - dssp * (-4.0*ue[1][m] + 6.0*ue[2][m] - 4.0*ue[3][m] +       ue[4][m]);
		} else if (j >= 3 && j < ny-3) {
			for (m = 0; m < 5; m++) forcing(m,i,j,k) = dtemp[m] - dssp*(ue[0][m] - 4.0*ue[1][m] + 6.0*ue[2][m] - 4.0*ue[3][m] + ue[4][m]);
		} else if (j == ny-3) {
			for (m = 0; m < 5; m++) forcing(m,i,j,k) = dtemp[m] - dssp * (ue[0][m] - 4.0*ue[1][m] + 6.0*ue[2][m] - 4.0*ue[3][m]);
		} else if (j == ny-2) {
			for (m = 0; m < 5; m++) forcing(m,i,j,k) = dtemp[m] - dssp * (ue[0][m] - 4.0*ue[1][m] + 5.0*ue[2][m]);
		}

		for (m = 0; m < 5; m++) {
			ue[0][m] = ue[1][m]; 
			ue[1][m] = ue[2][m];
			ue[2][m] = ue[3][m];
			ue[3][m] = ue[4][m];
			buf[0][m] = buf[1][m];
			buf[1][m] = buf[2][m];
		}
		cuf[0] = cuf[1]; cuf[1] = cuf[2];
		q[0] = q[1]; q[1] = q[2];

		if (j < ny-2) {
			dtpp = 1.0/ue[3][0];
			for (m = 1; m < 5; m++) buf[2][m] = dtpp * ue[3][m];
			cuf[2] = buf[2][2] * buf[2][2];
			buf[2][0] = cuf[2] + buf[2][1] * buf[2][1] + buf[2][3] * buf[2][3];
			q[2] = 0.5*(buf[2][1]*ue[3][1] + buf[2][2]*ue[3][2] + buf[2][3]*ue[3][3]);
		}
	}
}

__global__ static void exact_rhs_kernel_z (double *forcing, const int nx, const int ny, const int nz) {
	int i, j, k, m;
	double xi, eta, zta, dtpp, dtemp[5];
	double ue[5][5], buf[3][5], cuf[3], q[3];

	j = blockIdx.x*blockDim.x+threadIdx.x+1;
	i = blockIdx.y*blockDim.y+threadIdx.y+1;
	if (j >= ny-1 || i >= nx-1) return;

	eta = (double)j * dnym1;
	xi = (double)i * dnxm1;
	//---------------------------------------------------------------------
	//      zeta-direction flux differences                      
	//---------------------------------------------------------------------
	for (k = 0; k < 3; k++) {
		zta = (double)k * dnzm1;
		exact_solution_kernel(xi, eta, zta, dtemp);
		for (m = 0; m < 5; m++) ue[k+1][m] = dtemp[m];
		dtpp = 1.0/dtemp[0];
		for (m = 1; m < 5; m++) buf[k][m] = dtpp * dtemp[m];
		cuf[k] = buf[k][3] * buf[k][3];
		buf[k][0] = cuf[k] + buf[k][1] * buf[k][1] + buf[k][2] * buf[k][2];
		q[k] = 0.5*(buf[k][1]*ue[k+1][1] + buf[k][2]*ue[k+1][2] + buf[k][3]*ue[k+1][3]);
	}

	for (k = 1; k < nz-1; k++) {
		if (k+2 < nz) {
			zta = (double)(k+2) * dnzm1;
			exact_solution_kernel(xi, eta, zta, dtemp);
			for (m = 0; m < 5; m++) ue[4][m] = dtemp[m];
		}

		dtemp[0] = forcing(0,i,j,k) - tz2*(ue[3][3]-ue[1][3])+dz1tz1*(ue[3][0]-2.0*ue[2][0]+ue[1][0]);
		dtemp[1] = forcing(1,i,j,k) - tz2*(ue[3][1]*buf[2][3]-ue[1][1]*buf[0][3])+zzcon2*(buf[2][1]-2.0*buf[1][1]+buf[0][1])+dz2tz1*(ue[3][1]-2.0*ue[2][1]+ue[1][1]);
		dtemp[2] = forcing(2,i,j,k) - tz2*(ue[3][2]*buf[2][3]-ue[1][2]*buf[0][3])+zzcon2*(buf[2][2]-2.0*buf[1][2]+buf[0][2])+dz3tz1*(ue[3][2]-2.0*ue[2][2]+ue[1][2]);
		dtemp[3] = forcing(3,i,j,k) - tz2*((ue[3][3]*buf[2][3]+c2*(ue[3][4]-q[2]))-(ue[1][3]*buf[0][3]+c2*(ue[1][4]-q[0])))+zzcon1*(buf[2][3]-2.0*buf[1][3]+buf[0][3])+dz4tz1*(ue[3][3]-2.0*ue[2][3] +ue[1][3]);
		dtemp[4] = forcing(4,i,j,k) - tz2*(buf[2][3]*(c1*ue[3][4]-c2*q[2])-buf[0][3]*(c1*ue[1][4]-c2*q[0]))+0.5*zzcon3*(buf[2][0]-2.0*buf[1][0]+buf[0][0])+
					zzcon4*(cuf[2]-2.0*cuf[1]+cuf[0])+zzcon5*(buf[2][4]-2.0*buf[1][4]+buf[0][4])+dz5tz1*(ue[3][4]-2.0*ue[2][4]+ue[1][4]);
		//---------------------------------------------------------------------
		//            Fourth-order dissipation
		//---------------------------------------------------------------------
		if (k == 1) {
			for (m = 0; m < 5; m++) dtemp[m] = dtemp[m] - dssp*(5.0*ue[2][m]-4.0*ue[3][m]+ue[4][m]);
		} else if (k == 2) {
			for (m = 0; m < 5; m++) dtemp[m] = dtemp[m] - dssp*(-4.0*ue[1][m]+6.0*ue[2][m]-4.0*ue[3][m]+ue[4][m]);
		} else if (k >= 3 && k < nz-3) {
			for (m = 0; m < 5; m++) dtemp[m] = dtemp[m] - dssp*(ue[0][m]-4.0*ue[1][m]+6.0*ue[2][m]-4.0*ue[3][m]+ue[4][m]);
		} else if (k == nz-3) {
			for (m = 0; m < 5; m++) dtemp[m] = dtemp[m] - dssp*(ue[0][m]-4.0*ue[1][m] + 6.0*ue[2][m] - 4.0*ue[3][m]);
		} else if (k == nz-2) {
			for (m = 0; m < 5; m++) dtemp[m] = dtemp[m] - dssp*(ue[0][m]-4.0*ue[1][m]+5.0*ue[2][m]);
		}
		//---------------------------------------------------------------------
		// now change the sign of the forcing function, 
		//---------------------------------------------------------------------
		for (m = 0; m < 5; m++) forcing(m,i,j,k) = -1.0 * dtemp[m];

		for (m = 0; m < 5; m++) {
			ue[0][m] = ue[1][m]; 
			ue[1][m] = ue[2][m];
			ue[2][m] = ue[3][m];
			ue[3][m] = ue[4][m];
			buf[0][m] = buf[1][m];
			buf[1][m] = buf[2][m];
		}
		cuf[0] = cuf[1]; cuf[1] = cuf[2];
		q[0] = q[1]; q[1] = q[2];

		if (k < nz-2) {
			dtpp = 1.0/ue[3][0];
			for (m = 1; m < 5; m++) buf[2][m] = dtpp * ue[3][m];
			cuf[2] = buf[2][3] * buf[2][3];
			buf[2][0] = cuf[2] + buf[2][1] * buf[2][1] + buf[2][2] * buf[2][2];
			q[2] = 0.5*(buf[2][1]*ue[3][1] + buf[2][2]*ue[3][2] + buf[2][3]*ue[3][3]);
		}
	}
}

void exact_rhs (double* forcing, int nx, int ny, int nz) {
	dim3 gridinit(ny,nz);
	exact_rhs_kernel_init<<<gridinit,nx>>>(forcing, nx, ny, nz);

	int yblock = min(ERHS_BLOCK,ny);
	int ygrid = (ny+yblock-1)/yblock;
	int zblock_y = min(ERHS_BLOCK/yblock,nz);
	int zgrid_y = (nz+zblock_y-1)/zblock_y;
	dim3 grid_x(zgrid_y,ygrid), block_x(zblock_y,yblock);
	exact_rhs_kernel_x<<<grid_x,block_x>>>(forcing, nx, ny, nz);

	int xblock = min(ERHS_BLOCK,nx);
	int xgrid = (nx+xblock-1)/xblock;
	int zblock_x = min(ERHS_BLOCK/xblock,nz);
	int zgrid_x = (nz+zblock_x-1)/zblock_x;
	dim3 grid_y(zgrid_x,xgrid), block_y(zblock_x,xblock);
	exact_rhs_kernel_y<<<grid_y,block_y>>>(forcing, nx, ny, nz);

	int yblock_x = min(ERHS_BLOCK/xblock,ny);
	int ygrid_x = (ny+yblock_x-1)/yblock_x;
	dim3 grid_z(ygrid_x,xgrid), block_z(yblock_x,xblock);
	exact_rhs_kernel_z<<<grid_z,block_z>>>(forcing, nx, ny, nz);
}


//---------------------------------------------------------------------
// initialize_kernel
//---------------------------------------------------------------------
__global__ static void initialize_kernel (double *fu, const int nx, const int ny, const int nz) {
	int i, j, k, m;
	double xi, eta, zta, temp[5];
	double Pface11[5], Pface12[5], Pface21[5], Pface22[5], Pface31[5], Pface32[5];
  double zero, one;

	k = blockIdx.x;
	j = blockIdx.y;
	i = threadIdx.x;

	//---------------------------------------------------------------------
	//  to compute the whole thing with a simple loop. Make sure those 
	//  values are nonzero by initializing the whole thing here. 
	//---------------------------------------------------------------------
	fu(0,i,j,k) = (double)1.0;
	fu(1,i,j,k) = (double)0.0;
	fu(2,i,j,k) = (double)0.0;
	fu(3,i,j,k) = (double)0.0;
	fu(4,i,j,k) = (double)1.0;
  zero = (double)0.0;
  one = (double)1.0;

	//---------------------------------------------------------------------
	// first store the "interpolated" values everywhere on the zone    
	//---------------------------------------------------------------------
	zta = (double)k * dnzm1;
	eta = (double)j * dnym1;
	xi = (double)i * dnxm1;
	exact_solution_kernel (zero, eta, zta, Pface11);
	exact_solution_kernel (one, eta, zta, Pface12);
	exact_solution_kernel (xi, zero, zta, Pface21);
	exact_solution_kernel (xi, one, zta, Pface22);
	exact_solution_kernel (xi, eta, zero, Pface31);
	exact_solution_kernel (xi, eta, one, Pface32);
	for (m = 0; m < 5; m++) {
		double Pxi = xi * Pface12[m] + (1.0-xi)*Pface11[m];
		double Peta = eta * Pface22[m] + (1.0-eta)*Pface21[m];
		double Pzeta = zta * Pface32[m] + (1.0-zta)*Pface31[m];
		fu(m,i,j,k) = Pxi + Peta + Pzeta - Pxi*Peta - Pxi*Pzeta - Peta*Pzeta + Pxi*Peta*Pzeta;
	}

	//---------------------------------------------------------------------
	// now store the exact values on the boundaries        
	//---------------------------------------------------------------------

	//---------------------------------------------------------------------
	// west face                                                  
	//---------------------------------------------------------------------
	xi = (double)0.0;
	if (i == 0) {
		zta = (double)k * dnzm1;
		eta = (double)j * dnym1;
		exact_solution_kernel (xi, eta, zta, temp);
		for (m = 0; m < 5; m++) fu(m,i,j,k) = temp[m];
	}
	//---------------------------------------------------------------------
	// east face                                                      
	//---------------------------------------------------------------------
	xi = (double)1.0;
	if (i == nx-1) {
		zta = (double)k * dnzm1;
		eta = (double)j * dnym1;
		exact_solution_kernel (xi, eta, zta, temp);
		for (m = 0; m < 5; m++) fu(m,i,j,k) = temp[m];
	}
	//---------------------------------------------------------------------
	// south face                                                 
	//---------------------------------------------------------------------
	eta = (double)0.0;
	if (j == 0) {
		zta = (double)k * dnzm1;
		xi = (double)i * dnxm1;
		exact_solution_kernel (xi,eta,zta,temp);
		for (m = 0; m < 5; m++) fu(m,i,j,k) = temp[m];
	}
	//---------------------------------------------------------------------
	// north face                                    
	//---------------------------------------------------------------------
	eta = (double)1.0;
	if (j == ny-1) {
		zta = (double)k * dnzm1;
		xi = (double)i * dnxm1;
		exact_solution_kernel (xi,eta,zta,temp);
		for (m = 0; m < 5; m++) fu(m,i,j,k) = temp[m];
	}
	//---------------------------------------------------------------------
	// bottom face                                       
	//---------------------------------------------------------------------
	zta = (double)0.0;
	if (k == 0) {
		eta = (double)j * dnym1;
		xi = (double)i * dnxm1;
		exact_solution_kernel (xi, eta, zta, temp);
		for (m = 0; m < 5; m++) fu(m,i,j,k) = temp[m];
	}
	//---------------------------------------------------------------------
	// top face     
	//---------------------------------------------------------------------
	zta = (double)1.0;
	if (k == nz-1) {
		eta = (double)j * dnym1;
		xi = (double)i * dnxm1;
		exact_solution_kernel (xi, eta, zta, temp);
		for (m = 0; m < 5; m++) fu(m,i,j,k) = temp[m];
	}
}

//---------------------------------------------------------------------
// adi: compute_rhs
//---------------------------------------------------------------------
__global__ static void compute_rhs_kernel_1 (double *rho_i, double *us, double *vs, double *ws, double *speed, double *qs, double *square, double *fu, const int nx, const int ny, const int nz) {
	int i, j, k;
	k = blockIdx.y;
	j = blockIdx.x;
	i = threadIdx.x;
	//---------------------------------------------------------------------
	//      compute the reciprocal of density, and the kinetic energy, 
	//      and the speed of sound. 
	//---------------------------------------------------------------------
	double rho_nv = 1.0/fu(0,i,j,k);
	double square_ijk;
	rho_i(i,j,k) = rho_nv;
	us(i,j,k) = fu(1,i,j,k) * rho_nv;
	vs(i,j,k) = fu(2,i,j,k) * rho_nv;
	ws(i,j,k) = fu(3,i,j,k) * rho_nv;
	square_ijk = 0.5*(fu(1,i,j,k)*fu(1,i,j,k) + fu(2,i,j,k)*fu(2,i,j,k) + fu(3,i,j,k)*fu(3,i,j,k)) * rho_nv;
	square(i,j,k) = 0.5*(fu(1,i,j,k)*fu(1,i,j,k) + fu(2,i,j,k)*fu(2,i,j,k) + fu(3,i,j,k)*fu(3,i,j,k)) * rho_nv;
	qs(i,j,k) = square_ijk * rho_nv;
	//---------------------------------------------------------------------
	//               (don't need speed and ainx until the lhs computation)
	//---------------------------------------------------------------------
	speed(i,j,k) = sqrt(c1c2*rho_nv*(fu(4,i,j,k) - square_ijk));
}

__global__ static void compute_rhs_kernel_2 (double *rho_i, double *us, double *vs, double *ws, double *qs, double *square, double *rhs,  double *forcing, double *fu,  int nx, const int ny, const int nz) {
	int i, j, k, m;
	k = blockIdx.y;
	j = blockIdx.x;
	i = threadIdx.x;
	double rtmp[5];

	//---------------------------------------------------------------------
	// copy the exact forcing term to the right hand side;  because 
	// this forcing term is known, we can store it on the whole zone
	// including the boundary                   
	//---------------------------------------------------------------------
	for (m = 0; m < 5; m++) rtmp[m] = forcing(m,i,j,k);
	
	//---------------------------------------------------------------------
	//      compute xi-direction fluxes 
	//---------------------------------------------------------------------
	if (k >= 1 && k < nz-1 && j >= 1 && j < ny-1 && i >= 1 && i < nx-1) {
		double uijk = us(i,j,k);
		double up1 = us(i+1,j,k);
		double um1 = us(i-1,j,k);
				
		rtmp[0] = rtmp[0] + dx1tx1*(fu(0,i+1,j,k) - 2.0*fu(0,i,j,k) + fu(0,i-1,j,k)) - tx2*(fu(1,i+1,j,k)-fu(1,i-1,j,k));
		rtmp[1] = rtmp[1] + dx2tx1*(fu(1,i+1,j,k) - 2.0*fu(1,i,j,k) + fu(1,i-1,j,k)) + xxcon2*con43*(up1-2.0*uijk+um1) - tx2*(fu(1,i+1,j,k)*up1 - fu(1,i-1,j,k)*um1 + (fu(4,i+1,j,k)-square(i+1,j,k)-fu(4,i-1,j,k)+square(i-1,j,k))*c2);
		rtmp[2] = rtmp[2] + dx3tx1*(fu(2,i+1,j,k) - 2.0*fu(2,i,j,k) + fu(2,i-1,j,k)) + xxcon2*(vs(i+1,j,k)-2.0*vs(i,j,k)+vs(i-1,j,k)) - tx2*(fu(2,i+1,j,k)*up1 - fu(2,i-1,j,k)*um1);
		rtmp[3] = rtmp[3] + dx4tx1*(fu(3,i+1,j,k) - 2.0*fu(3,i,j,k) + fu(3,i-1,j,k)) + xxcon2*(ws(i+1,j,k)-2.0*ws(i,j,k)+ws(i-1,j,k)) - tx2*(fu(3,i+1,j,k)*up1 - fu(3,i-1,j,k)*um1);
		rtmp[4] = rtmp[4] + dx5tx1*(fu(4,i+1,j,k) - 2.0*fu(4,i,j,k) + fu(4,i-1,j,k)) + xxcon3*(qs(i+1,j,k)-2.0*qs(i,j,k)+qs(i-1,j,k))+ xxcon4*(up1*up1-2.0*uijk*uijk+um1*um1) +
				xxcon5*(fu(4,i+1,j,k)*rho_i(i+1,j,k) - 2.0*fu(4,i,j,k)*rho_i(i,j,k) + fu(4,i-1,j,k)*rho_i(i-1,j,k)) - tx2*((c1*fu(4,i+1,j,k) - c2*square(i+1,j,k))*up1 - (c1*fu(4,i-1,j,k) - c2*square(i-1,j,k))*um1 );
		//---------------------------------------------------------------------
		//      add fourth order xi-direction dissipation               
		//---------------------------------------------------------------------
		if (i == 1) {
			for (m = 0; m < 5; m++) rtmp[m] = rtmp[m] - dssp * (5.0*fu(m,i,j,k)-4.0*fu(m,i+1,j,k)+fu(m,i+2,j,k));
		} else if (i == 2) {
			for (m = 0; m < 5; m++) rtmp[m] = rtmp[m] - dssp * (-4.0*fu(m,i-1,j,k)+6.0*fu(m,i,j,k)-4.0*fu(m,i+1,j,k)+fu(m,i+2,j,k));
		} else if (i >= 3 && i < nx-3) {
			for (m = 0; m < 5; m++) rtmp[m] = rtmp[m] - dssp * ( fu(m,i-2,j,k)-4.0*fu(m,i-1,j,k)+6.0*fu(m,i,j,k)-4.0*fu(m,i+1,j,k)+fu(m,i+2,j,k));
		} else if (i == nx-3) {
			for (m = 0; m < 5; m++) rtmp[m] = rtmp[m] - dssp * (fu(m,i-2,j,k)-4.0*fu(m,i-1,j,k)+6.0*fu(m,i,j,k)-4.0*fu(m,i+1,j,k) );
		} else if (i == nx-2) {
			for (m = 0; m < 5; m++) rtmp[m] = rtmp[m] - dssp * (fu(m,i-2,j,k)-4.0*fu(m,i-1,j,k) + 5.0*fu(m,i,j,k));
		}
		//---------------------------------------------------------------------
		//      compute eta-direction fluxes 
		//---------------------------------------------------------------------
		double vijk = vs(i,j,k);
		double vp1 = vs(i,j+1,k);
		double vm1 = vs(i,j-1,k);
		rtmp[0] = rtmp[0] + dy1ty1*(fu(0,i,j+1,k) - 2.0*fu(0,i,j,k) + fu(0,i,j-1,k)) - ty2*(fu(2,i,j+1,k)-fu(2,i,j-1,k));
		rtmp[1] = rtmp[1] + dy2ty1*(fu(1,i,j+1,k) - 2.0*fu(1,i,j,k) + fu(1,i,j-1,k)) + yycon2*(us(i,j+1,k)-2.0*us(i,j,k)+us(i,j-1,k)) - ty2*(fu(1,i,j+1,k)*vp1-fu(1,i,j-1,k)*vm1);
		rtmp[2] = rtmp[2] + dy3ty1*(fu(2,i,j+1,k) - 2.0*fu(2,i,j,k) + fu(2,i,j-1,k)) + yycon2*con43*(vp1-2.0*vijk+vm1) - ty2*(fu(2,i,j+1,k)*vp1-fu(2,i,j-1,k)*vm1+(fu(4,i,j+1,k)-square(i,j+1,k)-fu(4,i,j-1,k)+square(i,j-1,k))*c2);
		rtmp[3] = rtmp[3] + dy4ty1*(fu(3,i,j+1,k) - 2.0*fu(3,i,j,k) + fu(3,i,j-1,k)) + yycon2*(ws(i,j+1,k)-2.0*ws(i,j,k)+ws(i,j-1,k))-ty2*(fu(3,i,j+1,k)*vp1-fu(3,i,j-1,k)*vm1);
		rtmp[4] = rtmp[4] + dy5ty1*(fu(4,i,j+1,k) - 2.0*fu(4,i,j,k) + fu(4,i,j-1,k)) + yycon3*(qs(i,j+1,k)-2.0*qs(i,j,k)+qs(i,j-1,k)) + yycon4*(vp1*vp1-2.0*vijk*vijk+vm1*vm1) +
				yycon5*(fu(4,i,j+1,k)*rho_i(i,j+1,k)-2.0*fu(4,i,j,k)*rho_i(i,j,k)+fu(4,i,j-1,k)*rho_i(i,j-1,k)) - ty2*((c1*fu(4,i,j+1,k)-c2*square(i,j+1,k))*vp1 - (c1*fu(4,i,j-1,k)-c2*square(i,j-1,k))*vm1);
		//---------------------------------------------------------------------
		//      add fourth order eta-direction dissipation         
		//---------------------------------------------------------------------
		if (j == 1) {
			for (m = 0; m < 5; m++) rtmp[m] = rtmp[m] - dssp*(5.0*fu(m,i,j,k)-4.0*fu(m,i,j+1,k)+fu(m,i,j+2,k));
		} else if (j == 2) {
			for (m = 0; m < 5; m++) rtmp[m] = rtmp[m] - dssp*(-4.0*fu(m,i,j-1,k)+6.0*fu(m,i,j,k)-4.0*fu(m,i,j+1,k)+fu(m,i,j+2,k));
		} else if (j >= 3 && j < ny-3) {
			for (m = 0; m < 5; m++) rtmp[m] = rtmp[m] - dssp*(fu(m,i,j-2,k)-4.0*fu(m,i,j-1,k)+6.0*fu(m,i,j,k)-4.0*fu(m,i,j+1,k)+fu(m,i,j+2,k));
		} else if (j == ny-3) {
			for (m = 0; m < 5; m++) rtmp[m] = rtmp[m] - dssp*(fu(m,i,j-2,k)-4.0*fu(m,i,j-1,k)+6.0*fu(m,i,j,k)-4.0*fu(m,i,j+1,k));
		} else if (j == ny-2) {
			for (m = 0; m < 5; m++) rtmp[m] = rtmp[m] - dssp*(fu(m,i,j-2,k)-4.0*fu(m,i,j-1,k)+5.0*fu(m,i,j,k));
		}
		//---------------------------------------------------------------------
		//      compute zeta-direction fluxes 
		//---------------------------------------------------------------------
		double wijk = ws(i,j,k);
		double wp1 = ws(i,j,k+1);
		double wm1 = ws(i,j,k-1);

		rtmp[0] = rtmp[0] + dz1tz1*(fu(0,i,j,k+1)-2.0*fu(0,i,j,k)+fu(0,i,j,k-1)) - tz2*(fu(3,i,j,k+1)-fu(3,i,j,k-1));
		rtmp[1] = rtmp[1] + dz2tz1*(fu(1,i,j,k+1)-2.0*fu(1,i,j,k)+fu(1,i,j,k-1)) + zzcon2*(us(i,j,k+1)-2.0*us(i,j,k)+us(i,j,k-1)) - tz2*(fu(1,i,j,k+1)*wp1-fu(1,i,j,k-1)*wm1);
		rtmp[2] = rtmp[2] + dz3tz1*(fu(2,i,j,k+1)-2.0*fu(2,i,j,k)+fu(2,i,j,k-1)) + zzcon2*(vs(i,j,k+1)-2.0*vs(i,j,k)+vs(i,j,k-1)) - tz2*(fu(2,i,j,k+1)*wp1-fu(2,i,j,k-1)*wm1);
		rtmp[3] = rtmp[3] + dz4tz1*(fu(3,i,j,k+1)-2.0*fu(3,i,j,k)+fu(3,i,j,k-1)) + zzcon2*con43*(wp1-2.0*wijk+wm1) - tz2*(fu(3,i,j,k+1)*wp1-fu(3,i,j,k-1)*wm1+(fu(4,i,j,k+1)-square(i,j,k+1)-fu(4,i,j,k-1)+square(i,j,k-1))*c2);
		rtmp[4] = rtmp[4] + dz5tz1*(fu(4,i,j,k+1)-2.0*fu(4,i,j,k)+fu(4,i,j,k-1)) + zzcon3*(qs(i,j,k+1)-2.0*qs(i,j,k)+qs(i,j,k-1)) + zzcon4*(wp1*wp1-2.0*wijk*wijk+wm1*wm1) +
			zzcon5*(fu(4,i,j,k+1)*rho_i(i,j,k+1)-2.0*fu(4,i,j,k)*rho_i(i,j,k)+fu(4,i,j,k-1)*rho_i(i,j,k-1)) - tz2*((c1*fu(4,i,j,k+1)-c2*square(i,j,k+1))*wp1-(c1*fu(4,i,j,k-1)-c2*square(i,j,k-1))*wm1);
		//---------------------------------------------------------------------
		//      add fourth order zeta-direction dissipation                
		//---------------------------------------------------------------------
		if (k == 1) {
			for (m = 0; m < 5; m++)	rtmp[m] = rtmp[m] - dssp*(5.0*fu(m,i,j,k)-4.0*fu(m,i,j,k+1)+fu(m,i,j,k+2));
		} else if (k == 2) {
			for (m = 0; m < 5; m++) rtmp[m] = rtmp[m] - dssp*(-4.0*fu(m,i,j,k-1)+6.0*fu(m,i,j,k)-4.0*fu(m,i,j,k+1)+fu(m,i,j,k+2));
		} else if (k >= 3 && k < nz-3) {
			for (m = 0; m < 5; m++) rtmp[m] = rtmp[m] - dssp*(fu(m,i,j,k-2)-4.0*fu(m,i,j,k-1)+6.0*fu(m,i,j,k)-4.0*fu(m,i,j,k+1)+fu(m,i,j,k+2));
		} else if (k == nz-3) {
			for (m = 0; m < 5; m++) rtmp[m] = rtmp[m] - dssp*(fu(m,i,j,k-2)-4.0*fu(m,i,j,k-1)+6.0*fu(m,i,j,k)-4.0*fu(m,i,j,k+1));
		} else if (k == nz-2) {
			for (m = 0; m < 5; m++) rtmp[m] = rtmp[m] - dssp*(fu(m,i,j,k-2)-4.0*fu(m,i,j,k-1)+5.0*fu(m,i,j,k));
		}

		for (m = 0; m < 5; m++) rtmp[m] *= dt;
	}

	for (m = 0; m < 5; m++) rhs(m,i,j,k) = rtmp[m];
}

//---------------------------------------------------------------------
// adi: txinvr
//---------------------------------------------------------------------
__global__ static void txinvr_kernel ( double *rho_i, double *us, double *vs, double *ws, double *speed, double *qs, double *rhs, const int nx, const int ny, const int nz) {
	int i, j, k;

	k = blockIdx.y+1;
	j = blockIdx.x+1;
	i = threadIdx.x+1;

	double ru1 = rho_i(i,j,k);
	double uu = us(i,j,k);
	double vv = vs(i,j,k);
	double ww = ws(i,j,k);
	double ap = speed(i,j,k);
	double ac2inv = 1.0/( ap*ap );

	double r1 = rhs(0,i,j,k);
	double r2 = rhs(1,i,j,k);
	double r3 = rhs(2,i,j,k);
	double r4 = rhs(3,i,j,k);
	double r5 = rhs(4,i,j,k);

	double t1 = c2*ac2inv*(qs(i,j,k)*r1 - uu*r2  - vv*r3 - ww*r4 + r5);
	double t2 = bt * ru1 * ( uu * r1 - r2 );
	double t3 = ( bt * ru1 * ap ) * t1;

	rhs(0,i,j,k) = r1 - t1;
	rhs(1,i,j,k) = -ru1*(ww*r1-r4);
	rhs(2,i,j,k) = ru1*(vv*r1-r3);
	rhs(3,i,j,k) = -t2+t3;
	rhs(4,i,j,k) = t2+t3;
}

//---------------------------------------------------------------------
// adi: x_solve
//---------------------------------------------------------------------
#define lhs(m,i,j,k) lhs[(j-1)+(ny-2)*((k-1)+(nz-2)*((i)+nx*(m-3)))]
#define lhsp(m,i,j,k) lhs[(j-1)+(ny-2)*((k-1)+(nz-2)*((i)+nx*(m+4)))]
#define lhsm(m,i,j,k) lhs[(j-1)+(ny-2)*((k-1)+(nz-2)*((i)+nx*(m-3+2)))]
#define rtmp(m,i,j,k) rstmp[(j)+ny*((k)+nz*((i)+nx*(m)))]
__global__ static void x_solve_kernel (double *rho_i, double *us, double *speed, double *rhs, double *lhs, double *rstmp, const int nx, const int ny, const int nz) {
	int i, j, k, m;
	double rhon[3], cv[3], _ls[3][5], _lp[3][5], _rs[3][5], fac1;
  double zero;

	k = blockIdx.x*blockDim.x+threadIdx.x+1;
	j = blockIdx.y*blockDim.y+threadIdx.y+1;
	if (k >= nz-1 || j >= ny-1) return;

	//---------------------------------------------------------------------
	// Computes the left hand side for the three x-factors  
	//---------------------------------------------------------------------
	//---------------------------------------------------------------------
	//     zap the whole left hand side for starters
	//---------------------------------------------------------------------
	_ls[0][0] = (double)0.0;
	_ls[0][1] = (double)0.0;
	_ls[0][2] = (double)1.0;
	_ls[0][3] = (double)0.0;
	_ls[0][4] = (double)0.0;
  lhsp(0,0,j,k) = (double)0.0;
  lhsp(1,0,j,k) = (double)0.0;
  lhsp(2,0,j,k) = (double)1.0;
  lhsp(3,0,j,k) = (double)0.0;
  lhsp(4,0,j,k) = (double)0.0;
  zero = (double)0.0;

	//---------------------------------------------------------------------
	// first fill the lhs for the u-eigenvalue                          
	//--------------------------------------------------------------------
	for (i = 0; i < 3; i++) {
		fac1 = c3c4*rho_i(i,j,k);
		//rhon[i] = max(max(max(dx2+con43*fac1, dx5+c1c5*fac1), dxmax+fac1), zero+dx1);
    if (dx2+con43*fac1>dx5+c1c5*fac1)
      rhon[i] = dx2+con43*fac1;
    else
      rhon[i] = dx5+c1c5*fac1;
    if (rhon[i]<dxmax+fac1)
      rhon[i] = dxmax+fac1;
    if (rhon[i]<zero+dx1)
      rhon[i] = zero+dx1;


		cv[i] = us(i,j,k);
	}
	_ls[1][0] = (double)0.0;
	_ls[1][1] = - dttx2 * cv[0] - dtx1 * rhon[0];
	_ls[1][2] = 1.0 + c2dttx1 * rhon[1];
	_ls[1][3] = dttx2 * cv[2] - dtx1 * rhon[2];
	_ls[1][4] = (double)0.0;
	_ls[1][2] += comz5;
	_ls[1][3] -= comz4;
	_ls[1][4] += comz1;
	for (m = 0; m < 5; m++) lhsp(m,1,j,k) = _ls[1][m];
	rhon[0] = rhon[1]; rhon[1] = rhon[2];
	cv[0] = cv[1]; cv[1] = cv[2];
	for (m = 0; m < 3; m++) {
		_rs[0][m] = rhs(m,0,j,k);
		_rs[1][m] = rhs(m,1,j,k);
	}

	//---------------------------------------------------------------------
	//      perform the Thomas algorithm; first, FORWARD ELIMINATION     
	//---------------------------------------------------------------------
	for (i = 0; i < nx-2; i++) {
		//---------------------------------------------------------------------
		// first fill the lhs for the u-eigenvalue                          
		//---------------------------------------------------------------------
		if (i+2 == nx-1) {
			_ls[2][0] = (double)0.0;
			_ls[2][1] = (double)0.0;
			_ls[2][2] = (double)1.0;
			_ls[2][3] = (double)0.0;
			_ls[2][4] = (double)0.0;
      lhsp(0,i+2,j,k) = (double)0.0;
      lhsp(1,i+2,j,k) = (double)0.0;
      lhsp(2,i+2,j,k) = (double)1.0;
      lhsp(3,i+2,j,k) = (double)0.0;
      lhsp(4,i+2,j,k) = (double)0.0;

		} else {
			fac1 = c3c4*rho_i(i+3,j,k);
			//rhon[2] = max(max(max(dx2+con43*fac1, dx5+c1c5*fac1), dxmax+fac1), zero+dx1);
      if (dx2+con43*fac1>dx5+c1c5*fac1)
        rhon[2] = dx2+con43*fac1;
      else
        rhon[2] = dx5+c1c5*fac1;
      if (rhon[2]<dxmax+fac1)
        rhon[2] = dxmax+fac1;
      if (rhon[2]<zero+dx1)
        rhon[2] = zero+dx1;

			cv[2] = us(i+3,j,k);
			_ls[2][0] = (double)0.0;
			_ls[2][1] = - dttx2 * cv[0] - dtx1 * rhon[0];
			_ls[2][2] = 1.0 + c2dttx1 * rhon[1];
			_ls[2][3] = dttx2 * cv[2] - dtx1 * rhon[2];
			_ls[2][4] = (double)0.0;
			//---------------------------------------------------------------------
			//      add fourth order dissipation                                  
			//---------------------------------------------------------------------
			if (i+2 == 2) {
				_ls[2][1] -= comz4;
				_ls[2][2] += comz6;
				_ls[2][3] -= comz4;
				_ls[2][4] += comz1;
			} else if (i+2 >= 3 && i+2 < nx-3) {
				_ls[2][0] += comz1;
				_ls[2][1] -= comz4;
				_ls[2][2] += comz6;
				_ls[2][3] -= comz4;
				_ls[2][4] += comz1;
			} else if (i+2 == nx-3) {
				_ls[2][0] += comz1;
				_ls[2][1] -= comz4;
				_ls[2][2] += comz6;
				_ls[2][3] -= comz4;
			} else if (i+2 == nx-2) {
				_ls[2][0] += comz1;
				_ls[2][1] -= comz4;
				_ls[2][2] += comz5;
			}

			//---------------------------------------------------------------------
			//      store computed lhs for later reuse
			//---------------------------------------------------------------------
			for (m = 0; m < 5; m++) lhsp(m,i+2,j,k) = _ls[2][m];
			rhon[0] = rhon[1]; rhon[1] = rhon[2];
			cv[0] = cv[1]; cv[1] = cv[2];
		}

		//---------------------------------------------------------------------
		//      load rhs values for current iteration
		//---------------------------------------------------------------------
		for (m = 0; m < 3; m++) _rs[2][m] = rhs(m,i+2,j,k);

		//---------------------------------------------------------------------
		//      perform current iteration
		//---------------------------------------------------------------------
		fac1 = 1.0/_ls[0][2];
		_ls[0][3] *= fac1;
		_ls[0][4] *= fac1;
		for (m = 0; m < 3; m++) _rs[0][m] *= fac1;
		_ls[1][2] -= _ls[1][1] * _ls[0][3];
		_ls[1][3] -= _ls[1][1] * _ls[0][4];
		for (m = 0; m < 3; m++) _rs[1][m] -= _ls[1][1] * _rs[0][m];
		_ls[2][1] -= _ls[2][0] * _ls[0][3];
		_ls[2][2] -= _ls[2][0] * _ls[0][4];
		for (m = 0; m < 3; m++) _rs[2][m] -= _ls[2][0] * _rs[0][m];

		//---------------------------------------------------------------------
		//      store computed lhs and prepare data for next iteration
		//	rhs is stored in a temp array such that write accesses are coalesced
		//---------------------------------------------------------------------
		lhs(3,i,j,k) = _ls[0][3];
		lhs(4,i,j,k) = _ls[0][4];
		for (m = 0; m < 5; m++) {
			_ls[0][m] = _ls[1][m];
			_ls[1][m] = _ls[2][m];
		}
		for (m = 0; m < 3; m++) {
			rtmp(m,i,j,k) = _rs[0][m];
			_rs[0][m] = _rs[1][m];
			_rs[1][m] = _rs[2][m];
		}
	}

	//---------------------------------------------------------------------
	//      The last two rows in this zone are a bit different, 
	//      since they do not have two more rows available for the
	//      elimination of off-diagonal entries
	//---------------------------------------------------------------------
	i = nx-2;
	fac1 = 1.0/_ls[0][2];
	_ls[0][3] *= fac1;
	_ls[0][4] *= fac1;
	for (m = 0; m < 3; m++) _rs[0][m] *= fac1;
	_ls[1][2] -= _ls[1][1] * _ls[0][3];
	_ls[1][3] -= _ls[1][1] * _ls[0][4];
	for (m = 0; m < 3; m++) _rs[1][m] -= _ls[1][1] * _rs[0][m];
	//---------------------------------------------------------------------
	//            scale the last row immediately 
	//---------------------------------------------------------------------
	fac1 = 1.0/_ls[1][2];
	for (m = 0; m < 3; m++) _rs[1][m] *= fac1;
	lhs(3,nx-2,j,k) = _ls[0][3];
	lhs(4,nx-2,j,k) = _ls[0][4];

	//---------------------------------------------------------------------
	//      subsequently, fill the other factors u+c, u-c 
	//---------------------------------------------------------------------
	for (i = 0; i < 3; i++) cv[i] = speed(i,j,k);
	for (m = 0; m < 5; m++) {
		_ls[0][m] = lhsp(m,0,j,k);
		_lp[0][m] = lhsp(m,0,j,k);
		_ls[1][m] = lhsp(m,1,j,k);
		_lp[1][m] = lhsp(m,1,j,k);
	}
	_lp[1][1] -= dttx2 * cv[0];
	_lp[1][3] += dttx2 * cv[2];
	_ls[1][1] += dttx2 * cv[0];
	_ls[1][3] -= dttx2 * cv[2];
	cv[0] = cv[1]; 
  cv[1] = cv[2];
	_rs[0][3] = rhs(3,0,j,k);
	_rs[0][4] = rhs(4,0,j,k);
	_rs[1][3] = rhs(3,1,j,k);
	_rs[1][4] = rhs(4,1,j,k);
	//---------------------------------------------------------------------
	//      do the u+c and the u-c factors               
	//---------------------------------------------------------------------
	for (i = 0; i < nx-2; i++) {
		//---------------------------------------------------------------------
		//      first, fill the other factors u+c, u-c 
		//---------------------------------------------------------------------
		for (m = 0; m < 5; m++) {
			_ls[2][m] = lhsp(m,i+2,j,k);
			_lp[2][m] = lhsp(m,i+2,j,k);
		}
		_rs[2][3] = rhs(3,i+2,j,k);
		_rs[2][4] = rhs(4,i+2,j,k);

		if (i+2 < nx-1) {
			cv[2] = speed(i+3,j,k);
			_lp[2][1] -= dttx2 * cv[0];
			_lp[2][3] += dttx2 * cv[2];
			_ls[2][1] += dttx2 * cv[0];
			_ls[2][3] -= dttx2 * cv[2];
			cv[0] = cv[1]; 
      cv[1] = cv[2];
		}

		m = 3;
		fac1 = 1.0/_lp[0][2];
		_lp[0][3] *= fac1;
		_lp[0][4] *= fac1;
		_rs[0][m] *= fac1;
		_lp[1][2] -= _lp[1][1]*_lp[0][3];
		_lp[1][3] -= _lp[1][1]*_lp[0][4];
		_rs[1][m] -= _lp[1][1]*_rs[0][m];
		_lp[2][1] -= _lp[2][0]*_lp[0][3];
		_lp[2][2] -= _lp[2][0]*_lp[0][4];
		_rs[2][m] -= _lp[2][0]*_rs[0][m];

		m = 4;
		fac1 = 1.0/_ls[0][2];
		_ls[0][3] *= fac1;
		_ls[0][4] *= fac1;
		_rs[0][m] *= fac1;
		_ls[1][2] -= _ls[1][1]*_ls[0][3];
		_ls[1][3] -= _ls[1][1]*_ls[0][4];
		_rs[1][m] -= _ls[1][1]*_rs[0][m];
		_ls[2][1] -= _ls[2][0]*_ls[0][3];
		_ls[2][2] -= _ls[2][0]*_ls[0][4];
		_rs[2][m] -= _ls[2][0]*_rs[0][m];

		//---------------------------------------------------------------------
		//      store computed lhs and prepare data for next iteration
		//	rhs is stored in a temp array such that write accesses are coalesced
		//---------------------------------------------------------------------
		for (m = 3; m < 5; m++) {
			lhsp(m,i,j,k) = _lp[0][m];
			lhsm(m,i,j,k) = _ls[0][m];
			rtmp(m,i,j,k) = _rs[0][m];
			_rs[0][m] = _rs[1][m];
			_rs[1][m] = _rs[2][m];
		}
		for (m = 0; m < 5; m++) {
			_lp[0][m] = _lp[1][m];
			_lp[1][m] = _lp[2][m];
			_ls[0][m] = _ls[1][m];
			_ls[1][m] = _ls[2][m];
		}
	}
	//---------------------------------------------------------------------
	//         And again the last two rows separately
	//---------------------------------------------------------------------
	i = nx-2;
	m = 3;
	fac1 = 1.0/_lp[0][2];
	_lp[0][3] *= fac1;
	_lp[0][4] *= fac1;
	_rs[0][m] *= fac1;
	_lp[1][2] -= _lp[1][1]*_lp[0][3];
	_lp[1][3] -= _lp[1][1]*_lp[0][4];
	_rs[1][m] -= _lp[1][1]*_rs[0][m];

	m = 4;
	fac1 = 1.0/_ls[0][2];
	_ls[0][3] *= fac1;
	_ls[0][4] *= fac1;
	_rs[0][m] *= fac1;
	_ls[1][2] -= _ls[1][1]*_ls[0][3];
	_ls[1][3] -= _ls[1][1]*_ls[0][4];
	_rs[1][m] -= _ls[1][1]*_rs[0][m];

	//---------------------------------------------------------------------
	//               Scale the last row immediately
	//---------------------------------------------------------------------
	_rs[1][3] /= _lp[1][2];
	_rs[1][4] /= _ls[1][2];

	//---------------------------------------------------------------------
	//                         BACKSUBSTITUTION 
	//---------------------------------------------------------------------
	for (m = 0; m < 3; m++) _rs[0][m] -= lhs(3,nx-2,j,k)*_rs[1][m];
	_rs[0][3] -= _lp[0][3]*_rs[1][3];
	_rs[0][4] -= _ls[0][3]*_rs[1][4];
	for (m = 0; m < 5; m++) {
		_rs[2][m] = _rs[1][m];
		_rs[1][m] = _rs[0][m];
	}

	for (i = nx-3; i >= 0; i--) {
		//---------------------------------------------------------------------
		//      The first three factors
		//---------------------------------------------------------------------
		for (m = 0; m < 3; m++) _rs[0][m] = rtmp(m,i,j,k) - lhs(3,i,j,k)*_rs[1][m] - lhs(4,i,j,k)*_rs[2][m];
		//---------------------------------------------------------------------
		//      And the remaining two
		//---------------------------------------------------------------------
		_rs[0][3] = rtmp(3,i,j,k) - lhsp(3,i,j,k)*_rs[1][3] - lhsp(4,i,j,k)*_rs[2][3];
		_rs[0][4] = rtmp(4,i,j,k) - lhsm(3,i,j,k)*_rs[1][4] - lhsm(4,i,j,k)*_rs[2][4];

		if (i+2 < nx-1) {
			//---------------------------------------------------------------------
			//      Do the block-diagonal inversion          
			//---------------------------------------------------------------------
				double r1 = _rs[2][0];
				double r2 = _rs[2][1];
				double r3 = _rs[2][2];
				double r4 = _rs[2][3];
				double r5 = _rs[2][4];
				double t1 = bt * r3;
				double t2 = 0.5 * (r4+r5);

				_rs[2][0] = -r2;
				_rs[2][1] =  r1;
				_rs[2][2] = bt * ( r4 - r5 );
				_rs[2][3] = -t1 + t2;
				_rs[2][4] =  t1 + t2;
		}

		for (m = 0; m < 5; m++) {
			rhs(m,i+2,j,k) = _rs[2][m];
			_rs[2][m] = _rs[1][m];
			_rs[1][m] = _rs[0][m];
		}
	}

	//---------------------------------------------------------------------
	//      Do the block-diagonal inversion          
	//---------------------------------------------------------------------
	double tf1 = bt * _rs[2][2];
	double tf2 = 0.5 * (_rs[2][3]+_rs[2][4]);
	rhs(0,1,j,k) = -_rs[2][1];
	rhs(1,1,j,k) =  _rs[2][0];
	rhs(2,1,j,k) = bt * ( _rs[2][3] - _rs[2][4] );
	rhs(3,1,j,k) = -tf1 + tf2;
	rhs(4,1,j,k) =  tf1 + tf2;

	for (m = 0; m < 5; m++) rhs(m,0,j,k) = _rs[1][m];
}
#undef lhs
#undef lhsp
#undef lhsm
#undef rtmp

//---------------------------------------------------------------------
// adi: y_solve
//---------------------------------------------------------------------
#define lhs(m,i,j,k) lhs[(i-1)+(nx-2)*((k-1)+(nz-2)*((j)+ny*(m-3)))]
#define lhsp(m,i,j,k) lhs[(i-1)+(nx-2)*((k-1)+(nz-2)*((j)+ny*(m+4)))]
#define lhsm(m,i,j,k) lhs[(i-1)+(nx-2)*((k-1)+(nz-2)*((j)+ny*(m-3+2)))]
#define rtmp(m,i,j,k) rstmp[(i)+nx*((k)+nz*((j)+ny*(m)))]
__global__ static void y_solve_kernel (double *rho_i, double *vs, double *speed, double *rhs, double *lhs, double *rstmp, const int nx, const int ny, const int nz) {
	int i, j, k, m;
	double rhoq[3], cv[3], _ls[3][5], _lp[3][5], _rs[3][5], fac1;
  double zero;

	k = blockIdx.x*blockDim.x+threadIdx.x+1;
	i = blockIdx.y*blockDim.y+threadIdx.y+1;
	if (k >= nz-1 || i >= nx-1) return;

	//---------------------------------------------------------------------
	// Computes the left hand side for the three y-factors   
	//---------------------------------------------------------------------
	//---------------------------------------------------------------------
	//     zap the whole left hand side for starters
	//---------------------------------------------------------------------
	_ls[0][0] = (double)0.0;
	_ls[0][1] = (double)0.0;
	_ls[0][2] = (double)1.0;
	_ls[0][3] = (double)0.0;
	_ls[0][4] = (double)0.0;
  lhsp(0,i,0,k) = (double)0.0;
  lhsp(1,i,0,k) = (double)0.0;
  lhsp(2,i,0,k) = (double)1.0;
  lhsp(3,i,0,k) = (double)0.0;
  lhsp(4,i,0,k) = (double)0.0;
  zero = (double)0.0;


	//---------------------------------------------------------------------
	//      first fill the lhs for the u-eigenvalue         
	//---------------------------------------------------------------------
	for (j = 0; j < 3; j++) {
		fac1 = c3c4*rho_i(i,j,k);
		//rhoq[j] = max(max(max(dy3+con43*fac1, dy5+c1c5*fac1), dymax+fac1), zero+dy1);
   if (dy3+con43*fac1>dy5+c1c5*fac1)
     rhoq[j] = dy3+con43*fac1;
   else
     rhoq[j] = dy5+c1c5*fac1;
   if (rhoq[j]<dymax+fac1)
     rhoq[j] = dymax+fac1;
   if (rhoq[j]<zero+dy1)
     rhoq[j] = zero+dy1;
		cv[j] = vs(i,j,k);
	}
	_ls[1][0] =  (double)0.0;
	_ls[1][1] = -dtty2*cv[0]-dty1 * rhoq[0];
	_ls[1][2] =  1.0 + c2dtty1 * rhoq[1];
	_ls[1][3] =  dtty2*cv[2]-dty1 * rhoq[2];
	_ls[1][4] =  (double)0.0;
	_ls[1][2] += comz5;
	_ls[1][3] -= comz4;
	_ls[1][4] += comz1;
	for (m = 0; m < 5; m++) lhsp(m,i,1,k) = _ls[1][m];
	rhoq[0] = rhoq[1]; 
  rhoq[1] = rhoq[2];
	cv[0] = cv[1]; cv[1] = cv[2];
	for (m = 0; m < 3; m++) {
		_rs[0][m] = rhs(m,i,0,k);
		_rs[1][m] = rhs(m,i,1,k);
	}

	//---------------------------------------------------------------------
	//                          FORWARD ELIMINATION  
	//---------------------------------------------------------------------
	for (j = 0; j < ny-2; j++) {
		//---------------------------------------------------------------------
		// first fill the lhs for the u-eigenvalue                          
		//---------------------------------------------------------------------
		if (j+2 == ny-1) {
			_ls[2][0] = (double)0.0;
			_ls[2][1] = (double)0.0;
			_ls[2][2] = (double)1.0;
			_ls[2][3] = (double)0.0;
			_ls[2][4] = (double)0.0;
      lhsp(0,i,j+2,k) = (double)0.0;
      lhsp(1,i,j+2,k) = (double)0.0;
      lhsp(2,i,j+2,k) = (double)1.0;
      lhsp(3,i,j+2,k) = (double)0.0;
      lhsp(4,i,j+2,k) = (double)0.0;
		} else {
			fac1 = c3c4*rho_i(i,j+3,k);
			//rhoq[2] = max(max(max(dy3+con43*fac1, dy5+c1c5*fac1), dymax+fac1), zero+dy1);
      if (dy3+con43*fac1>dy5+c1c5*fac1)
        rhoq[2] = dy3+con43*fac1;
      else
        rhoq[2] = dy5+c1c5*fac1;
      if (rhoq[2]<dymax+fac1)
        rhoq[2] = dymax+fac1;
      if (rhoq[2]<zero+dy1)
        rhoq[2] = zero+dy1;
			cv[2] = vs(i,j+3,k);
			_ls[2][0] =  (double)0.0;
			_ls[2][1] = -dtty2*cv[0]-dty1 * rhoq[0];
			_ls[2][2] =  1.0 + c2dtty1 * rhoq[1];
			_ls[2][3] =  dtty2*cv[2]-dty1 * rhoq[2];
			_ls[2][4] =  (double)0.0;
			//---------------------------------------------------------------------
			//      add fourth order dissipation                             
			//---------------------------------------------------------------------
			if (j+2 == 2) {
				_ls[2][1] -= comz4;
				_ls[2][2] += comz6;
				_ls[2][3] -= comz4;
				_ls[2][4] += comz1;
			} else if (j+2 >= 3 && j+2 < ny-3) {
				_ls[2][0] += comz1;
				_ls[2][1] -= comz4;
				_ls[2][2] += comz6;
				_ls[2][3] -= comz4;
				_ls[2][4] += comz1;
			} else if (j+2 == ny-3) {
				_ls[2][0] += comz1;
				_ls[2][1] -= comz4;
				_ls[2][2] += comz6;
				_ls[2][3] -= comz4;
			} else if (j+2 == ny-2) {
				_ls[2][0] += comz1;
				_ls[2][1] -= comz4;
				_ls[2][2] += comz5;
			}

			//---------------------------------------------------------------------
			//      store computed lhs for later reuse
			//---------------------------------------------------------------------
			for (m = 0; m < 5; m++) lhsp(m,i,j+2,k) = _ls[2][m];
			rhoq[0] = rhoq[1]; rhoq[1] = rhoq[2];
			cv[0] = cv[1]; cv[1] = cv[2];
		}

		//---------------------------------------------------------------------
		//      load rhs values for current iteration
		//---------------------------------------------------------------------
		for (m = 0; m < 3; m++) _rs[2][m] = rhs(m,i,j+2,k);

		//---------------------------------------------------------------------
		//      perform current iteration
		//---------------------------------------------------------------------
		fac1 = 1.0/_ls[0][2];
		_ls[0][3] *= fac1;
		_ls[0][4] *= fac1;
		for (m = 0; m < 3; m++) _rs[0][m] *= fac1;
		_ls[1][2] -= _ls[1][1] * _ls[0][3];
		_ls[1][3] -= _ls[1][1] * _ls[0][4];
		for (m = 0; m < 3; m++) _rs[1][m] -= _ls[1][1] * _rs[0][m];
		_ls[2][1] -= _ls[2][0] * _ls[0][3];
		_ls[2][2] -= _ls[2][0] * _ls[0][4];
		for (m = 0; m < 3; m++) _rs[2][m] -= _ls[2][0] * _rs[0][m];

		//---------------------------------------------------------------------
		//      store computed lhs and prepare data for next iteration
		//	rhs is stored in a temp array such that write accesses are coalesced
		//---------------------------------------------------------------------
		lhs(3,i,j,k) = _ls[0][3];
		lhs(4,i,j,k) = _ls[0][4];
		for (m = 0; m < 5; m++) {
			_ls[0][m] = _ls[1][m];
			_ls[1][m] = _ls[2][m];
		}
		for (m = 0; m < 3; m++) {
			rtmp(m,i,j,k) = _rs[0][m];
			_rs[0][m] = _rs[1][m];
			_rs[1][m] = _rs[2][m];
		}
	}
	//---------------------------------------------------------------------
	//      The last two rows in this zone are a bit different, 
	//      since they do not have two more rows available for the
	//      elimination of off-diagonal entries
	//---------------------------------------------------------------------
	j = ny-2;
	fac1 = 1.0/_ls[0][2];
	_ls[0][3] *= fac1;
	_ls[0][4] *= fac1;
	for (m = 0; m < 3; m++) _rs[0][m] *= fac1;
	_ls[1][2] -= _ls[1][1] * _ls[0][3];
	_ls[1][3] -= _ls[1][1] * _ls[0][4];
	for (m = 0; m < 3; m++) _rs[1][m] -= _ls[1][1] * _rs[0][m];
	//---------------------------------------------------------------------
	//            scale the last row immediately 
	//---------------------------------------------------------------------
	fac1 = 1.0/_ls[1][2];
	for (m = 0; m < 3; m++) _rs[1][m] *= fac1;
	lhs(3,i,ny-2,k) = _ls[0][3];
	lhs(4,i,ny-2,k) = _ls[0][4];

	//---------------------------------------------------------------------
	//      do the u+c and the u-c factors                 
	//---------------------------------------------------------------------
	for (j = 0; j < 3; j++) cv[j] = speed(i,j,k);
	for (m = 0; m < 5; m++) {
		_ls[0][m] = lhsp(m,i,0,k);
		_lp[0][m] = lhsp(m,i,0,k);
		_ls[1][m] = lhsp(m,i,1,k);
		_lp[1][m] = lhsp(m,i,1,k);
	}
	_lp[1][1] -= dtty2*cv[0];
	_lp[1][3] += dtty2*cv[2];
	_ls[1][1] += dtty2*cv[0];
	_ls[1][3] -= dtty2*cv[2];
	cv[0] = cv[1]; 
  cv[1] = cv[2];
	_rs[0][3] = rhs(3,i,0,k);
	_rs[0][4] = rhs(4,i,0,k);
	_rs[1][3] = rhs(3,i,1,k);
	_rs[1][4] = rhs(4,i,1,k);
	for (j = 0; j < ny-2; j++) {
		for (m = 0; m < 5; m++) {
			_ls[2][m] = lhsp(m,i,j+2,k);
			_lp[2][m] = lhsp(m,i,j+2,k);
		}
		_rs[2][3] = rhs(3,i,j+2,k);
		_rs[2][4] = rhs(4,i,j+2,k);
		if (j+2 < ny-1) {
			cv[2] = speed(i,j+3,k);
			_lp[2][1] -= dtty2*cv[0];
			_lp[2][3] += dtty2*cv[2];
			_ls[2][1] += dtty2*cv[0];
			_ls[2][3] -= dtty2*cv[2];
			cv[0] = cv[1]; 
      cv[1] = cv[2];
		}

		fac1 = 1.0/_lp[0][2];
		m = 3;
		_lp[0][3] *= fac1;
		_lp[0][4] *= fac1;
		_rs[0][m] *= fac1;
		_lp[1][2] -= _lp[1][1] * _lp[0][3];
		_lp[1][3] -= _lp[1][1] * _lp[0][4];
		_rs[1][m] -= _lp[1][1] * _rs[0][m];
		_lp[2][1] -= _lp[2][0] * _lp[0][3];
		_lp[2][2] -= _lp[2][0] * _lp[0][4];
		_rs[2][m] -= _lp[2][0] * _rs[0][m];

		m = 4;
		fac1 = 1.0/_ls[0][2];
		_ls[0][3] *= fac1;
		_ls[0][4] *= fac1;
		_rs[0][m] *= fac1;
		_ls[1][2] -= _ls[1][1] * _ls[0][3];
		_ls[1][3] -= _ls[1][1] * _ls[0][4];
		_rs[1][m] -= _ls[1][1] * _rs[0][m];
		_ls[2][1] -= _ls[2][0] * _ls[0][3];
		_ls[2][2] -= _ls[2][0] * _ls[0][4];
		_rs[2][m] -= _ls[2][0] * _rs[0][m];

		//---------------------------------------------------------------------
		//      store computed lhs and prepare data for next iteration
		//	rhs is stored in a temp array such that write accesses are coalesced
		//---------------------------------------------------------------------
		for (m = 3; m < 5; m++) {
			lhsp(m,i,j,k) = _lp[0][m];
			lhsm(m,i,j,k) = _ls[0][m];
			rtmp(m,i,j,k) = _rs[0][m];
			_rs[0][m] = _rs[1][m];
			_rs[1][m] = _rs[2][m];
		}
		for (m = 0; m < 5; m++) {
			_lp[0][m] = _lp[1][m];
			_lp[1][m] = _lp[2][m];
			_ls[0][m] = _ls[1][m];
			_ls[1][m] = _ls[2][m];
		}
	}
	//---------------------------------------------------------------------
	//         And again the last two rows separately
	//---------------------------------------------------------------------
	j = ny-2;
	m = 3;
	fac1 = 1.0/_lp[0][2];
	_lp[0][3] *= fac1;
	_lp[0][4] *= fac1;
	_rs[0][m] *= fac1;
	_lp[1][2] -= _lp[1][1] * _lp[0][3];
	_lp[1][3] -= _lp[1][1] * _lp[0][4];
	_rs[1][m] -= _lp[1][1] * _rs[0][m];

	m = 4;
	fac1 = 1.0/_ls[0][2];
	_ls[0][3] *= fac1;
	_ls[0][4] *= fac1;
	_rs[0][m] *= fac1;
	_ls[1][2] -= _ls[1][1] * _ls[0][3];
	_ls[1][3] -= _ls[1][1] * _ls[0][4];
	_rs[1][m] -= _ls[1][1] * _rs[0][m];
	//---------------------------------------------------------------------
	//               Scale the last row immediately 
	//---------------------------------------------------------------------
	_rs[1][3] /= _lp[1][2];
	_rs[1][4] /= _ls[1][2];

	//---------------------------------------------------------------------
	//                         BACKSUBSTITUTION 
	//---------------------------------------------------------------------
	for (m = 0; m < 3; m++) _rs[0][m] -= lhs(3,i,ny-2,k) * _rs[1][m];
	_rs[0][3] -= _lp[0][3] * _rs[1][3];
	_rs[0][4] -= _ls[0][3] * _rs[1][4];
	for (m = 0; m < 5; m++) {
		_rs[2][m] = _rs[1][m];
		_rs[1][m] = _rs[0][m];
	}
	for (j = ny-3; j >= 0; j--) {
		//---------------------------------------------------------------------
		//      The first three factors
		//---------------------------------------------------------------------
		for (m = 0; m < 3; m++) _rs[0][m] = rtmp(m,i,j,k) - lhs(3,i,j,k)*_rs[1][m] - lhs(4,i,j,k)*_rs[2][m];
		//---------------------------------------------------------------------
		//      And the remaining two
		//---------------------------------------------------------------------
		_rs[0][3] = rtmp(3,i,j,k) - lhsp(3,i,j,k)*_rs[1][3] - lhsp(4,i,j,k)*_rs[2][3];
		_rs[0][4] = rtmp(4,i,j,k) - lhsm(3,i,j,k)*_rs[1][4] - lhsm(4,i,j,k)*_rs[2][4];
	
		if (j+2 < ny-1) {
			//---------------------------------------------------------------------
			//   block-diagonal matrix-vector multiplication                       
			//---------------------------------------------------------------------
			double r1 = _rs[2][0];
			double r2 = _rs[2][1];
			double r3 = _rs[2][2];
			double r4 = _rs[2][3];
			double r5 = _rs[2][4];

			double t1 = bt * r1;
			double t2 = 0.5 * ( r4 + r5 );

			_rs[2][0] =  bt * ( r4 - r5 );
			_rs[2][1] = -r3;
			_rs[2][2] =  r2;
			_rs[2][3] = -t1 + t2;
			_rs[2][4] =  t1 + t2;
		}

		for (m = 0; m < 5; m++) {
			rhs(m,i,j+2,k) = _rs[2][m];
			_rs[2][m] = _rs[1][m];
			_rs[1][m] = _rs[0][m];
		}
	}

	//---------------------------------------------------------------------
	//   block-diagonal matrix-vector multiplication                       
	//---------------------------------------------------------------------
	double tf1 = bt * _rs[2][0];
	double tf2 = 0.5 * ( _rs[2][3] + _rs[2][4] );
	rhs(0,i,1,k) =  bt * ( _rs[2][3] - _rs[2][4] );
	rhs(1,i,1,k) = -_rs[2][2];
	rhs(2,i,1,k) =  _rs[2][1];
	rhs(3,i,1,k) = -tf1 + tf2;
	rhs(4,i,1,k) =  tf1 + tf2;

	for (m = 0; m < 5; m++) rhs(m,i,0,k) = _rs[1][m];
}
#undef lhs
#undef lhsp
#undef lhsm
#undef rtmp

//---------------------------------------------------------------------
// adi: z_solve
//---------------------------------------------------------------------
#define lhs(m,i,j,k) lhs[(i-1)+(nx-2)*((j-1)+(ny-2)*((k)+nz*(m-3)))]
#define lhsp(m,i,j,k) lhs[(i-1)+(nx-2)*((j-1)+(ny-2)*((k)+nz*(m+4)))]
#define lhsm(m,i,j,k) lhs[(i-1)+(nx-2)*((j-1)+(ny-2)*((k)+nz*(m-3+2)))]
#define rtmp(m,i,j,k) rstmp[(i)+nx*((j)+ny*((k)+nz*(m)))]
__global__ static void z_solve_kernel (double *rho_i, double *us, double *vs, double *ws, double *speed, double *qs, double *fu, double *rhs, double *lhs, double *rstmp, const int nx, const int ny, const int nz) {
	int i, j, k, m;
	double rhos[3], cv[3], _ls[3][5], _lp[3][5], _rs[3][5], fac1;
  double zero;

	j = blockIdx.x*blockDim.x+threadIdx.x+1;
	i = blockIdx.y*blockDim.y+threadIdx.y+1;
	if (j >= ny-1 || i >= nx-1) return;

	//---------------------------------------------------------------------
	// Computes the left hand side for the three z-factors   
	//---------------------------------------------------------------------
	//---------------------------------------------------------------------
	//     zap the whole left hand side for starters
	//---------------------------------------------------------------------
	_ls[0][0] = (double)0.0;
	_ls[0][1] = (double)0.0;
	_ls[0][2] = (double)1.0;
	_ls[0][3] = (double)0.0;
	_ls[0][4] = (double)0.0;
  lhsp(0,i,j,0) = (double)0.0;
  lhsp(1,i,j,0) = (double)0.0;
  lhsp(2,i,j,0) = (double)1.0;
  lhsp(3,i,j,0) = (double)0.0;
  lhsp(4,i,j,0) = (double)0.0;
  zero = (double)0.0;

	//---------------------------------------------------------------------
	// first fill the lhs for the u-eigenvalue                          
	//---------------------------------------------------------------------
	for (k = 0; k < 3; k++) {
		fac1 = c3c4*rho_i(i,j,k);
		//rhos[k] = max(max(max(dz4+con43*fac1, dz5+c1c5*fac1), dzmax+fac1), zero+dz1);
    if (dz4+con43*fac1>dz5+c1c5*fac1)
      rhos[k] = dz4+con43*fac1;
    else
      rhos[k] = dz5+c1c5*fac1;
    if (rhos[k]<dzmax+fac1)
      rhos[k] = dzmax+fac1;
    if (rhos[k]<zero+dz1)
      rhos[k] = zero+dz1;
		cv[k] = ws(i,j,k);
	}
	_ls[1][0] =  (double)0.0;
	_ls[1][1] = -dttz2*cv[0] - dtz1*rhos[0];
	_ls[1][2] =  1.0 + c2dttz1 * rhos[1];
	_ls[1][3] =  dttz2*cv[2] - dtz1*rhos[2];
	_ls[1][4]=  (double)0.0;
	_ls[1][2] += comz5;
	_ls[1][3] -= comz4;
	_ls[1][4] += comz1;
	for (m = 0; m < 5; m++) lhsp(m,i,j,1) = _ls[1][m];
	rhos[0] = rhos[1]; rhos[1] = rhos[2];
	cv[0] = cv[1]; cv[1] = cv[2];
	for (m = 0; m < 3; m++) {
		_rs[0][m] = rhs(m,i,j,0);
		_rs[1][m] = rhs(m,i,j,1);
	}

	//---------------------------------------------------------------------
	//                          FORWARD ELIMINATION  
	//---------------------------------------------------------------------
	for (k = 0; k < nz-2; k++) {
		//---------------------------------------------------------------------
		// first fill the lhs for the u-eigenvalue                          
		//---------------------------------------------------------------------
		if (k+2 == nz-1) {
			_ls[2][0] = (double)0.0;
			_ls[2][1] = (double)0.0;
			_ls[2][2] = (double)1.0;
			_ls[2][3] = (double)0.0;
			_ls[2][4] = (double)0.0;
      lhsp(0,i,j,k+2) = (double)0.0;
      lhsp(1,i,j,k+2) = (double)0.0;
      lhsp(2,i,j,k+2) = (double)1.0;
      lhsp(3,i,j,k+2) = (double)0.0;
      lhsp(4,i,j,k+2) = (double)0.0;
		} else {
			fac1 = c3c4*rho_i(i,j,k+3);
			//rhos[2] = max(max(max(dz4+con43*fac1, dz5+c1c5*fac1), dzmax+fac1), zero+dz1);
      if (dz4+con43*fac1>dz5+c1c5*fac1)
        rhos[2] = dz4+con43*fac1;
      else
        rhos[2] = dz5+c1c5*fac1;
      if (rhos[2]<dzmax+fac1)
        rhos[2] = dzmax+fac1;
      if (rhos[2]<zero+dz1)
        rhos[2] = zero+dz1;
			cv[2] = ws(i,j,k+3);
			_ls[2][0] =  (double)0.0;
			_ls[2][1] = -dttz2*cv[0] - dtz1*rhos[0];
			_ls[2][2] =  1.0 + c2dttz1 * rhos[1];
			_ls[2][3] =  dttz2*cv[2] - dtz1*rhos[2];
			_ls[2][4] =  (double)0.0;
			//---------------------------------------------------------------------
			//      add fourth order dissipation                                  
			//---------------------------------------------------------------------
			if (k+2 == 2) {
				_ls[2][1] -= comz4;
				_ls[2][2] += comz6;
				_ls[2][3] -= comz4;
				_ls[2][4] += comz1;
			} else if (k+2 >= 3 && k+2 < nz-3) {
				_ls[2][0] += comz1;
				_ls[2][1] -= comz4;
				_ls[2][2] += comz6;
				_ls[2][3] -= comz4;
				_ls[2][4] += comz1;
			} else if (k+2 == nz-3) {
				_ls[2][0] += comz1;
				_ls[2][1] -= comz4;
				_ls[2][2] += comz6;
				_ls[2][3] -= comz4;
			} else if (k+2 == nz-2) {
				_ls[2][0] += comz1;
				_ls[2][1] -= comz4;
				_ls[2][2] += comz5;
			}

			//---------------------------------------------------------------------
			//      store computed lhs for later reuse
			//---------------------------------------------------------------------
			for (m = 0; m < 5; m++) lhsp(m,i,j,k+2) = _ls[2][m];
			rhos[0] = rhos[1]; rhos[1] = rhos[2];
			cv[0] = cv[1]; 
      cv[1] = cv[2];
		}

		//---------------------------------------------------------------------
		//      load rhs values for current iteration
		//---------------------------------------------------------------------
		for (m = 0; m < 3; m++) _rs[2][m] = rhs(m,i,j,k+2);

		//---------------------------------------------------------------------
		//      perform current iteration
		//---------------------------------------------------------------------
		fac1 = 1.0/_ls[0][2];
		_ls[0][3] *= fac1;
		_ls[0][4] *= fac1;
		for (m = 0; m < 3; m++) _rs[0][m] *= fac1;
		_ls[1][2] -= _ls[1][1] * _ls[0][3];
		_ls[1][3] -= _ls[1][1] * _ls[0][4];
		for (m = 0; m < 3; m++) _rs[1][m] -= _ls[1][1] * _rs[0][m];
		_ls[2][1] -= _ls[2][0] * _ls[0][3];
		_ls[2][2] -= _ls[2][0] * _ls[0][4];
		for (m = 0; m < 3; m++) _rs[2][m] -= _ls[2][0] * _rs[0][m];

		//---------------------------------------------------------------------
		//      store computed lhs and prepare data for next iteration
		//	rhs is stored in a temp array such that write accesses are coalesced
		//---------------------------------------------------------------------
		lhs(3,i,j,k) = _ls[0][3];
		lhs(4,i,j,k) = _ls[0][4];
		for (m = 0; m < 5; m++) {
			_ls[0][m] = _ls[1][m];
			_ls[1][m] = _ls[2][m];
		}
		for (m = 0; m < 3; m++) {
			rtmp(m,i,j,k) = _rs[0][m];
			_rs[0][m] = _rs[1][m];
			_rs[1][m] = _rs[2][m];
		}
	}
	//---------------------------------------------------------------------
	//      The last two rows in this zone are a bit different, 
	//      since they do not have two more rows available for the
	//      elimination of off-diagonal entries
	//---------------------------------------------------------------------
	k = nz-2;
	fac1 = 1.0/_ls[0][2];
	_ls[0][3] *= fac1;
	_ls[0][4] *= fac1;
	for (m = 0; m < 3; m++) _rs[0][m] *= fac1;
	_ls[1][2] -= _ls[1][1] * _ls[0][3];
	_ls[1][3] -= _ls[1][1] * _ls[0][4];
	for (m = 0; m < 3; m++) _rs[1][m] -= _ls[1][1] * _rs[0][m];
	//---------------------------------------------------------------------
	//               scale the last row immediately
	//---------------------------------------------------------------------
	fac1 = 1.0/_ls[1][2];
	for (m = 0; m < 3; m++) _rs[1][m] *= fac1;
	lhs(3,i,j,k) = _ls[0][3];
	lhs(4,i,j,k) = _ls[0][4];

	//---------------------------------------------------------------------
	//      subsequently, fill the other factors u+c, u-c 
	//---------------------------------------------------------------------
	for (k = 0; k < 3; k++) cv[k] = speed(i,j,k);
	for (m = 0; m < 5; m++) {
		_ls[0][m] = lhsp(m,i,j,0);
		_lp[0][m] = lhsp(m,i,j,0);
		_ls[1][m] = lhsp(m,i,j,1);
		_lp[1][m] = lhsp(m,i,j,1);
	}
	_lp[1][1] -= dttz2*cv[0];
	_lp[1][3] += dttz2*cv[2];
	_ls[1][1] += dttz2*cv[0];
	_ls[1][3] -= dttz2*cv[2];
	cv[0] = cv[1]; 
  cv[1] = cv[2];
	_rs[0][3] = rhs(3,i,j,0);
	_rs[0][4] = rhs(4,i,j,0);
	_rs[1][3] = rhs(3,i,j,1);
	_rs[1][4] = rhs(4,i,j,1);
	//---------------------------------------------------------------------
	//      do the u+c and the u-c factors               
	//---------------------------------------------------------------------
	for (k = 0; k < nz-2; k++) {
		//---------------------------------------------------------------------
		//      first, fill the other factors u+c, u-c 
		//---------------------------------------------------------------------
		for (m = 0; m < 5; m++) {
			_ls[2][m] = lhsp(m,i,j,k+2);
			_lp[2][m] = lhsp(m,i,j,k+2);
		}
		_rs[2][3] = rhs(3,i,j,k+2);
		_rs[2][4] = rhs(4,i,j,k+2);
		if (k+2 < nz-1) {
			cv[2] = speed(i,j,k+3);
			_lp[2][1] -= dttz2*cv[0];
			_lp[2][3] += dttz2*cv[2];
			_ls[2][1] += dttz2*cv[0];
			_ls[2][3] -= dttz2*cv[2];
			cv[0] = cv[1]; 
      cv[1] = cv[2];
		}

		m = 3;
		fac1 = 1.0/_lp[0][2];
		_lp[0][3] *= fac1;
		_lp[0][4] *= fac1;
		_rs[0][m] *= fac1;
		_lp[1][2] -= _lp[1][1] * _lp[0][3];
		_lp[1][3] -= _lp[1][1] * _lp[0][4];
		_rs[1][m] -= _lp[1][1] * _rs[0][m];
		_lp[2][1] -= _lp[2][0] * _lp[0][3];
		_lp[2][2] -= _lp[2][0] * _lp[0][4];
		_rs[2][m] -= _lp[2][0] * _rs[0][m];

		m = 4;
		fac1 = 1.0/_ls[0][2];
		_ls[0][3] *= fac1;
		_ls[0][4] *= fac1;
		_rs[0][m] *= fac1;
		_ls[1][2] -= _ls[1][1] * _ls[0][3];
		_ls[1][3] -= _ls[1][1] * _ls[0][4];
		_rs[1][m] -= _ls[1][1] * _rs[0][m];
		_ls[2][1] -= _ls[2][0] * _ls[0][3];
		_ls[2][2] -= _ls[2][0] * _ls[0][4];
		_rs[2][m] -= _ls[2][0] * _rs[0][m];

		//---------------------------------------------------------------------
		//      store computed lhs and prepare data for next iteration
		//	rhs is stored in a temp array such that write accesses are coalesced
		//---------------------------------------------------------------------
		for (m = 3; m < 5; m++) {
			lhsp(m,i,j,k) = _lp[0][m];
			lhsm(m,i,j,k) = _ls[0][m];
			rtmp(m,i,j,k) = _rs[0][m];
			_rs[0][m] = _rs[1][m];
			_rs[1][m] = _rs[2][m];
		}
		for (m = 0; m < 5; m++) {
			_lp[0][m] = _lp[1][m];
			_lp[1][m] = _lp[2][m];
			_ls[0][m] = _ls[1][m];
			_ls[1][m] = _ls[2][m];
		}
	}
	//---------------------------------------------------------------------
	//         And again the last two rows separately
	//---------------------------------------------------------------------
	k = nz-2;
	m = 3;
	fac1 = 1.0/_lp[0][2];
	_lp[0][3] *= fac1;
	_lp[0][4] *= fac1;
	_rs[0][m] *= fac1;
	_lp[1][2] -= _lp[1][1] * _lp[0][3];
	_lp[1][3] -= _lp[1][1] * _lp[0][4];
	_rs[1][m] -= _lp[1][1] * _rs[0][m];

	m = 4;
	fac1 = 1.0/_ls[0][2];
	_ls[0][3] *= fac1;
	_ls[0][4] *= fac1;
	_rs[0][m] *= fac1;
	_ls[1][2] -= _ls[1][1] * _ls[0][3];
	_ls[1][3] -= _ls[1][1] * _ls[0][4];
	_rs[1][m] -= _ls[1][1] * _rs[0][m];
	//---------------------------------------------------------------------
	//               Scale the last row immediately some of this is overkill
	//               if this is the last cell
	//---------------------------------------------------------------------
	_rs[1][3] /= _lp[1][2];
	_rs[1][4] /= _ls[1][2];
		
	//---------------------------------------------------------------------
	//                         BACKSUBSTITUTION 
	//---------------------------------------------------------------------
	for (m = 0; m < 3; m++) _rs[0][m] -= lhs(3,i,j,nz-2) * _rs[1][m];
	_rs[0][3] -= _lp[0][3] * _rs[1][3];
	_rs[0][4] -= _ls[0][3] * _rs[1][4];
	for (m = 0; m < 5; m++) {
		_rs[2][m] = _rs[1][m];
		_rs[1][m] = _rs[0][m];
	}
	
	for (k = nz-3; k >= 0; k--) {
		//---------------------------------------------------------------------
		//      The first three factors
		//---------------------------------------------------------------------
		for (m = 0; m < 3; m++) _rs[0][m] = rtmp(m,i,j,k) - lhs(3,i,j,k)*_rs[1][m] - lhs(4,i,j,k)*_rs[2][m];
		//---------------------------------------------------------------------
		//      And the remaining two
		//---------------------------------------------------------------------
		_rs[0][3] = rtmp(3,i,j,k) - lhsp(3,i,j,k)*_rs[1][3] - lhsp(4,i,j,k)*_rs[2][3];
		_rs[0][4] = rtmp(4,i,j,k) - lhsm(3,i,j,k)*_rs[1][4] - lhsm(4,i,j,k)*_rs[2][4];

		if (k+2 < nz-1) {
			//---------------------------------------------------------------------
			//   block-diagonal matrix-vector multiplication tzetar
			//---------------------------------------------------------------------
			double xvel = us(i,j,k+2);
			double yvel = vs(i,j,k+2);
			double zvel = ws(i,j,k+2);
			double ac = speed(i,j,k+2);
			double uzik1 = fu(0,i,j,k+2);
			double t1 = (bt*uzik1)/ac * (_rs[2][3] + _rs[2][4]);
			double t2 = _rs[2][2] + t1;
			double t3 = bt*uzik1 * (_rs[2][3] - _rs[2][4]);

			_rs[2][4] =  uzik1*(-xvel*_rs[2][1] + yvel*_rs[2][0]) + qs(i,j,k+2)*t2 + c2iv*(ac*ac)*t1 + zvel*t3;
			_rs[2][3] =  zvel*t2  + t3;
			_rs[2][2] =  uzik1*_rs[2][0] + yvel*t2;
			_rs[2][1] = -uzik1*_rs[2][1] + xvel*t2;
			_rs[2][0] = t2;
		}

		for (m = 0; m < 5; m++) {
			rhs(m,i,j,k+2) = _rs[2][m];
			_rs[2][m] = _rs[1][m];
			_rs[1][m] = _rs[0][m];
		}
	}

	//---------------------------------------------------------------------
	//   block-diagonal matrix-vector multiplication tzetar
	//---------------------------------------------------------------------
	double xfvel = us(i,j,1);
	double yfvel = vs(i,j,1);
	double zfvel = ws(i,j,1);
	double afc = speed(i,j,1);
	double ufzik1 = fu(0,i,j,1);
	double tf1 = (bt*ufzik1)/afc * (_rs[2][3] + _rs[2][4]);
	double tf2 = _rs[2][2] + tf1;
	double tf3 = bt*ufzik1 * (_rs[2][3] - _rs[2][4]);

	rhs(4,i,j,1) =  ufzik1*(-xfvel*_rs[2][1] + yfvel*_rs[2][0]) + qs(i,j,1)*tf2 + c2iv*(afc*afc)*tf1 + zfvel*tf3;
	rhs(3,i,j,1) =  zfvel*tf2  + tf3;
	rhs(2,i,j,1) =  ufzik1*_rs[2][0] + yfvel*tf2;
	rhs(1,i,j,1) = -ufzik1*_rs[2][1] + xfvel*tf2;
	rhs(0,i,j,1) = tf2;

	for (m = 0; m < 5; m++) rhs(m,i,j,0) = _rs[1][m];
}
#undef lhs
#undef lhsp
#undef lhsm
#undef rtmp
//---------------------------------------------------------------------
// 	addition of update to the vector u
//---------------------------------------------------------------------
__global__ static void add_kernel (double *fu, double *rhs, const int nx, const int ny, const int nz) {
	int i, j, k, m;

	k = blockIdx.y+1;
	j = blockIdx.x+1;
	i = threadIdx.x+1;
	m = threadIdx.y;

	fu(m,i,j,k) += rhs(m,i,j,k);
}

//---------------------------------------------------------------------
// adi
//---------------------------------------------------------------------
void adi(bool singlestep, int nx, int ny, int nz, int niter, double* rho_i, double* us, double* vs, double* ws, 
         double* speed, double* qs, double* square, double* rhs, double* lhs, double* forcing, double* fu, double* rstmp) {

	HANDLE_ERROR(cudaDeviceSynchronize());

	int itmax = singlestep ? 1 : niter;
  int xblock, xgrid, yblock, ygrid, zblock, zgrid;
	for (int step = 1; step <= itmax; step++) {
		if (step % 20 == 0 || step == 1 && !singlestep)
			printf(" Time step %4d\n", step);

		//compute_rhs();
	  dim3 grid1(ny,nz);
	  compute_rhs_kernel_1<<<grid1,nx>>>(rho_i, us, vs, ws, speed, qs, square, fu, nx, ny, nz);

	  compute_rhs_kernel_2<<<grid1,nx>>>(rho_i, us, vs, ws, qs, square, rhs, forcing, fu, nx, ny, nz);

		//txinvr();
	  dim3 grid2(ny-2,nz-2);
	  txinvr_kernel<<<grid2,nx-2>>> (rho_i, us, vs, ws, speed, qs, rhs, nx, ny, nz);

		//x_solve();
	  yblock = min(SOLVE_BLOCK,ny);
	  ygrid = (ny+yblock-1)/yblock;
	  zblock = min(SOLVE_BLOCK/yblock,nz);
	  zgrid = (nz+zblock-1)/zblock;
	  dim3 grid3(zgrid,ygrid), block3(zblock,yblock);
	  x_solve_kernel<<<grid3,block3>>>(rho_i, us, speed, rhs, lhs, rstmp, nx, ny, nz);

		//y_solve();
	  xblock = min(SOLVE_BLOCK,nx);
	  xgrid = (nx+xblock-1)/xblock;
	  zblock = min(SOLVE_BLOCK/xblock,nz);
	  zgrid = (nz+zblock-1)/zblock;
	  dim3 grid4(zgrid,xgrid), block4(zblock,xblock);
	  y_solve_kernel<<<grid4,block4>>>(rho_i, vs, speed, rhs, lhs, rstmp, nx, ny, nz);

		//z_solve();
	  xblock = min(SOLVE_BLOCK,nx);
	  xgrid = (nx+xblock-1)/xblock;
	  yblock = min(SOLVE_BLOCK/xblock,ny);
	  ygrid = (ny+yblock-1)/yblock;
	  dim3 grid5(ygrid,xgrid), block5(yblock,xblock);
	  z_solve_kernel<<<grid5,block5>>>(rho_i, us, vs, ws, speed, qs, fu, rhs, lhs, rstmp, nx, ny, nz);

		//add();
	  dim3 grid6(ny-2,nz-2);
	  dim3 block6(nx-2,5);
	  add_kernel<<<grid6,block6>>>(fu, rhs, nx, ny, nz);
	}

	HANDLE_ERROR(cudaDeviceSynchronize());
}
//---------------------------------------------------------------------
//      defaults from parameters
//---------------------------------------------------------------------
void read_input(char benchclass, double* dd_td, int* nx, int* ny, int* nz, int* niter) {
	FILE *file;

	if ((file = fopen("inputsp.data", "rt")) != NULL) {
		char line[1024];
		printf(" Reading from input file inputsp.data\n");
		
		fgets(line, sizeof(line)-1, file);
		sscanf(line, "%i", niter);
		fgets(line, sizeof(line)-1, file);
		sscanf(line, "%lf", dd_td);
		fgets(line, sizeof(line)-1, file);
		sscanf(line, "%i %i %i", nx, ny, nz);
		fclose(file);
	} else {
//		printf(" No input file inputsp.data. Using compiled defaults\n");
		int problem_size;
		switch (benchclass) {
			case 's':
			case 'S': problem_size = 12; *dd_td = 0.015; *niter = 100; break;
			case 'w':
			case 'W': problem_size = 36; *dd_td = 0.0015; *niter = 400; break;
			case 'a':
			case 'A': problem_size = 64; *dd_td = 0.0015; *niter = 400; break;
			case 'b':
			case 'B': problem_size = 102; *dd_td = 0.001; *niter = 400; break;
			case 'c':
			case 'C': problem_size = 162; *dd_td = 0.00067; *niter = 400; break;
			case 'd':
			case 'D': problem_size = 408; *dd_td = 0.00030; *niter = 500; break;
			case 'e':
			case 'E': problem_size = 1020; *dd_td = 0.0001; *niter = 500; break;
			default: printf("setparams: Internal error: invalid class %c\n", benchclass); exit(EXIT_FAILURE);
		}
		*nx = *ny = *nz = problem_size;
	}

	printf("\n\n NAS Parallel Benchmarks (NPB3.3-CUDA) - SP Benchmark\n\n");
	printf(" Size: %4dx%4dx%4d\n", *nx, *ny, *nz);
	printf(" Iterations: %4d    dt_d: %10.6F\n", *niter, *dd_td);
	printf("\n");
}




int main(int argc, char **argv) {
	char benchclass = argc > 1 ? argv[1][0] : 'S';

  struct timeval start_t;
  struct timeval end_t;
  struct timeval skt_t;
  struct timeval ske_t;

	int niter;
	int nx, ny, nz;
  double hdd;
  double dd_d;
	
	double *fu, *forcing, *rhs, *rho_i, *us, *vs, *ws, *qs, *speed, *square, *lhs, *rstmp;
  //double* rmsbuf;
	//double xce[5], xcr[5];

	char CUDAname[256];
	int CUDAmp, CUDAclock, CUDAmemclock, CUDAl2cache;
	size_t CUDAmem;

	//---------------------------------------------------------------------
	//   read input data
	//---------------------------------------------------------------------
	read_input(benchclass, &hdd, &nx, &ny, &nz, &niter);
	dd_d = hdd;

	//---------------------------------------------------------------------
	//   allocate CUDA device memory
	//---------------------------------------------------------------------
	int gridsize = nx*ny*nz;
	int facesize = max(max(nx*ny, nx*nz), ny*nz);

  gettimeofday(&start_t, NULL);

	HANDLE_ERROR(cudaMalloc((void **)&fu, 5*gridsize*sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void **)&forcing, 5*gridsize*sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void **)&rhs, 5*gridsize*sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void **)&rho_i, gridsize*sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void **)&us, gridsize*sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void **)&vs, gridsize*sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void **)&ws, gridsize*sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void **)&qs, gridsize*sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void **)&speed, gridsize*sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void **)&square, gridsize*sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void **)&lhs, 9*gridsize*sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void **)&rstmp, 5*gridsize*sizeof(double)));
	//HANDLE_ERROR(cudaMalloc((void **)&rmsbuf, 5*facesize*sizeof(double)));

	  double ce_d[13][5];
	  ce_d[0][0] = (double)2.0;
	  ce_d[1][0] = (double)0.0;
	  ce_d[2][0] = (double)0.0;
	  ce_d[3][0] = (double)4.0;
	  ce_d[4][0] = (double)5.0;
	  ce_d[5][0] = (double)3.0;
	  ce_d[6][0] = (double)0.5;
	  ce_d[7][0] = (double)0.02;
	  ce_d[8][0] = (double)0.01;
	  ce_d[9][0] = (double)0.03;
	  ce_d[10][0] = (double)0.5;
	  ce_d[11][0] = (double)0.4;
	  ce_d[12][0] = (double)0.3;

	  ce_d[0][1] = (double)1.0;
	  ce_d[1][1] = (double)0.0;
	  ce_d[2][1] = (double)0.0;
	  ce_d[3][1] = (double)0.0;
	  ce_d[4][1] = (double)1.0;
	  ce_d[5][1] = (double)2.0;
	  ce_d[6][1] = (double)3.0;
	  ce_d[7][1] = (double)0.01;
	  ce_d[8][1] = (double)0.03;
	  ce_d[9][1] = (double)0.02;
	  ce_d[10][1] = (double)0.4;
	  ce_d[11][1] = (double)0.3;
	  ce_d[12][1] = (double)0.5;

	  ce_d[0][2] = (double)2.0;
	  ce_d[1][2] = (double)2.0;
	  ce_d[2][2] = (double)0.0;
	  ce_d[3][2] = (double)0.0;
	  ce_d[4][2] = (double)0.0;
	  ce_d[5][2] = (double)2.0;
	  ce_d[6][2] = (double)3.0;
	  ce_d[7][2] = (double)0.04;
	  ce_d[8][2] = (double)0.03;
	  ce_d[9][2] = (double)0.05;
	  ce_d[10][2] = (double)0.3;
	  ce_d[11][2] = (double)0.5;
	  ce_d[12][2] = (double)0.4;

	  ce_d[0][3] = (double)2.0;
	  ce_d[1][3] = (double)2.0;
	  ce_d[2][3] = (double)0.0;
	  ce_d[3][3] = (double)0.0;
	  ce_d[4][3] = (double)0.0;
	  ce_d[5][3] = (double)2.0;
	  ce_d[6][3] = (double)3.0;
	  ce_d[7][3] = (double)0.03;
	  ce_d[8][3] = (double)0.05;
	  ce_d[9][3] = (double)0.04;
	  ce_d[10][3] = (double)0.2;
	  ce_d[11][3] = (double)0.1;
	  ce_d[12][3] = (double)0.3;

	  ce_d[0][4] = (double)5.0;
	  ce_d[1][4] = (double)4.0;
	  ce_d[2][4] = (double)3.0;
	  ce_d[3][4] = (double)2.0;
	  ce_d[4][4] = (double)0.1;
	  ce_d[5][4] = (double)0.4;
	  ce_d[6][4] = (double)0.3;
	  ce_d[7][4] = (double)0.05;
	  ce_d[8][4] = (double)0.04;
	  ce_d[9][4] = (double)0.03;
	  ce_d[10][4] = (double)0.1;
	  ce_d[11][4] = (double)0.3;
	  ce_d[12][4] = (double)0.2;

	  double bt_d = sqrt(0.5);

	  double dnxm1_d = 1.0/((double)nx-1.0);
	  double dnym1_d = 1.0/((double)ny-1.0);
	  double dnzm1_d = 1.0/((double)nz-1.0);

	  double tx1_d = 1.0 / (dnxm1_d * dnxm1_d);
	  double tx2_d = 1.0 / (2.0 * dnxm1_d);
	  double tx3_d = 1.0 / dnxm1_d;

	  double ty1_d = 1.0 / (dnym1_d * dnym1_d);
	  double ty2_d = 1.0 / (2.0 * dnym1_d);
	  double ty3_d = 1.0 / dnym1_d;
 
	  double tz1_d = 1.0 / (dnzm1_d * dnzm1_d);
	  double tz2_d = 1.0 / (2.0 * dnzm1_d);
	  double tz3_d = 1.0 / dnzm1_d;

	  double dtx1_d = dd_d*tx1_d;
	  double dttx2_d = dd_d*tx2_d;
	  double dty1_d = dd_d*ty1_d;
	  double dtty2_d = dd_d*ty2_d;
	  double dtz1_d = dd_d*tz1_d;
	  double dttz2_d = dd_d*tz2_d;

	  double c2dttx1_d = 2.0*dtx1_d;
	  double c2dtty1_d = 2.0*dty1_d;
	  double c2dttz1_d = 2.0*dtz1_d;

	  double dtdssp_d = dd_d*dssp;

	  double comz1_d  = dtdssp_d;
	  double comz4_d  = 4.0*dtdssp_d;
	  double comz5_d  = 5.0*dtdssp_d;
	  double comz6_d  = 6.0*dtdssp_d;

	  double c3c4tx3_d = c3c4*tx3_d;
	  double c3c4ty3_d = c3c4*ty3_d;
	  double c3c4tz3_d = c3c4*tz3_d;

	  double dx1tx1_d = dx1*tx1_d;
	  double dx2tx1_d = dx2*tx1_d;
	  double dx3tx1_d = dx3*tx1_d;
	  double dx4tx1_d = dx4*tx1_d;
	  double dx5tx1_d = dx5*tx1_d;

	  double dy1ty1_d = dy1*ty1_d;
	  double dy2ty1_d = dy2*ty1_d;
	  double dy3ty1_d = dy3*ty1_d;
	  double dy4ty1_d = dy4*ty1_d;
	  double dy5ty1_d = dy5*ty1_d;

	  double dz1tz1_d = dz1*tz1_d;
	  double dz2tz1_d = dz2*tz1_d;
	  double dz3tz1_d = dz3*tz1_d;
	  double dz4tz1_d = dz4*tz1_d;
	  double dz5tz1_d = dz5*tz1_d;

	  double xxcon1_d = c3c4tx3_d*con43*tx3_d;
	  double xxcon2_d = c3c4tx3_d*tx3_d;
	  double xxcon3_d = c3c4tx3_d*conz1*tx3_d;
	  double xxcon4_d = c3c4tx3_d*con16*tx3_d;
	  double xxcon5_d = c3c4tx3_d*c1c5*tx3_d;

	  double yycon1_d = c3c4ty3_d*con43*ty3_d;
	  double yycon2_d = c3c4ty3_d*ty3_d;
	  double yycon3_d = c3c4ty3_d*conz1*ty3_d;
	  double yycon4_d = c3c4ty3_d*con16*ty3_d;
	  double yycon5_d = c3c4ty3_d*c1c5*ty3_d;

	  double zzcon1_d = c3c4tz3_d*con43*tz3_d;
	  double zzcon2_d = c3c4tz3_d*tz3_d;
	  double zzcon3_d = c3c4tz3_d*conz1*tz3_d;
	  double zzcon4_d = c3c4tz3_d*con16*tz3_d;
	  double zzcon5_d = c3c4tz3_d*c1c5*tz3_d;

	  HANDLE_ERROR (cudaMemcpyToSymbol (&ce, &ce_d, 13*5*sizeof(double)));
	  HANDLE_ERROR (cudaMemcpyToSymbol (&bt, &bt_d, sizeof(double)));
	  HANDLE_ERROR (cudaMemcpyToSymbol (&dnxm1, &dnxm1_d, sizeof(double)));
	  HANDLE_ERROR (cudaMemcpyToSymbol (&dnym1, &dnym1_d, sizeof(double)));
	  HANDLE_ERROR (cudaMemcpyToSymbol (&dnzm1, &dnzm1_d, sizeof(double)));
	  HANDLE_ERROR (cudaMemcpyToSymbol (&tx1, &tx1_d, sizeof(double)));
	  HANDLE_ERROR (cudaMemcpyToSymbol (&tx2, &tx2_d, sizeof(double)));
	  HANDLE_ERROR (cudaMemcpyToSymbol (&tx3, &tx3_d, sizeof(double)));
	  HANDLE_ERROR (cudaMemcpyToSymbol (&ty1, &ty1_d, sizeof(double)));
	  HANDLE_ERROR (cudaMemcpyToSymbol (&ty2, &ty2_d, sizeof(double)));
	  HANDLE_ERROR (cudaMemcpyToSymbol (&ty3, &ty3_d, sizeof(double)));
	  HANDLE_ERROR (cudaMemcpyToSymbol (&tz1, &tz1_d, sizeof(double)));
	  HANDLE_ERROR (cudaMemcpyToSymbol (&tz2, &tz2_d, sizeof(double)));
	  HANDLE_ERROR (cudaMemcpyToSymbol (&tz3, &tz3_d, sizeof(double)));
	  HANDLE_ERROR (cudaMemcpyToSymbol (&dtx1, &dtx1_d, sizeof(double)));
	  HANDLE_ERROR (cudaMemcpyToSymbol (&dttx2, &dttx2_d, sizeof(double)));
	  HANDLE_ERROR (cudaMemcpyToSymbol (&dty1, &dty1_d, sizeof(double)));
	  HANDLE_ERROR (cudaMemcpyToSymbol (&dtty2, &dtty2_d, sizeof(double)));
	  HANDLE_ERROR (cudaMemcpyToSymbol (&dtz1, &dtz1_d, sizeof(double)));
	  HANDLE_ERROR (cudaMemcpyToSymbol (&dttz2, &dttz2_d, sizeof(double)));
	  HANDLE_ERROR (cudaMemcpyToSymbol (&c2dttx1, &c2dttx1_d, sizeof(double)));
	  HANDLE_ERROR (cudaMemcpyToSymbol (&c2dtty1, &c2dtty1_d, sizeof(double)));
	  HANDLE_ERROR (cudaMemcpyToSymbol (&c2dttz1, &c2dttz1_d, sizeof(double)));
	  HANDLE_ERROR (cudaMemcpyToSymbol (&dt, &dd_d, sizeof(double)));
	  HANDLE_ERROR (cudaMemcpyToSymbol (&dtdssp, &dtdssp_d, sizeof(double)));
	  HANDLE_ERROR (cudaMemcpyToSymbol (&comz1, &comz1_d, sizeof(double)));
	  HANDLE_ERROR (cudaMemcpyToSymbol (&comz4, &comz4_d, sizeof(double)));
	  HANDLE_ERROR (cudaMemcpyToSymbol (&comz5, &comz5_d, sizeof(double)));
	  HANDLE_ERROR (cudaMemcpyToSymbol (&comz6, &comz6_d, sizeof(double)));
	  HANDLE_ERROR (cudaMemcpyToSymbol (&c3c4tx3, &c3c4tx3_d, sizeof(double)));
	  HANDLE_ERROR (cudaMemcpyToSymbol (&c3c4ty3, &c3c4ty3_d, sizeof(double)));
	  HANDLE_ERROR (cudaMemcpyToSymbol (&c3c4tz3, &c3c4tz3_d, sizeof(double)));
	  HANDLE_ERROR (cudaMemcpyToSymbol (&dx1tx1, &dx1tx1_d, sizeof(double)));
	  HANDLE_ERROR (cudaMemcpyToSymbol (&dx2tx1, &dx2tx1_d, sizeof(double)));
	  HANDLE_ERROR (cudaMemcpyToSymbol (&dx3tx1, &dx3tx1_d, sizeof(double)));
	  HANDLE_ERROR (cudaMemcpyToSymbol (&dx4tx1, &dx4tx1_d, sizeof(double)));
	  HANDLE_ERROR (cudaMemcpyToSymbol (&dx5tx1, &dx5tx1_d, sizeof(double)));
	  HANDLE_ERROR (cudaMemcpyToSymbol (&dy1ty1, &dy1ty1_d, sizeof(double)));
	  HANDLE_ERROR (cudaMemcpyToSymbol (&dy2ty1, &dy2ty1_d, sizeof(double)));
	  HANDLE_ERROR (cudaMemcpyToSymbol (&dy3ty1, &dy3ty1_d, sizeof(double)));
	  HANDLE_ERROR (cudaMemcpyToSymbol (&dy4ty1, &dy4ty1_d, sizeof(double)));
	  HANDLE_ERROR (cudaMemcpyToSymbol (&dy5ty1, &dy5ty1_d, sizeof(double)));
	  HANDLE_ERROR (cudaMemcpyToSymbol (&dz1tz1, &dz1tz1_d, sizeof(double)));
	  HANDLE_ERROR (cudaMemcpyToSymbol (&dz2tz1, &dz2tz1_d, sizeof(double)));
	  HANDLE_ERROR (cudaMemcpyToSymbol (&dz3tz1, &dz3tz1_d, sizeof(double)));
	  HANDLE_ERROR (cudaMemcpyToSymbol (&dz4tz1, &dz4tz1_d, sizeof(double)));
	  HANDLE_ERROR (cudaMemcpyToSymbol (&dz5tz1, &dz5tz1_d, sizeof(double)));
	  HANDLE_ERROR (cudaMemcpyToSymbol (&xxcon1, &xxcon1_d, sizeof(double)));
	  HANDLE_ERROR (cudaMemcpyToSymbol (&xxcon2, &xxcon2_d, sizeof(double)));
	  HANDLE_ERROR (cudaMemcpyToSymbol (&xxcon3, &xxcon3_d, sizeof(double)));
	  HANDLE_ERROR (cudaMemcpyToSymbol (&xxcon4, &xxcon4_d, sizeof(double)));
	  HANDLE_ERROR (cudaMemcpyToSymbol (&xxcon5, &xxcon5_d, sizeof(double)));
	  HANDLE_ERROR (cudaMemcpyToSymbol (&yycon1, &yycon1_d, sizeof(double)));
	  HANDLE_ERROR (cudaMemcpyToSymbol (&yycon2, &yycon2_d, sizeof(double)));
	  HANDLE_ERROR (cudaMemcpyToSymbol (&yycon3, &yycon3_d, sizeof(double)));
	  HANDLE_ERROR (cudaMemcpyToSymbol (&yycon4, &yycon4_d, sizeof(double)));
	  HANDLE_ERROR (cudaMemcpyToSymbol (&yycon5, &yycon5_d, sizeof(double)));
	  HANDLE_ERROR (cudaMemcpyToSymbol (&zzcon1, &zzcon1_d, sizeof(double)));
	  HANDLE_ERROR (cudaMemcpyToSymbol (&zzcon2, &zzcon2_d, sizeof(double)));
	  HANDLE_ERROR (cudaMemcpyToSymbol (&zzcon3, &zzcon3_d, sizeof(double)));
	  HANDLE_ERROR (cudaMemcpyToSymbol (&zzcon4, &zzcon4_d, sizeof(double)));
	  HANDLE_ERROR (cudaMemcpyToSymbol (&zzcon5, &zzcon5_d, sizeof(double)));

  gettimeofday(&skt_t, NULL);
	exact_rhs(forcing, nx, ny, nz);

	//sp->initialize();
  dim3 grid(nz,ny);
	initialize_kernel<<<grid,nx>>> (fu, nx, ny, nz);

	//---------------------------------------------------------------------
	//      do one time step to touch all code, and reinitialize
	//---------------------------------------------------------------------
	adi(true,  nx,  ny,  nz,  niter,  rho_i,  us,  vs,  ws, 
       speed,  qs,  square,  rhs,  lhs,  forcing,  fu,  rstmp);
	//sp->initialize();
	initialize_kernel<<<grid,nx>>> (fu, nx, ny, nz);

	//---------------------------------------------------------------------
	//   main time stepping loop
	//---------------------------------------------------------------------
	//sp->adi(false);
	adi(false,  nx,  ny,  nz,  niter,  rho_i,  us,  vs,  ws, 
       speed,  qs,  square,  rhs,  lhs,  forcing,  fu,  rstmp);

  gettimeofday(&ske_t, NULL);
  gettimeofday(&end_t, NULL);


	std::cout  << "time: "<<((end_t.tv_sec-start_t.tv_sec)+(end_t.tv_usec-start_t.tv_usec)*1e-6) << std::endl;
	std::cout  << "kernel: "<<((ske_t.tv_sec-skt_t.tv_sec)+(ske_t.tv_usec-skt_t.tv_usec)*1e-6) << std::endl;
	//std::cout  << (sdkGetAverageTimerValue(&timer)/1000.0)  / iterations << " seconds per iteration" << std::endl;
	//---------------------------------------------------------------------
	//   verification test
	//---------------------------------------------------------------------
	//char verifyclass;
  //bool verified;

	//---------------------------------------------------------------------
	//      More timers
	//---------------------------------------------------------------------
	//sp->print_timers();

	//delete sp;
	return EXIT_SUCCESS;
}

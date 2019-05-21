#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include "GPU_kernel.h"
#include <cuda_runtime.h>
#define M 10
#define eps 2.22e-17

/*__global__ float absMaxval_d(float *a, int nx, int ny, int nz)
{
					 
}
*/
__global__ void grad(float *u2, float *ux, float *uy, float *uz, float *d_c,
										 int nx, int ny, int nz, float dx, float dy, float dz)
{
	int idx = blockDim.y * blockIdx.y + threadIdx.y;
	int idy = blockDim.x * blockIdx.x + threadIdx.x;
	int idz = blockDim.z * blockIdx.z + threadIdx.z;
	if (idx >= M / 2 && idx < nx - M / 2 && idy >= M / 2 && idy < ny - M / 2 && idz >= M / 2 && idz < nz - M / 2)
	{
		int m, N = M / 2;
		float Ux = 0.0, Uy = 0.0, Uz = 0.0;
		for (m = 1; m < N + 1; m++)
		{
			Ux = Ux + d_c[N * (M / 2 + 1) + m] * (u2[idz * ny * nx + (idx + m) * ny + idy] - u2[idz * ny * nx + (idx - m) * ny + idy]) / dx;
			Uy = Uy + d_c[N * (M / 2 + 1) + m] * (u2[idz * ny * nx + idx * ny + idy + m] - u2[idz * ny * nx + idx * ny + idy - m]) / dy;
			Uz = Uz + d_c[N * (M / 2 + 1) + m] * (u2[(idz + m) * ny * nx + idx * ny + idy] - u2[(idz - m) * ny * nx + idx * ny + idy]) / dz;
		}
		ux[idz * ny * nx + idx * ny + idy] = Ux;
		uy[idz * ny * nx + idx * ny + idy] = Uy;
		uz[idz * ny * nx + idx * ny + idy] = Uz;
	}
}

__global__ void grad1(float *R, float *u2, float *ux, float *uy, float *uz, float *d_c, int nx, int ny, int nz, float dx, float dy, float dz)
{
	int idx = blockDim.y * blockIdx.y + threadIdx.y;
	int idy = blockDim.x * blockIdx.x + threadIdx.x;
	int idz = blockDim.z * blockIdx.z + threadIdx.z;
	// if(idx>=M/2 && idx<nx-M/2 && idy>=M/2 && idy<ny-M/2 && idz>=M/2 && idz<nz)
	if (idx >= M / 2 && idx < nx - M / 2 && idy >= M / 2 && idy < ny - M / 2 && idz >= M / 2 && idz < nz - M / 2)
	{
		int m;
		float Ux = 0.0, Uy = 0.0, Uz = 0.0;
		for (m = 1; m < M / 2 + 1; m++)
		{
			Ux = Ux + d_c[M / 2 * (M / 2 + 1) + m] * (u2[idz * ny * nx + (idx + m) * ny + idy] - u2[idz * ny * nx + (idx - m) * ny + idy]) / dx;
			Uy = Uy + d_c[M / 2 * (M / 2 + 1) + m] * (u2[idz * ny * nx + idx * ny + idy + m] - u2[idz * ny * nx + idx * ny + idy - m]) / dy;
			Uz = Uz + d_c[M / 2 * (M / 2 + 1) + m] * (u2[(idz + m) * ny * nx + idx * ny + idy] - u2[(idz - m) * ny * nx + idx * ny + idy]) / dz;

			/*	 if(idz<nz-M/2)
							Uz = Uz + d_c[M/2*(M/2+1)+m]*(u2[(idz+m)*ny*nx+idx*ny+idy] - u2[(idz-m)*ny*nx+idx*ny+idy])/dz;
					 else	
							Uz = Uz + d_c[M/2*(M/2+1)+m]*(R[(M/2-m)*ny*nx+idx*ny+idy] - u2[(idz-m)*ny*nx+idx*ny+idy])/dz;*/
		}
		ux[idz * ny * nx + idx * ny + idy] = Ux;
		uy[idz * ny * nx + idx * ny + idy] = Uy;
		uz[idz * ny * nx + idx * ny + idy] = Uz;
	}
}

/*__global__ void grad2(float *R, float *u2, float *ux, float *uy, float *uz, float *d_c, int nx, int ny, int nz, float dx, float dy, float dz)
{
		int idx =  blockDim.y * blockIdx.y + threadIdx.y;
		int idy =  blockDim.x * blockIdx.x + threadIdx.x;
		int idz =  blockDim.z * blockIdx.z + threadIdx.z;
	  if(idx>=5 && idx<nx-5 && idy>=5 && idy<ny-5 && idz<nz-5)
	  {
				int m, N = 5;
				float Ux = 0.0, Uy = 0.0, Uz = 0.0;
	  		for(m=1;m<N+1;m++)
       {   
    			 Ux = Ux + d_c[N*(M/2+1)+m]*(u2[idz*ny*nx+(idx+m)*ny+idy] - u2[idz*ny*nx+(idx-m)*ny+idy])/dx;
					 Uy = Uy + d_c[N*(M/2+1)+m]*(u2[idz*ny*nx+idx*ny+idy+m] - u2[idz*ny*nx+idx*ny+idy-m])/dy;
					 
					 if(idz>=5)
								Uz = Uz + d_c[N*(M/2+1)+m]*(u2[(idz+m)*ny*nx+idx*ny+idy] - u2[(idz-m)*ny*nx+idx*ny+idy])/dz;
					 else
								Uz = Uz + d_c[N*(M/2+1)+m]*(u2[(idz+m)*ny*nx+idx*ny+idy] - R[(5-m)*ny*nx+idx*ny+idy])/dz;
       }
       ux[idz*ny*nx+idx*ny+idy] = Ux;
		 	 uy[idz*ny*nx+idx*ny+idy] = Uy;	
		 	 uz[idz*ny*nx+idx*ny+idy] = Uz;	
	  }  
}
*/
__global__ void scalar_operator(float uxyzMax, float *ux, float *uy, float *uz, float *d_epsilon, float *d_delta, float *S, int nx, int ny, int nz)
{
	int idx = blockDim.y * blockIdx.y + threadIdx.y;
	int idy = blockDim.x * blockIdx.x + threadIdx.x;
	int idz = blockDim.z * blockIdx.z + threadIdx.z;
	float nxvector, nyvector, nzvector, mode;
	if (idx < nx && idy < ny && idz < nz)
		S[idz * ny * nx + idx * ny + idy] = 1.0;
	if (idx >= 5 && idx < nx - 5 && idy >= 5 && idy < ny - 5 && idz >= 5 && idz < nz - 5)
	{
		mode = sqrt(ux[idz * ny * nx + idx * ny + idy] * ux[idz * ny * nx + idx * ny + idy] + uy[idz * ny * nx + idx * ny + idy] * uy[idz * ny * nx + idx * ny + idy] +
								uz[idz * ny * nx + idx * ny + idy] * uz[idz * ny * nx + idx * ny + idy]);
		if (mode > 1.0e-02 * uxyzMax)
		{
			nxvector = ux[idz * ny * nx + idx * ny + idy] / (mode + eps);
			nyvector = uy[idz * ny * nx + idx * ny + idy] / (mode + eps);
			nzvector = uz[idz * ny * nx + idx * ny + idy] / (mode + eps);
		}
		else
		{
			nxvector = 0.0;
			nyvector = 0.0;
			nzvector = 0.0;
		}
		S[idz * ny * nx + idx * ny + idy] = 0.5 * (1 + sqrt(1 - (8 * (d_epsilon[idz * ny * nx + idx * ny + idy] - d_delta[idz * ny * nx + idx * ny + idy]) * (nxvector * nxvector + nyvector * nyvector) * nzvector * nzvector) / pow((1 + 2 * d_epsilon[idz * ny * nx + idx * ny + idy] * (nxvector * nxvector + nyvector * nyvector)), 2)));
		//S[idz*ny*nx+idx*ny+idy]= 1.0;
	}
}

__global__ void scalar_operator1(float uxyzMax, float *ux, float *uy, float *uz, float *d_epsilon, float *d_delta, float *S, int nx, int ny, int nz)
{
	int idx = blockDim.y * blockIdx.y + threadIdx.y;
	int idy = blockDim.x * blockIdx.x + threadIdx.x;
	int idz = blockDim.z * blockIdx.z + threadIdx.z;
	float nxvector, nyvector, nzvector, mode;
	if (idx < nx && idy < ny && idz < nz)
		S[idz * ny * nx + idx * ny + idy] = 1.0;

	//  if(idx>=M/2 && idx<nx-M/2 && idy>=M/2 && idy<ny-M/2 && idz>=M/2 && idz<nz)
	if (idx >= M / 2 && idx < nx - M / 2 && idy >= M / 2 && idy < ny - M / 2 && idz >= M / 2 && idz < nz - M / 2)
	{
		mode = sqrtf(ux[idz * ny * nx + idx * ny + idy] * ux[idz * ny * nx + idx * ny + idy] + uy[idz * ny * nx + idx * ny + idy] * uy[idz * ny * nx + idx * ny + idy] +
								 uz[idz * ny * nx + idx * ny + idy] * uz[idz * ny * nx + idx * ny + idy]);
		if (mode > 1.0e-02 * uxyzMax)
		{
			nxvector = ux[idz * ny * nx + idx * ny + idy] / (mode + eps);
			nyvector = uy[idz * ny * nx + idx * ny + idy] / (mode + eps);
			nzvector = uz[idz * ny * nx + idx * ny + idy] / (mode + eps);
		}
		else
		{
			nxvector = 0.0;
			nyvector = 0.0;
			nzvector = 0.0;
		}
		S[idz * ny * nx + idx * ny + idy] = 0.5 * (1 + sqrtf(1 - (8 * (d_epsilon[idz * ny * nx + idx * ny + idy] - d_delta[idz * ny * nx + idx * ny + idy]) * (nxvector * nxvector + nyvector * nyvector) * nzvector * nzvector) / pow((1 + 2 * d_epsilon[idz * ny * nx + idx * ny + idy] * (nxvector * nxvector + nyvector * nyvector)), 2)));
	}
}

/*
__global__ void u_update_middle1(float *d_c2, float *d_epsilon,float *d_delta,float *d_vp, float dx, float dy, float dz,  float dt,
                                int nx, int ny, int nz, int pml, float *R, float *u1,  float *u2,float *u3, float *S)
{
		int idx =  blockDim.y * blockIdx.y + threadIdx.y;
		int idy =  blockDim.x * blockIdx.x + threadIdx.x;
		int idz =  blockDim.z * blockIdx.z + threadIdx.z;
		
		if(idx>=pml && idx<nx-pml && idy>=pml && idy<ny-pml && idz>=pml && idz<nz)
		//if(idx>=pml && idx<nx-pml && idy>=pml && idy<ny-pml && idz>=pml && idz<nz-pml)
	 {
				int m, i;
				float Uxx = 0.0, Uyy = 0.0, Uzz = 0.0;
				i = idz*ny*nx+idx*ny+idy;
				for(m=1;m<M/2+1;m++)
				{
					 Uxx = Uxx + d_c2[(M/2)*(M/2+1)+m]*(u2[idz*ny*nx+(idx+m)*ny+idy] - 2*u2[idz*ny*nx+idx*ny+idy] + u2[idz*ny*nx+(idx-m)*ny+idy])/(dx*dx);
					 Uyy = Uyy + d_c2[(M/2)*(M/2+1)+m]*(u2[idz*ny*nx+idx*ny+idy+m] - 2*u2[idz*ny*nx+idx*ny+idy] + u2[idz*ny*nx+idx*ny+(idy-m)])/(dy*dy);	
					// Uzz = Uzz + d_c2[(M/2)*(M/2+1)+m]*(u2[(idz+m)*ny*nx+idx*ny+idy] - 2*u2[idz*ny*nx+idx*ny+idy] + u2[(idz-m)*ny*nx+idx*ny+idy])/(dz*dz);
					 
					 if(idz<nz-M/2)
							Uzz = Uzz + d_c2[(M/2)*(M/2+1)+m]*(u2[(idz+m)*ny*nx+idx*ny+idy] - 2*u2[idz*ny*nx+idx*ny+idy] + u2[(idz-m)*ny*nx+idx*ny+idy])/(dz*dz);	
					 else
							Uzz = Uzz + d_c2[(M/2)*(M/2+1)+m]*(R[(5-m)*nx*ny+idx*ny+idy] - 2*u2[idz*ny*nx+idx*ny+idy] + u2[(idz-m)*ny*nx+idx*ny+idy])/(dz*dz);
				}						 				
				u3[i] = 2*u2[i]-u1[i]+ dt*dt*((1+2*d_epsilon[i])*(Uxx+Uyy)+Uzz)*S[i]*d_vp[i]*d_vp[i];
	 }
	
	
}

	
__global__ void u_update_middle2(float *d_c2, float *d_epsilon,float *d_delta,float *d_vp, float dx, float dy, float dz,  float dt,
                                int nx, int ny, int nz, int pml, float *R, float *u1,  float *u2,float *u3, float *S)
{
		int idx =  blockDim.y * blockIdx.y + threadIdx.y;
		int idy =  blockDim.x * blockIdx.x + threadIdx.x;
		int idz =  blockDim.z * blockIdx.z + threadIdx.z;
		
		if(idx>=pml && idx<nx-pml && idy>=pml && idy<ny-pml && idz<nz-pml)
	 {
				int m, i;
				float Uxx = 0.0, Uyy = 0.0, Uzz = 0.0;
				i = idz*ny*nx+idx*ny+idy;
				for(m=1;m<M/2+1;m++)
				{
					 Uxx = Uxx + d_c2[(M/2)*(M/2+1)+m]*(u2[idz*ny*nx+(idx+m)*ny+idy] - 2*u2[idz*ny*nx+idx*ny+idy] + u2[idz*ny*nx+(idx-m)*ny+idy])/(dx*dx);
					 Uyy = Uyy + d_c2[(M/2)*(M/2+1)+m]*(u2[idz*ny*nx+idx*ny+idy+m] - 2*u2[idz*ny*nx+idx*ny+idy] + u2[idz*ny*nx+idx*ny+(idy-m)])/(dy*dy);	
					 
					 if(idz>=5)
							Uzz = Uzz + d_c2[(M/2)*(M/2+1)+m]*(u2[(idz+m)*ny*nx+idx*ny+idy] - 2*u2[idz*ny*nx+idx*ny+idy] + u2[(idz-m)*ny*nx+idx*ny+idy])/(dz*dz);	
					 else
							Uzz = Uzz + d_c2[(M/2)*(M/2+1)+m]*(u2[(idz+m)*ny*nx+idx*ny+idy] - 2*u2[idz*ny*nx+idx*ny+idy] + R[(5-m)*nx*ny+idx*ny+idy])/(dz*dz);	 	 
				}						 				
				u3[i] = 2*u2[i]-u1[i]+ dt*dt*((1+2*d_epsilon[i])*(Uxx+Uyy)+Uzz)*S[i]*d_vp[i]*d_vp[i];
	 }
	
	
}

__global__ void u_update_lpml(float *d_c2, float *d_dlr, float *d_ddlr, float *d_epsilon,float *d_delta, float *d_vp, float dx, 
															float dy,float dz, float dt, int nx, int ny, int nz, int pml, float *R, float *ux,float *u3,float *u2, float *S,
															float *wl11,float *wl12,float *wl13,float *wl21,float *wl31,float *wl32,float *wl33,float *pl1,float *pl2,float *pl3)
{
		int idx =  blockDim.y * blockIdx.y + threadIdx.y;
		int idy =  blockDim.x * blockIdx.x + threadIdx.x;
		int idz =  blockDim.z * blockIdx.z + threadIdx.z;
		if(idx>=M/2 && idx<pml && idy>=M/2 && idy<ny-M/2 && idz>=M/2 && idz<nz) //left
		//if(idx>=M/2 && idx<pml && idy>=M/2 && idy<ny-M/2 && idz>=M/2 && idz<nz-M/2)
	 {
				 int m, i, i1;
				 float Uxx = 0.0, Uyy = 0.0, Uzz = 0.0;
				 i = idz*ny*nx+idx*ny+idy;
				 i1 = idz*ny*pml+idx*ny+idy;
				 
				 for(m=1;m<M/2+1;m++)
				 {
					 Uxx = Uxx + d_c2[(M/2)*(M/2+1)+m]*(u2[idz*ny*nx+(idx+m)*ny+idy] - 2*u2[idz*ny*nx+idx*ny+idy] + u2[idz*ny*nx+(idx-m)*ny+idy])/(dx*dx);
					 Uyy = Uyy + d_c2[(M/2)*(M/2+1)+m]*(u2[idz*ny*nx+idx*ny+idy+m] - 2*u2[idz*ny*nx+idx*ny+idy] + u2[idz*ny*nx+idx*ny+(idy-m)])/(dy*dy);
					// Uzz = Uzz + d_c2[(M/2)*(M/2+1)+m]*(u2[(idz+m)*ny*nx+idx*ny+idy] - 2*u2[idz*ny*nx+idx*ny+idy] + u2[(idz-m)*ny*nx+idx*ny+idy])/(dz*dz);	

					 if(idz<nz-M/2)
							Uzz = Uzz + d_c2[(M/2)*(M/2+1)+m]*(u2[(idz+m)*ny*nx+idx*ny+idy] - 2*u2[idz*ny*nx+idx*ny+idy] + u2[(idz-m)*ny*nx+idx*ny+idy])/(dz*dz);	
					 else
							Uzz = Uzz + d_c2[(M/2)*(M/2+1)+m]*(R[(5-m)*nx*ny+idx*ny+idy] - 2*u2[idz*ny*nx+idx*ny+idy] + u2[(idz-m)*ny*nx+idx*ny+idy])/(dz*dz);	
				 }																				
								
				 wl13[i1] = (1+2*d_epsilon[i])*S[i]*pow(d_vp[i]*dt,2)*Uxx-(pow((d_dlr[pml-idx]*dt+1),2)-3)*wl12[i1] +(2*d_dlr[pml-idx]*dt-1)*wl11[i1];
				 pl3[i1] = (3-pow((1+dt*d_dlr[pml-idx]),2))*pl2[i1]+(2*d_dlr[pml-idx]*dt-1)*pl1[i1]+(1+2*d_epsilon[i])*S[i]*pow(d_vp[i]*dt,2)*d_ddlr[pml-idx]*ux[i];
				 wl21[i1] = dt*pl3[i1]+wl21[i1]*(1-dt*d_dlr[pml-idx]);
				 wl33[i1] = S[i]*pow(d_vp[i]*dt,2)*(Uzz+(1+2*d_epsilon[i])*Uyy) + 2*wl32[i1]-wl31[i1];
				 u3[i] = wl13[i1] + wl21[i1] + wl33[i1];
	 }
	 
}

__global__ void u_update_lpml2(float *d_c2, float *d_dlr, float *d_ddlr, float *d_epsilon,float *d_delta, float *d_vp, float dx, 
															float dy,float dz, float dt, int nx, int ny, int nz, int pml, float *R, float *ux,float *u3,float *u2, float *S,
															float *wl11,float *wl12,float *wl13,float *wl21,float *wl31,float *wl32,float *wl33,float *pl1,float *pl2,float *pl3)
{
		int idx =  blockDim.y * blockIdx.y + threadIdx.y;
		int idy =  blockDim.x * blockIdx.x + threadIdx.x;
		int idz =  blockDim.z * blockIdx.z + threadIdx.z;
		if(idx>=5 && idx<pml && idy>=5 && idy<ny-5 &&  idz<nz-5) //left
	 {
				 int m, i, i1;
				 float Uxx = 0.0, Uyy = 0.0, Uzz = 0.0;
				 i = idz*ny*nx+idx*ny+idy;
				 i1 = idz*ny*pml+idx*ny+idy;
				 
				 for(m=1;m<M/2+1;m++)
				 {
					 Uxx = Uxx + d_c2[(M/2)*(M/2+1)+m]*(u2[idz*ny*nx+(idx+m)*ny+idy] - 2*u2[idz*ny*nx+idx*ny+idy] + u2[idz*ny*nx+(idx-m)*ny+idy])/(dx*dx);
					 Uyy = Uyy + d_c2[(M/2)*(M/2+1)+m]*(u2[idz*ny*nx+idx*ny+idy+m] - 2*u2[idz*ny*nx+idx*ny+idy] + u2[idz*ny*nx+idx*ny+(idy-m)])/(dy*dy);

					 if(idz>=5)
							Uzz = Uzz + d_c2[(M/2)*(M/2+1)+m]*(u2[(idz+m)*ny*nx+idx*ny+idy] - 2*u2[idz*ny*nx+idx*ny+idy] + u2[(idz-m)*ny*nx+idx*ny+idy])/(dz*dz);	
					 else
							Uzz = Uzz + d_c2[(M/2)*(M/2+1)+m]*(u2[(idz+m)*ny*nx+idx*ny+idy] - 2*u2[idz*ny*nx+idx*ny+idy] + R[(5-m)*nx*ny+idx*ny+idy])/(dz*dz);	
				 }																				
								
				 wl13[i1] = (1+2*d_epsilon[i])*S[i]*pow(d_vp[i]*dt,2)*Uxx-(pow((d_dlr[pml-idx]*dt+1),2)-3)*wl12[i1] +(2*d_dlr[pml-idx]*dt-1)*wl11[i1];
				 pl3[i1] = (3-pow((1+dt*d_dlr[pml-idx]),2))*pl2[i1]+(2*d_dlr[pml-idx]*dt-1)*pl1[i1]+(1+2*d_epsilon[i])*S[i]*pow(d_vp[i]*dt,2)*d_ddlr[pml-idx]*ux[i];
				 wl21[i1] = dt*pl3[i1]+wl21[i1]*(1-dt*d_dlr[pml-idx]);
				 wl33[i1] = S[i]*pow(d_vp[i]*dt,2)*(Uzz+(1+2*d_epsilon[i])*Uyy) + 2*wl32[i1]-wl31[i1];
				 u3[i] = wl13[i1] + wl21[i1] + wl33[i1];
	 }
	 
}
__global__ void u_update_rpml(float *d_c2, float *d_dlr, float *d_ddlr, float *d_epsilon,float *d_delta, float *d_vp, float dx, 
															float dy,float dz, float dt, int nx, int ny, int nz, int pml, float *R,float *ux,float *u3,float *u2, float *S, 
															float *wr11,float *wr12,float *wr13,float *wr21,float *wr31,float *wr32,float *wr33,float *pr1,float *pr2,float *pr3)
{
		int idx =  blockDim.y * blockIdx.y + threadIdx.y;
		int idy =  blockDim.x * blockIdx.x + threadIdx.x;
		int idz =  blockDim.z * blockIdx.z + threadIdx.z;
		if(idx>=nx-pml && idx<nx-M/2 && idy>=M/2 && idy<ny-M/2 && idz>=M/2 && idz<nz) //right
	//	if(idx>=nx-pml && idx<nx-M/2 && idy>=M/2 && idy<ny-M/2 && idz>=M/2 && idz<nz-M/2)			
		{
				 int m, i, i2;
				 float Uxx = 0.0, Uyy = 0.0, Uzz = 0.0;
				 i = idz*ny*nx+idx*ny+idy;
				 i2 = idz*ny*pml+(idx-nx+pml)*ny+idy;	
				 for(m=1;m<M/2+1;m++)
				 {
					 Uxx = Uxx + d_c2[(M/2)*(M/2+1)+m]*(u2[idz*ny*nx+(idx+m)*ny+idy] - 2*u2[idz*ny*nx+idx*ny+idy] + u2[idz*ny*nx+(idx-m)*ny+idy])/(dx*dx);
					 Uyy = Uyy + d_c2[(M/2)*(M/2+1)+m]*(u2[idz*ny*nx+idx*ny+idy+m] - 2*u2[idz*ny*nx+idx*ny+idy] + u2[idz*ny*nx+idx*ny+(idy-m)])/(dy*dy);
				//	 Uzz = Uzz + d_c2[(M/2)*(M/2+1)+m]*(u2[(idz+m)*ny*nx+idx*ny+idy] - 2*u2[idz*ny*nx+idx*ny+idy] + u2[(idz-m)*ny*nx+idx*ny+idy])/(dz*dz);			
					 
					 if(idz<nz-M/2)
							Uzz = Uzz + d_c2[(M/2)*(M/2+1)+m]*(u2[(idz+m)*ny*nx+idx*ny+idy] - 2*u2[idz*ny*nx+idx*ny+idy] + u2[(idz-m)*ny*nx+idx*ny+idy])/(dz*dz);	
					 else	 
							Uzz = Uzz + d_c2[(M/2)*(M/2+1)+m]*(R[(M/2-m)*nx*ny+idx*ny+idy] - 2*u2[idz*ny*nx+idx*ny+idy] + u2[(idz-m)*ny*nx+idx*ny+idy])/(dz*dz);	 
				 }											
					wr13[i2] = (1+2*d_epsilon[i])*S[i]*pow(d_vp[i]*dt,2)*Uxx-(pow((d_dlr[idx-nx+pml]*dt+1),2)-3)*wr12[i2] +(2*d_dlr[idx-nx+pml]*dt-1)*wr11[i2];
					pr3[i2] = (3-pow((1+dt*d_dlr[idx-nx+pml]),2))*pr2[i2]+(2*d_dlr[idx-nx+pml]*dt-1)*pr1[i2]-(1+2*d_epsilon[i])*S[i]*pow(d_vp[i]*dt,2)*d_ddlr[idx-nx+pml]*ux[i];
					wr21[i2] = dt*pr3[i2]+wr21[i2]*(1-dt*d_dlr[idx-nx+pml]);
					wr33[i2] = S[i]*pow(d_vp[i]*dt,2)*(Uzz+(1+2*d_epsilon[i])*Uyy) + 2*wr32[i2]-wr31[i2];
					u3[i] = wr13[i2] + wr21[i2] + wr33[i2];
		}
		
}

__global__ void u_update_rpml2(float *d_c2, float *d_dlr, float *d_ddlr, float *d_epsilon,float *d_delta, float *d_vp, float dx, 
															float dy,float dz, float dt, int nx, int ny, int nz, int pml, float *R,float *ux,float *u3,float *u2, float *S, 
															float *wr11,float *wr12,float *wr13,float *wr21,float *wr31,float *wr32,float *wr33,float *pr1,float *pr2,float *pr3)
{
		int idx =  blockDim.y * blockIdx.y + threadIdx.y;
		int idy =  blockDim.x * blockIdx.x + threadIdx.x;
		int idz =  blockDim.z * blockIdx.z + threadIdx.z;
		if(idx>=nx-pml && idx<nx-5 && idy>=5 && idy<ny-5 &&  idz<nz-5) //right			
		{
				 int m, i, i2;
				 float Uxx = 0.0, Uyy = 0.0, Uzz = 0.0;
				 i = idz*ny*nx+idx*ny+idy;
				 i2 = idz*ny*pml+(idx-nx+pml)*ny+idy;	
				 for(m=1;m<M/2+1;m++)
				 {
					 Uxx = Uxx + d_c2[(M/2)*(M/2+1)+m]*(u2[idz*ny*nx+(idx+m)*ny+idy] - 2*u2[idz*ny*nx+idx*ny+idy] + u2[idz*ny*nx+(idx-m)*ny+idy])/(dx*dx);
					 Uyy = Uyy + d_c2[(M/2)*(M/2+1)+m]*(u2[idz*ny*nx+idx*ny+idy+m] - 2*u2[idz*ny*nx+idx*ny+idy] + u2[idz*ny*nx+idx*ny+(idy-m)])/(dy*dy);		
					 
					 if(idz>=5)
							Uzz = Uzz + d_c2[(M/2)*(M/2+1)+m]*(u2[(idz+m)*ny*nx+idx*ny+idy] - 2*u2[idz*ny*nx+idx*ny+idy] + u2[(idz-m)*ny*nx+idx*ny+idy])/(dz*dz);	
					 else
							Uzz = Uzz + d_c2[(M/2)*(M/2+1)+m]*(u2[(idz+m)*ny*nx+idx*ny+idy] - 2*u2[idz*ny*nx+idx*ny+idy] + R[(5-m)*nx*ny+idx*ny+idy])/(dz*dz);	 
				 }											
					wr13[i2] = (1+2*d_epsilon[i])*S[i]*pow(d_vp[i]*dt,2)*Uxx-(pow((d_dlr[idx-nx+pml]*dt+1),2)-3)*wr12[i2] +(2*d_dlr[idx-nx+pml]*dt-1)*wr11[i2];                                 
					pr3[i2] = (3-pow((1+dt*d_dlr[idx-nx+pml]),2))*pr2[i2]+(2*d_dlr[idx-nx+pml]*dt-1)*pr1[i2]-(1+2*d_epsilon[i])*S[i]*pow(d_vp[i]*dt,2)*d_ddlr[idx-nx+pml]*ux[i];
					wr21[i2] = dt*pr3[i2]+wr21[i2]*(1-dt*d_dlr[idx-nx+pml]);
					wr33[i2] = S[i]*pow(d_vp[i]*dt,2)*(Uzz+(1+2*d_epsilon[i])*Uyy) + 2*wr32[i2]-wr31[i2];
					u3[i] = wr13[i2] + wr21[i2] + wr33[i2];
		}
		
}
__global__ void u_update_tpml(float *d_c2, float *d_dtb, float *d_ddtb, float *d_epsilon,float *d_delta, float *d_vp, float dx, 
															float dy,float dz, float dt, int nx, int ny, int nz, int pml, float *uz,float *u3,float *u2, float *S,
															float *wt11,float *wt12,float *wt13,float *wt21,float *wt31,float *wt32,float *wt33,float *pt1,float *pt2,float *pt3)
{
		int idx =  blockDim.y * blockIdx.y + threadIdx.y;
		int idy =  blockDim.x * blockIdx.x + threadIdx.x;
		int idz =  blockDim.z * blockIdx.z + threadIdx.z;
		if(idz>=M/2 && idz<pml && idx>=M/2 && idx<nx-M/2 && idy>=M/2 && idy<ny-M/2) //top
		{
				 int m, i, i5;
				 float Uxx = 0.0, Uyy = 0.0, Uzz = 0.0;
				 i = idz*ny*nx+idx*ny+idy;
				 i5 = idz*nx*ny+idx*ny+idy;
				 for(m=1;m<M/2+1;m++)
				{
					 Uxx = Uxx + d_c2[(M/2)*(M/2+1)+m]*(u2[idz*ny*nx+(idx+m)*ny+idy] - 2*u2[idz*ny*nx+idx*ny+idy] + u2[idz*ny*nx+(idx-m)*ny+idy])/(dx*dx);
					 Uyy = Uyy + d_c2[(M/2)*(M/2+1)+m]*(u2[idz*ny*nx+idx*ny+idy+m] - 2*u2[idz*ny*nx+idx*ny+idy] + u2[idz*ny*nx+idx*ny+(idy-m)])/(dy*dy);				
					 Uzz = Uzz + d_c2[(M/2)*(M/2+1)+m]*(u2[(idz+m)*ny*nx+idx*ny+idy] - 2*u2[idz*ny*nx+idx*ny+idy] + u2[(idz-m)*ny*nx+idx*ny+idy])/(dz*dz);	 
				}				 				
				 wt13[i5] = S[i]*pow(d_vp[i]*dt,2)*Uzz-(pow((d_dtb[pml-idz]*dt+1),2)-3)*wt12[i5] +(2*d_dtb[pml-idz]*dt-1)*wt11[i5];
				 pt3[i5] = (3-pow((1+dt*d_dtb[pml-idz]),2))*pt2[i5]+(2*d_dtb[pml-idz]*dt-1)*pt1[i5]+S[i]*pow(d_vp[i]*dt,2)*d_ddtb[pml-idz]*uz[i];
				 wt21[i5] = dt*pt3[i5]+wt21[i5]*(1-dt*d_dtb[pml-idz]);
				 wt33[i5] = (1+2*d_epsilon[i])*S[i]*pow(d_vp[i]*dt,2)*(Uxx+Uyy)+2*wt32[i5]-wt31[i5];
				 u3[i] = wt13[i5]+wt21[i5]+ wt33[i5];
		}
		
}


__global__ void u_update_bpml(float *R, float *d_c2, float *d_dtb, float *d_ddtb, float *d_epsilon,float *d_delta, float *d_vp, float dx, 
															float dy,float dz, float dt, int nx, int ny, int nz, int pml, float *uz,float *u3,float *u2, float *S,
															 float *wb11,float *wb12,float *wb13,float *wb21,float *wb31,float *wb32,float *wb33,float *pb1,float *pb2,float*pb3)
{
		int idx =  blockDim.y * blockIdx.y + threadIdx.y;
		int idy =  blockDim.x * blockIdx.x + threadIdx.x;
		int idz =  blockDim.z * blockIdx.z + threadIdx.z;
	//	if(idz>=nz-pml && idz<nz && idx>=M/2 && idx<nx-M/2 && idy>=M/2 && idy<ny-M/2) //bottom3
		if(idz>=nz-pml && idz<nz-M/2 && idx>=M/2 && idx<nx-M/2 && idy>=M/2 && idy<ny-M/2)
		{		
				 int m, i, i6;
				 float Uxx = 0.0, Uyy = 0.0, Uzz = 0.0;
				 i = idz*ny*nx+idx*ny+idy;
				 i6 = (idz-nz+pml)*ny*nx+idx*ny+idy;	
				 for(m=1;m<M/2+1;m++)
				{
					 Uxx = Uxx + d_c2[(M/2)*(M/2+1)+m]*(u2[idz*ny*nx+(idx+m)*ny+idy] - 2*u2[idz*ny*nx+idx*ny+idy] + u2[idz*ny*nx+(idx-m)*ny+idy])/(dx*dx);
					 Uyy = Uyy + d_c2[(M/2)*(M/2+1)+m]*(u2[idz*ny*nx+idx*ny+idy+m] - 2*u2[idz*ny*nx+idx*ny+idy] + u2[idz*ny*nx+idx*ny+(idy-m)])/(dy*dy);
					 Uzz = Uzz + d_c2[(M/2)*(M/2+1)+m]*(u2[(idz+m)*ny*nx+idx*ny+idy] - 2*u2[idz*ny*nx+idx*ny+idy] + u2[(idz-m)*ny*nx+idx*ny+idy])/(dz*dz);

		//			 if(idz<nz-5)
		//					Uzz = Uzz + d_c2[(M/2)*(M/2+1)+m]*(u2[(idz+m)*ny*nx+idx*ny+idy] - 2*u2[idz*ny*nx+idx*ny+idy] + u2[(idz-m)*ny*nx+idx*ny+idy])/(dz*dz);	
		//			 else 
		//					Uzz = Uzz + d_c2[(M/2)*(M/2+1)+m]*(R[(5-m)*nx*ny+idx*ny+idy] - 2*u2[idz*ny*nx+idx*ny+idy] + u2[(idz-m)*ny*nx+idx*ny+idy])/(dz*dz);	
						
			
				}				 
				 wb13[i6] = S[i]*pow(d_vp[i]*dt,2)*Uzz-(pow((d_dtb[idz-nz+pml]*dt+1),2)-3)*wb12[i6]+(2*d_dtb[idz-nz+pml]*dt-1)*wb11[i6];
				 pb3[i6] = (3-pow((1+dt*d_dtb[idz-nz+pml]),2))*pb2[i6]+(2*d_dtb[idz-nz+pml]*dt-1)*pb1[i6]-S[i]*pow(d_vp[i]*dt,2)*d_ddtb[idz-nz+pml]*uz[i];
				 wb21[i6] = dt*pb3[i6]+wb21[i6]*(1-dt*d_dtb[idz-nz+pml]);
				 wb33[i6] =  (1+2*d_epsilon[i])*S[i]*pow(d_vp[i]*dt,2)*(Uxx+Uyy)+2*wb32[i6]-wb31[i6];
				 u3[i] = wb13[i6] + wb21[i6] + wb33[i6];
		}
}


__global__ void u_update_fpml(float *d_c2, float *d_dfb, float *d_ddfb, float *d_epsilon,float *d_delta, float *d_vp, float dx, 
															float dy,float dz, float dt, int nx, int ny, int nz, int pml, float *R, float *uy,float *u3,float *u2, float *S,
															float *wf11,float *wf12,float *wf13,float *wf21,float *wf31,float *wf32,float *wf33,float *pf1,float *pf2,float*pf3)
{
		int idx =  blockDim.y * blockIdx.y + threadIdx.y;
		int idy =  blockDim.x * blockIdx.x + threadIdx.x;
		int idz =  blockDim.z * blockIdx.z + threadIdx.z;
		if(idy>=M/2 && idy<pml && idx>=M/2 && idx<nx-M/2 && idz>=M/2 && idz<nz) //forward
	//	if(idy>=M/2 && idy<pml && idx>=M/2 && idx<nx-M/2 && idz>=M/2 && idz<nz-M/2)
		{	  		 
				 int m, i, i3;
				 float Uxx = 0.0, Uyy = 0.0, Uzz = 0.0;
				 i = idz*ny*nx+idx*ny+idy;
				 i3 = idz*nx*pml+idx*pml+idy;		
				 for(m=1;m<M/2+1;m++)
				{
					 Uxx = Uxx + d_c2[(M/2)*(M/2+1)+m]*(u2[idz*ny*nx+(idx+m)*ny+idy] - 2*u2[idz*ny*nx+idx*ny+idy] + u2[idz*ny*nx+(idx-m)*ny+idy])/(dx*dx);
					 Uyy = Uyy + d_c2[(M/2)*(M/2+1)+m]*(u2[idz*ny*nx+idx*ny+idy+m] - 2*u2[idz*ny*nx+idx*ny+idy] + u2[idz*ny*nx+idx*ny+(idy-m)])/(dy*dy);
					// Uzz = Uzz + d_c2[(M/2)*(M/2+1)+m]*(u2[(idz+m)*ny*nx+idx*ny+idy] - 2*u2[idz*ny*nx+idx*ny+idy] + u2[(idz-m)*ny*nx+idx*ny+idy])/(dz*dz);
					 
					 if(idz<nz-M/2)
							Uzz = Uzz + d_c2[(M/2)*(M/2+1)+m]*(u2[(idz+m)*ny*nx+idx*ny+idy] - 2*u2[idz*ny*nx+idx*ny+idy] + u2[(idz-m)*ny*nx+idx*ny+idy])/(dz*dz);	
					 else	 
							Uzz = Uzz + d_c2[(M/2)*(M/2+1)+m]*(R[(M/2-m)*nx*ny+idx*ny+idy] - 2*u2[idz*ny*nx+idx*ny+idy] + u2[(idz-m)*ny*nx+idx*ny+idy])/(dz*dz);	 
				}
				 wf13[i3] = (1+2*d_epsilon[i])*S[i]*pow(d_vp[i]*dt,2)*Uyy-(pow((d_dfb[pml-idy]*dt+1),2)-3)*wf12[i3] + (2*d_dfb[pml-idy]*dt-1)*wf11[i3];		                               
				 pf3[i3] = (3-pow((1+dt*d_dfb[pml-idy]),2))*pf2[i3]+(2*d_dfb[pml-idy]*dt-1)*pf1[i3]+(1+2*d_epsilon[i])*S[i]*pow(d_vp[i]*dt,2)*d_ddfb[pml-idy]*uy[i];
				 wf21[i3] = dt*pf3[i3]+wf21[i3]*(1-dt*d_dfb[pml-idy]);
				 wf33[i3] = S[i]*pow(d_vp[i]*dt,2)*(Uzz+(1+2*d_epsilon[i])*Uxx)+ 2*wf32[i3]-wf31[i3];
				 u3[i] = wf13[i3] + wf21[i3] + wf33[i3];
		}
}

__global__ void u_update_fpml2(float *d_c2, float *d_dfb, float *d_ddfb, float *d_epsilon,float *d_delta, float *d_vp, float dx, 
															float dy,float dz, float dt, int nx, int ny, int nz, int pml, float *R, float *uy,float *u3,float *u2, float *S,
															float *wf11,float *wf12,float *wf13,float *wf21,float *wf31,float *wf32,float *wf33,float *pf1,float *pf2,float*pf3)
{
		int idx =  blockDim.y * blockIdx.y + threadIdx.y;
		int idy =  blockDim.x * blockIdx.x + threadIdx.x;
		int idz =  blockDim.z * blockIdx.z + threadIdx.z;
		if(idy>=5 && idy<pml && idx>=5 && idx<nx-5 && idz<nz-5) //forward
		{	  		 
				 int m, i, i3;
				 float Uxx = 0.0, Uyy = 0.0, Uzz = 0.0;
				 i = idz*ny*nx+idx*ny+idy;
				 i3 = idz*nx*pml+idx*pml+idy;		
				 for(m=1;m<M/2+1;m++)
				{
					 Uxx = Uxx + d_c2[(M/2)*(M/2+1)+m]*(u2[idz*ny*nx+(idx+m)*ny+idy] - 2*u2[idz*ny*nx+idx*ny+idy] + u2[idz*ny*nx+(idx-m)*ny+idy])/(dx*dx);
					 Uyy = Uyy + d_c2[(M/2)*(M/2+1)+m]*(u2[idz*ny*nx+idx*ny+idy+m] - 2*u2[idz*ny*nx+idx*ny+idy] + u2[idz*ny*nx+idx*ny+(idy-m)])/(dy*dy);
					 
					 if(idz>=5)
							Uzz = Uzz + d_c2[(M/2)*(M/2+1)+m]*(u2[(idz+m)*ny*nx+idx*ny+idy] - 2*u2[idz*ny*nx+idx*ny+idy] + u2[(idz-m)*ny*nx+idx*ny+idy])/(dz*dz);	
					 else
							Uzz = Uzz + d_c2[(M/2)*(M/2+1)+m]*(u2[(idz+m)*ny*nx+idx*ny+idy] - 2*u2[idz*ny*nx+idx*ny+idy] + R[(5-m)*nx*ny+idx*ny+idy])/(dz*dz);	 
				}
				 wf13[i3] = (1+2*d_epsilon[i])*S[i]*pow(d_vp[i]*dt,2)*Uyy-(pow((d_dfb[pml-idy]*dt+1),2)-3)*wf12[i3] + (2*d_dfb[pml-idy]*dt-1)*wf11[i3];					                               
				 pf3[i3] = (3-pow((1+dt*d_dfb[pml-idy]),2))*pf2[i3]+(2*d_dfb[pml-idy]*dt-1)*pf1[i3]+(1+2*d_epsilon[i])*S[i]*pow(d_vp[i]*dt,2)*d_ddfb[pml-idy]*uy[i];
				 wf21[i3] = dt*pf3[i3]+wf21[i3]*(1-dt*d_dfb[pml-idy]);
				 wf33[i3] = S[i]*pow(d_vp[i]*dt,2)*(Uzz+(1+2*d_epsilon[i])*Uxx)+ 2*wf32[i3]-wf31[i3];
				 u3[i] = wf13[i3] + wf21[i3] + wf33[i3];
		}
}

__global__ void u_update_bapml(float *d_c2, float *d_dfb, float *d_ddfb, float *d_epsilon,float *d_delta, float *d_vp, float dx, 
															 float dy,float dz, float dt, int nx, int ny, int nz, int pml, float *R, float *uy,float *u3,float *u2, float *S,
															 float *wba11,float *wba12,float *wba13,float *wba21,float *wba31,float *wba32,float *wba33,float *pba1,float *pba2,float*pba3)  															 											                                                          
{
		int idx =  blockDim.y * blockIdx.y + threadIdx.y;
		int idy =  blockDim.x * blockIdx.x + threadIdx.x;
		int idz =  blockDim.z * blockIdx.z + threadIdx.z;
		if(idy>=ny-pml && idy<ny-M/2 && idx>=M/2 && idx<nx-M/2 && idz>=M/2 && idz<nz) //backward		
	//	if(idy>=ny-pml && idy<ny-M/2 && idx>=M/2 && idx<nx-M/2 && idz>=M/2 && idz<nz-M/2) 
	 {
				int m, i, i4;
				float Uxx = 0.0, Uyy = 0.0, Uzz = 0.0;
				i = idz*ny*nx+idx*ny+idy;
				i4 = idz*nx*pml+idx*pml+idy-ny+pml;		
				for(m=1;m<M/2+1;m++)
				{
					 Uxx = Uxx + d_c2[(M/2)*(M/2+1)+m]*(u2[idz*ny*nx+(idx+m)*ny+idy] - 2*u2[idz*ny*nx+idx*ny+idy] + u2[idz*ny*nx+(idx-m)*ny+idy])/(dx*dx);
					 Uyy = Uyy + d_c2[(M/2)*(M/2+1)+m]*(u2[idz*ny*nx+idx*ny+idy+m] - 2*u2[idz*ny*nx+idx*ny+idy] + u2[idz*ny*nx+idx*ny+(idy-m)])/(dy*dy);	
				//	 Uzz = Uzz + d_c2[(M/2)*(M/2+1)+m]*(u2[(idz+m)*ny*nx+idx*ny+idy] - 2*u2[idz*ny*nx+idx*ny+idy] + u2[(idz-m)*ny*nx+idx*ny+idy])/(dz*dz);	
					 
					 if(idz<nz-5)
							Uzz = Uzz + d_c2[(M/2)*(M/2+1)+m]*(u2[(idz+m)*ny*nx+idx*ny+idy] - 2*u2[idz*ny*nx+idx*ny+idy] + u2[(idz-m)*ny*nx+idx*ny+idy])/(dz*dz);	
					 else 	 
							Uzz = Uzz + d_c2[(M/2)*(M/2+1)+m]*(R[(M/2-m)*nx*ny+idx*ny+idy] - 2*u2[idz*ny*nx+idx*ny+idy] + u2[(idz-m)*ny*nx+idx*ny+idy])/(dz*dz);	 	 
				}
				wba13[i4] = (1+2*d_epsilon[i])*S[i]*pow(d_vp[i]*dt,2)*Uyy-(pow((d_dfb[idy-ny+pml]*dt+1),2)-3)*wba12[i4] +(2*d_dfb[idy-ny+pml]*dt-1)*wba11[i4];
				pba3[i4] = (3-pow((1+dt*d_dfb[idy-ny+pml]),2))*pba2[i4]+(2*d_dfb[idy-ny+pml]*dt-1)*pba1[i4]-(1+2*d_epsilon[i])*S[i]*pow(d_vp[i]*dt,2)*d_ddfb[idy-ny+pml]*uy[i];
				wba21[i4] = dt*pba3[i4]+wba21[i4]*(1-dt*d_dfb[idy-ny+pml]);
				wba33[i4] =  S[i]*pow(d_vp[i]*dt,2)*(Uzz+(1+2*d_epsilon[i])*Uxx)+2*wba32[i4]-wba31[i4];																							
				u3[i] = wba13[i4] + wba21[i4] + wba33[i4];
	 }
}

__global__ void u_update_bapml2(float *d_c2, float *d_dfb, float *d_ddfb, float *d_epsilon,float *d_delta, float *d_vp, float dx, 
															 float dy,float dz, float dt, int nx, int ny, int nz, int pml, float *R, float *uy,float *u3,float *u2, float *S,
															 float *wba11,float *wba12,float *wba13,float *wba21,float *wba31,float *wba32,float *wba33,float *pba1,float *pba2,float*pba3)                                                           
{
		int idx =  blockDim.y * blockIdx.y + threadIdx.y;
		int idy =  blockDim.x * blockIdx.x + threadIdx.x;
		int idz =  blockDim.z * blockIdx.z + threadIdx.z;
		if(idy>=ny-pml && idy<ny-5 && idx>=5 && idx<nx-5 && idz<nz-5) //backward		
	 {
				int m, i, i4;
				float Uxx = 0.0, Uyy = 0.0, Uzz = 0.0;
				i = idz*ny*nx+idx*ny+idy;
				i4 = idz*nx*pml+idx*pml+idy-ny+pml;		
				for(m=1;m<M/2+1;m++)
				{
					 Uxx = Uxx + d_c2[(M/2)*(M/2+1)+m]*(u2[idz*ny*nx+(idx+m)*ny+idy] - 2*u2[idz*ny*nx+idx*ny+idy] + u2[idz*ny*nx+(idx-m)*ny+idy])/(dx*dx);
					 Uyy = Uyy + d_c2[(M/2)*(M/2+1)+m]*(u2[idz*ny*nx+idx*ny+idy+m] - 2*u2[idz*ny*nx+idx*ny+idy] + u2[idz*ny*nx+idx*ny+(idy-m)])/(dy*dy);		
					 
					 if(idz>=5)
							Uzz = Uzz + d_c2[(M/2)*(M/2+1)+m]*(u2[(idz+m)*ny*nx+idx*ny+idy] - 2*u2[idz*ny*nx+idx*ny+idy] + u2[(idz-m)*ny*nx+idx*ny+idy])/(dz*dz);	
					 else
							Uzz = Uzz + d_c2[(M/2)*(M/2+1)+m]*(u2[(idz+m)*ny*nx+idx*ny+idy] - 2*u2[idz*ny*nx+idx*ny+idy] + R[(5-m)*nx*ny+idx*ny+idy])/(dz*dz);	 	 
				}
				wba13[i4] = (1+2*d_epsilon[i])*S[i]*pow(d_vp[i]*dt,2)*Uyy-(pow((d_dfb[idy-ny+pml]*dt+1),2)-3)*wba12[i4] +(2*d_dfb[idy-ny+pml]*dt-1)*wba11[i4];
				pba3[i4] = (3-pow((1+dt*d_dfb[idy-ny+pml]),2))*pba2[i4]+(2*d_dfb[idy-ny+pml]*dt-1)*pba1[i4]-(1+2*d_epsilon[i])*S[i]*pow(d_vp[i]*dt,2)*d_ddfb[idy-ny+pml]*uy[i];
				wba21[i4] = dt*pba3[i4]+wba21[i4]*(1-dt*d_dfb[idy-ny+pml]);
				wba33[i4] =  S[i]*pow(d_vp[i]*dt,2)*(Uzz+(1+2*d_epsilon[i])*Uxx)+2*wba32[i4]-wba31[i4];																							
				u3[i] = wba13[i4] + wba21[i4] + wba33[i4];
	 }
}
*/
__global__ void wavefield_update(float *d_c, float *d_c2, float *d_dlr, float *d_ddlr, float *d_dtb, float *d_ddtb, float *d_dfb,
																 float *d_ddfb, float *d_epsilon, float *d_delta, float *d_vp, float dx, float dy, float dz, float dt,
																 int nx, int ny, int nz, int pml, int sz, float *ux, float *uy, float *uz, float *u1, float *u3, float *u2, float *S,
																 float *wl11, float *wl12, float *wl13, float *wl21, float *wl31, float *wl32, float *wl33, float *pl1, float *pl2, float *pl3,
																 float *wr11, float *wr12, float *wr13, float *wr21, float *wr31, float *wr32, float *wr33, float *pr1, float *pr2, float *pr3,
																 float *wt11, float *wt12, float *wt13, float *wt21, float *wt31, float *wt32, float *wt33, float *pt1, float *pt2, float *pt3,
																 float *wb11, float *wb12, float *wb13, float *wb21, float *wb31, float *wb32, float *wb33, float *pb1, float *pb2, float *pb3,
																 float *wf11, float *wf12, float *wf13, float *wf21, float *wf31, float *wf32, float *wf33, float *pf1, float *pf2, float *pf3,
																 float *wba11, float *wba12, float *wba13, float *wba21, float *wba31, float *wba32, float *wba33, float *pba1, float *pba2, float *pba3)
{
	int idx = blockDim.y * blockIdx.y + threadIdx.y;
	int idy = blockDim.x * blockIdx.x + threadIdx.x;
	int idz = blockDim.z * blockIdx.z + threadIdx.z;
	if (idx >= 5 && idx < nx - 5 && idy >= 5 && idy < ny - 5 && idz >= 5 && idz < nz - 5)
	{
		int m, i, i1, i2, i3, i4, i5, i6;
		float Uxx = 0.0, Uyy = 0.0, Uzz = 0.0;
		i = idz * ny * nx + idx * ny + idy;
		for (m = 1; m < M / 2 + 1; m++)
		{
			Uxx = Uxx + d_c2[(M / 2) * (M / 2 + 1) + m] * (u2[idz * ny * nx + (idx + m) * ny + idy] - 2 * u2[idz * ny * nx + idx * ny + idy] + u2[idz * ny * nx + (idx - m) * ny + idy]) / (dx * dx);
			Uyy = Uyy + d_c2[(M / 2) * (M / 2 + 1) + m] * (u2[idz * ny * nx + idx * ny + idy + m] - 2 * u2[idz * ny * nx + idx * ny + idy] + u2[idz * ny * nx + idx * ny + (idy - m)]) / (dy * dy);
			Uzz = Uzz + d_c2[(M / 2) * (M / 2 + 1) + m] * (u2[(idz + m) * ny * nx + idx * ny + idy] - 2 * u2[idz * ny * nx + idx * ny + idy] + u2[(idz - m) * ny * nx + idx * ny + idy]) / (dz * dz);
		}
		if (idx >= pml && idx < nx - pml && idy >= pml && idy < ny - pml && idz >= pml && idz < nz - pml)
		{
			u3[i] = 2 * u2[i] - u1[i] + dt * dt * ((1 + 2 * d_epsilon[i]) * (Uxx + Uyy) + Uzz) * S[i] * d_vp[i] * d_vp[i];
		}
		else if (idx < pml && idy < ny && idz < nz) //left
		{
			i1 = idz * ny * pml + idx * ny + idy;
			wl13[i1] = (1 + 2 * d_epsilon[i]) * S[i] * pow(d_vp[i] * dt, 2) * Uxx - (pow((d_dlr[pml - idx] * dt + 1), 2) - 3) * wl12[i1] + (2 * d_dlr[pml - idx] * dt - 1) * wl11[i1];
			pl3[i1] = (3 - pow((1 + dt * d_dlr[pml - idx]), 2)) * pl2[i1] + (2 * d_dlr[pml - idx] * dt - 1) * pl1[i1] + (1 + 2 * d_epsilon[i]) * S[i] * pow(d_vp[i] * dt, 2) * d_ddlr[pml - idx] * ux[i];
			wl21[i1] = dt * pl3[i1] + wl21[i1] * (1 - dt * d_dlr[pml - idx]);
			wl33[i1] = S[i] * pow(d_vp[i] * dt, 2) * (Uzz + (1 + 2 * d_epsilon[i]) * Uyy) + 2 * wl32[i1] - wl31[i1];
			u3[i] = wl13[i1] + wl21[i1] + wl33[i1];
		}
		else if (idx >= nx - pml && idx < nx && idy < ny && idz < nz) //right
		{
			i2 = idz * ny * pml + (idx - nx + pml) * ny + idy;
			wr13[i2] = (1 + 2 * d_epsilon[i]) * S[i] * pow(d_vp[i] * dt, 2) * Uxx - (pow((d_dlr[idx - nx + pml] * dt + 1), 2) - 3) * wr12[i2] + (2 * d_dlr[idx - nx + pml] * dt - 1) * wr11[i2];
			pr3[i2] = (3 - pow((1 + dt * d_dlr[idx - nx + pml]), 2)) * pr2[i2] + (2 * d_dlr[idx - nx + pml] * dt - 1) * pr1[i2] - (1 + 2 * d_epsilon[i]) * S[i] * pow(d_vp[i] * dt, 2) * d_ddlr[idx - nx + pml] * ux[i];
			wr21[i2] = dt * pr3[i2] + wr21[i2] * (1 - dt * d_dlr[idx - nx + pml]);
			wr33[i2] = S[i] * pow(d_vp[i] * dt, 2) * (Uzz + (1 + 2 * d_epsilon[i]) * Uyy) + 2 * wr32[i2] - wr31[i2];
			u3[i] = wr13[i2] + wr21[i2] + wr33[i2];
		}
		else if (idy < pml && idx < nx && idz < nz) //forward
		{
			i3 = idz * nx * pml + idx * pml + idy;
			wf13[i3] = (1 + 2 * d_epsilon[i]) * S[i] * pow(d_vp[i] * dt, 2) * Uyy - (pow((d_dfb[pml - idy] * dt + 1), 2) - 3) * wf12[i3] + (2 * d_dfb[pml - idy] * dt - 1) * wf11[i3];
			pf3[i3] = (3 - pow((1 + dt * d_dfb[pml - idy]), 2)) * pf2[i3] + (2 * d_dfb[pml - idy] * dt - 1) * pf1[i3] + (1 + 2 * d_epsilon[i]) * S[i] * pow(d_vp[i] * dt, 2) * d_ddfb[pml - idy] * uy[i];
			wf21[i3] = dt * pf3[i3] + wf21[i3] * (1 - dt * d_dfb[pml - idy]);
			wf33[i3] = S[i] * pow(d_vp[i] * dt, 2) * (Uzz + (1 + 2 * d_epsilon[i]) * Uxx) + 2 * wf32[i3] - wf31[i3];
			u3[i] = wf13[i3] + wf21[i3] + wf33[i3];
		}
		else if (idy >= ny - pml && idy < ny && idx < nx && idz < nz) //backward
		{
			i4 = idz * nx * pml + idx * pml + idy - ny + pml;
			wba13[i4] = (1 + 2 * d_epsilon[i]) * S[i] * pow(d_vp[i] * dt, 2) * Uyy - (pow((d_dfb[idy - ny + pml] * dt + 1), 2) - 3) * wba12[i4] + (2 * d_dfb[idy - ny + pml] * dt - 1) * wba11[i4];
			pba3[i4] = (3 - pow((1 + dt * d_dfb[idy - ny + pml]), 2)) * pba2[i4] + (2 * d_dfb[idy - ny + pml] * dt - 1) * pba1[i4] - (1 + 2 * d_epsilon[i]) * S[i] * pow(d_vp[i] * dt, 2) * d_ddfb[idy - ny + pml] * uy[i];
			wba21[i4] = dt * pba3[i4] + wba21[i4] * (1 - dt * d_dfb[idy - ny + pml]);
			wba33[i4] = S[i] * pow(d_vp[i] * dt, 2) * (Uzz + (1 + 2 * d_epsilon[i]) * Uxx) + 2 * wba32[i4] - wba31[i4];
			u3[i] = wba13[i4] + wba21[i4] + wba33[i4];
		}
		else if (idz < pml && idx < nx && idy < ny) //top
		{
			i5 = idz * nx * ny + idx * ny + idy;
			wt13[i5] = S[i] * pow(d_vp[i] * dt, 2) * Uzz - (pow((d_dtb[pml - idz] * dt + 1), 2) - 3) * wt12[i5] + (2 * d_dtb[pml - idz] * dt - 1) * wt11[i5];
			pt3[i5] = (3 - pow((1 + dt * d_dtb[pml - idz]), 2)) * pt2[i5] + (2 * d_dtb[pml - idz] * dt - 1) * pt1[i5] + S[i] * pow(d_vp[i] * dt, 2) * d_ddtb[pml - idz] * uz[i];
			wt21[i5] = dt * pt3[i5] + wt21[i5] * (1 - dt * d_dtb[pml - idz]);
			wt33[i5] = (1 + 2 * d_epsilon[i]) * S[i] * pow(d_vp[i] * dt, 2) * (Uxx + Uyy) + 2 * wt32[i5] - wt31[i5];
			u3[i] = wt13[i5] + wt21[i5] + wt33[i5];
		}
		else if (idz >= nz - pml && idz < nz && idx < nx && idy < ny) //bottom
		{
			i6 = (idz - nz + pml) * ny * nx + idx * ny + idy;
			wb13[i6] = S[i] * pow(d_vp[i] * dt, 2) * Uzz - (pow((d_dtb[idz - nz + pml] * dt + 1), 2) - 3) * wb12[i6] + (2 * d_dtb[idz - nz + pml] * dt - 1) * wb11[i6];
			pb3[i6] = (3 - pow((1 + dt * d_dtb[idz - nz + pml]), 2)) * pb2[i6] + (2 * d_dtb[idz - nz + pml] * dt - 1) * pb1[i6] - S[i] * pow(d_vp[i] * dt, 2) * d_ddtb[idz - nz + pml] * uz[i];
			wb21[i6] = dt * pb3[i6] + wb21[i6] * (1 - dt * d_dtb[idz - nz + pml]);
			wb33[i6] = (1 + 2 * d_epsilon[i]) * S[i] * pow(d_vp[i] * dt, 2) * (Uxx + Uyy) + 2 * wb32[i6] - wb31[i6];
			u3[i] = wb13[i6] + wb21[i6] + wb33[i6];
		}
	}
}

__global__ void wavefield_update1(float *d_R, float *d_c, float *d_c2, float *d_dlr, float *d_ddlr, float *d_dtb, float *d_ddtb, float *d_dfb,
																	float *d_ddfb, float *d_epsilon, float *d_delta, float *d_vp, float dx, float dy, float dz, float dt,
																	int nx, int ny, int nz, int pml, float *ux, float *uy, float *uz, float *u1, float *u3, float *u2, float *S,
																	float *wl11, float *wl12, float *wl13, float *wl21, float *wl31, float *wl32, float *wl33, float *pl1, float *pl2, float *pl3,
																	float *wr11, float *wr12, float *wr13, float *wr21, float *wr31, float *wr32, float *wr33, float *pr1, float *pr2, float *pr3,
																	float *wt11, float *wt12, float *wt13, float *wt21, float *wt31, float *wt32, float *wt33, float *pt1, float *pt2, float *pt3,
																	float *wf11, float *wf12, float *wf13, float *wf21, float *wf31, float *wf32, float *wf33, float *pf1, float *pf2, float *pf3,
																	float *wba11, float *wba12, float *wba13, float *wba21, float *wba31, float *wba32, float *wba33, float *pba1, float *pba2, float *pba3)
{
	int idx = blockDim.y * blockIdx.y + threadIdx.y;
	int idy = blockDim.x * blockIdx.x + threadIdx.x;
	int idz = blockDim.z * blockIdx.z + threadIdx.z;
	if (idx >= M / 2 && idx < nx - M / 2 && idy >= M / 2 && idy < ny - M / 2 && idz >= M / 2 && idz < nz - M / 2)
	{
		int m, i, i1, i2, i3, i4, i5, i6;
		float Uxx = 0.0, Uyy = 0.0, Uzz = 0.0;
		i = idz * ny * nx + idx * ny + idy;
		for (m = 1; m < M / 2 + 1; m++)
		{
			Uxx = Uxx + d_c2[(M / 2) * (M / 2 + 1) + m] * (u2[idz * ny * nx + (idx + m) * ny + idy] - 2 * u2[idz * ny * nx + idx * ny + idy] + u2[idz * ny * nx + (idx - m) * ny + idy]) / (dx * dx);
			Uyy = Uyy + d_c2[(M / 2) * (M / 2 + 1) + m] * (u2[idz * ny * nx + idx * ny + idy + m] - 2 * u2[idz * ny * nx + idx * ny + idy] + u2[idz * ny * nx + idx * ny + (idy - m)]) / (dy * dy);
			Uzz = Uzz + d_c2[(M / 2) * (M / 2 + 1) + m] * (u2[(idz + m) * ny * nx + idx * ny + idy] - 2 * u2[idz * ny * nx + idx * ny + idy] + u2[(idz - m) * ny * nx + idx * ny + idy]) / (dz * dz);

			/*	if(idz<nz-M/2)
						Uzz = Uzz + d_c2[(M/2)*(M/2+1)+m]*(u2[(idz+m)*ny*nx+idx*ny+idy] - 2*u2[idz*ny*nx+idx*ny+idy] + u2[(idz-m)*ny*nx+idx*ny+idy])/(dz*dz);	
					else
						Uzz = Uzz + d_c2[(M/2)*(M/2+1)+m]*(d_R[(5-m)*nx*ny+idx*ny+idy] - 2*u2[idz*ny*nx+idx*ny+idy] + u2[(idz-m)*ny*nx+idx*ny+idy])/(dz*dz);*/
		}
		if (idx >= pml && idx < nx - pml && idy >= pml && idy < ny - pml && idz >= pml && idz < nz)
		{
			u3[i] = 2 * u2[i] - u1[i] + dt * dt * ((1 + 2 * d_epsilon[i]) * (Uxx + Uyy) + Uzz) * S[i] * d_vp[i] * d_vp[i];
		}
		else if (idx < pml && idy < ny && idz < nz) //left
		{
			i1 = idz * ny * pml + idx * ny + idy;
			wl13[i1] = (1 + 2 * d_epsilon[i]) * S[i] * pow(d_vp[i] * dt, 2) * Uxx - (pow((d_dlr[pml - idx] * dt + 1), 2) - 3) * wl12[i1] + (2 * d_dlr[pml - idx] * dt - 1) * wl11[i1];
			pl3[i1] = (3 - pow((1 + dt * d_dlr[pml - idx]), 2)) * pl2[i1] + (2 * d_dlr[pml - idx] * dt - 1) * pl1[i1] + (1 + 2 * d_epsilon[i]) * S[i] * pow(d_vp[i] * dt, 2) * d_ddlr[pml - idx] * ux[i];
			wl21[i1] = dt * pl3[i1] + wl21[i1] * (1 - dt * d_dlr[pml - idx]);
			wl33[i1] = S[i] * pow(d_vp[i] * dt, 2) * (Uzz + (1 + 2 * d_epsilon[i]) * Uyy) + 2 * wl32[i1] - wl31[i1];
			u3[i] = wl13[i1] + wl21[i1] + wl33[i1];
		}
		else if (idx >= nx - pml && idx < nx && idy < ny && idz < nz) //right
		{
			i2 = idz * ny * pml + (idx - nx + pml) * ny + idy;
			wr13[i2] = (1 + 2 * d_epsilon[i]) * S[i] * pow(d_vp[i] * dt, 2) * Uxx - (pow((d_dlr[idx - nx + pml] * dt + 1), 2) - 3) * wr12[i2] + (2 * d_dlr[idx - nx + pml] * dt - 1) * wr11[i2];
			pr3[i2] = (3 - pow((1 + dt * d_dlr[idx - nx + pml]), 2)) * pr2[i2] + (2 * d_dlr[idx - nx + pml] * dt - 1) * pr1[i2] - (1 + 2 * d_epsilon[i]) * S[i] * pow(d_vp[i] * dt, 2) * d_ddlr[idx - nx + pml] * ux[i];
			wr21[i2] = dt * pr3[i2] + wr21[i2] * (1 - dt * d_dlr[idx - nx + pml]);
			wr33[i2] = S[i] * pow(d_vp[i] * dt, 2) * (Uzz + (1 + 2 * d_epsilon[i]) * Uyy) + 2 * wr32[i2] - wr31[i2];
			u3[i] = wr13[i2] + wr21[i2] + wr33[i2];
		}
		else if (idy < pml && idx < nx && idz < nz) //forward
		{
			i3 = idz * nx * pml + idx * pml + idy;
			wf13[i3] = (1 + 2 * d_epsilon[i]) * S[i] * pow(d_vp[i] * dt, 2) * Uyy - (pow((d_dfb[pml - idy] * dt + 1), 2) - 3) * wf12[i3] + (2 * d_dfb[pml - idy] * dt - 1) * wf11[i3];
			pf3[i3] = (3 - pow((1 + dt * d_dfb[pml - idy]), 2)) * pf2[i3] + (2 * d_dfb[pml - idy] * dt - 1) * pf1[i3] + (1 + 2 * d_epsilon[i]) * S[i] * pow(d_vp[i] * dt, 2) * d_ddfb[pml - idy] * uy[i];
			wf21[i3] = dt * pf3[i3] + wf21[i3] * (1 - dt * d_dfb[pml - idy]);
			wf33[i3] = S[i] * pow(d_vp[i] * dt, 2) * (Uzz + (1 + 2 * d_epsilon[i]) * Uxx) + 2 * wf32[i3] - wf31[i3];
			u3[i] = wf13[i3] + wf21[i3] + wf33[i3];
		}
		else if (idy >= ny - pml && idy < ny && idx < nx && idz < nz) //backward
		{
			i4 = idz * nx * pml + idx * pml + idy - ny + pml;
			wba13[i4] = (1 + 2 * d_epsilon[i]) * S[i] * pow(d_vp[i] * dt, 2) * Uyy - (pow((d_dfb[idy - ny + pml] * dt + 1), 2) - 3) * wba12[i4] + (2 * d_dfb[idy - ny + pml] * dt - 1) * wba11[i4];
			pba3[i4] = (3 - pow((1 + dt * d_dfb[idy - ny + pml]), 2)) * pba2[i4] + (2 * d_dfb[idy - ny + pml] * dt - 1) * pba1[i4] - (1 + 2 * d_epsilon[i]) * S[i] * pow(d_vp[i] * dt, 2) * d_ddfb[idy - ny + pml] * uy[i];
			wba21[i4] = dt * pba3[i4] + wba21[i4] * (1 - dt * d_dfb[idy - ny + pml]);
			wba33[i4] = S[i] * pow(d_vp[i] * dt, 2) * (Uzz + (1 + 2 * d_epsilon[i]) * Uxx) + 2 * wba32[i4] - wba31[i4];
			u3[i] = wba13[i4] + wba21[i4] + wba33[i4];
		}
		else if (idz < pml && idx < nx && idy < ny) //top
		{
			i5 = idz * nx * ny + idx * ny + idy;
			wt13[i5] = S[i] * pow(d_vp[i] * dt, 2) * Uzz - (pow((d_dtb[pml - idz] * dt + 1), 2) - 3) * wt12[i5] + (2 * d_dtb[pml - idz] * dt - 1) * wt11[i5];
			pt3[i5] = (3 - pow((1 + dt * d_dtb[pml - idz]), 2)) * pt2[i5] + (2 * d_dtb[pml - idz] * dt - 1) * pt1[i5] + S[i] * pow(d_vp[i] * dt, 2) * d_ddtb[pml - idz] * uz[i];
			wt21[i5] = dt * pt3[i5] + wt21[i5] * (1 - dt * d_dtb[pml - idz]);
			wt33[i5] = (1 + 2 * d_epsilon[i]) * S[i] * pow(d_vp[i] * dt, 2) * (Uxx + Uyy) + 2 * wt32[i5] - wt31[i5];
			u3[i] = wt13[i5] + wt21[i5] + wt33[i5];
		}
	}
}
__global__ void addsource(float *d_source, float wavelet, float *u3, int nx, int ny, int nz)
{
	int idx = blockDim.y * blockIdx.y + threadIdx.y;
	int idy = blockDim.x * blockIdx.x + threadIdx.x;
	int idz = blockDim.z * blockIdx.z + threadIdx.z;
	if (idx < nx && idy < ny && idz < nz)
	{
		u3[idz * ny * nx + idx * ny + idy] += d_source[idz * ny * nx + idx * ny + idy] * wavelet;
	}
}

__global__ void exchange(int nx, int ny, int nz, int pml, float *u1, float *u2, float *u3,
												 float *wl11, float *wl12, float *wl13, float *wl31, float *wl32, float *wl33, float *pl1, float *pl2, float *pl3,
												 float *wr11, float *wr12, float *wr13, float *wr31, float *wr32, float *wr33, float *pr1, float *pr2, float *pr3,
												 float *wt11, float *wt12, float *wt13, float *wt31, float *wt32, float *wt33, float *pt1, float *pt2, float *pt3,
												 float *wb11, float *wb12, float *wb13, float *wb31, float *wb32, float *wb33, float *pb1, float *pb2, float *pb3,
												 float *wf11, float *wf12, float *wf13, float *wf31, float *wf32, float *wf33, float *pf1, float *pf2, float *pf3,
												 float *wba11, float *wba12, float *wba13, float *wba31, float *wba32, float *wba33, float *pba1, float *pba2, float *pba3)
{
	int idx = blockDim.y * blockIdx.y + threadIdx.y;
	int idy = blockDim.x * blockIdx.x + threadIdx.x;
	int idz = blockDim.z * blockIdx.z + threadIdx.z;
	int i, i1, i2, i3, i4, i5, i6;
	i = idz * ny * nx + idx * ny + idy;
	if (idx < nx && idy < ny && idz < nz)
	{
		u1[i] = u2[i];
		u2[i] = u3[i];
		//pml-domain time exchange
		if (idx < pml)
		{
			i1 = idz * ny * pml + idx * ny + idy;
			wl11[i1] = wl12[i1];
			wl12[i1] = wl13[i1];
			wl31[i1] = wl32[i1];
			wl32[i1] = wl33[i1];
			pl1[i1] = pl2[i1];
			pl2[i1] = pl3[i1];
		}
		if (idx >= nx - pml)
		{
			i2 = idz * ny * pml + (idx - nx + pml) * ny + idy;
			wr11[i2] = wr12[i2];
			wr12[i2] = wr13[i2];
			wr31[i2] = wr32[i2];
			wr32[i2] = wr33[i2];
			pr1[i2] = pr2[i2];
			pr2[i2] = pr3[i2];
		}
		if (idy < pml)
		{
			i3 = idz * nx * pml + idx * pml + idy;
			wf11[i3] = wf12[i3];
			wf12[i3] = wf13[i3];
			wf31[i3] = wf32[i3];
			wf32[i3] = wf33[i3];
			pf1[i3] = pf2[i3];
			pf2[i3] = pf3[i3];
		}
		if (idy >= ny - pml)
		{
			i4 = idz * nx * pml + idx * pml + idy - ny + pml;
			wba11[i4] = wba12[i4];
			wba12[i4] = wba13[i4];
			wba31[i4] = wba32[i4];
			wba32[i4] = wba33[i4];
			pba1[i4] = pba2[i4];
			pba2[i4] = pba3[i4];
		}
		if (idz < pml)
		{
			i5 = i;
			wt11[i5] = wt12[i5];
			wt12[i5] = wt13[i5];
			wt31[i5] = wt32[i5];
			wt32[i5] = wt33[i5];
			pt1[i5] = pt2[i5];
			pt2[i5] = pt3[i5];
		}
		if (idz >= nz - pml)
		{
			i6 = (idz - nz + pml) * ny * nx + idx * ny + idy;
			wb11[i6] = wb12[i6];
			wb12[i6] = wb13[i6];
			wb31[i6] = wb32[i6];
			wb32[i6] = wb33[i6];
			pb1[i6] = pb2[i6];
			pb2[i6] = pb3[i6];
		}
	}
}

__global__ void exchange1(int nx, int ny, int nz, int pml, float *u1, float *u2, float *u3,
													float *wl11, float *wl12, float *wl13, float *wl31, float *wl32, float *wl33, float *pl1, float *pl2, float *pl3,
													float *wr11, float *wr12, float *wr13, float *wr31, float *wr32, float *wr33, float *pr1, float *pr2, float *pr3,
													float *wt11, float *wt12, float *wt13, float *wt31, float *wt32, float *wt33, float *pt1, float *pt2, float *pt3,
													float *wf11, float *wf12, float *wf13, float *wf31, float *wf32, float *wf33, float *pf1, float *pf2, float *pf3,
													float *wba11, float *wba12, float *wba13, float *wba31, float *wba32, float *wba33, float *pba1, float *pba2, float *pba3)
{
	int idx = blockDim.y * blockIdx.y + threadIdx.y;
	int idy = blockDim.x * blockIdx.x + threadIdx.x;
	int idz = blockDim.z * blockIdx.z + threadIdx.z;
	int i, i1, i2, i3, i4, i5;
	i = idz * ny * nx + idx * ny + idy;
	if (idx < nx && idy < ny && idz < nz)
	{
		u1[i] = u2[i];
		u2[i] = u3[i];

		//pml-domain time exchange
		if (idx < pml)
		{
			i1 = idz * ny * pml + idx * ny + idy;
			wl11[i1] = wl12[i1];
			wl12[i1] = wl13[i1];
			wl31[i1] = wl32[i1];
			wl32[i1] = wl33[i1];
			pl1[i1] = pl2[i1];
			pl2[i1] = pl3[i1];
		}
		if (idx >= nx - pml)
		{
			i2 = idz * ny * pml + (idx - nx + pml) * ny + idy;
			wr11[i2] = wr12[i2];
			wr12[i2] = wr13[i2];
			wr31[i2] = wr32[i2];
			wr32[i2] = wr33[i2];
			pr1[i2] = pr2[i2];
			pr2[i2] = pr3[i2];
		}
		if (idy < pml)
		{
			i3 = idz * nx * pml + idx * pml + idy;
			wf11[i3] = wf12[i3];
			wf12[i3] = wf13[i3];
			wf31[i3] = wf32[i3];
			wf32[i3] = wf33[i3];
			pf1[i3] = pf2[i3];
			pf2[i3] = pf3[i3];
		}
		if (idy >= ny - pml)
		{
			i4 = idz * nx * pml + idx * pml + idy - ny + pml;
			wba11[i4] = wba12[i4];
			wba12[i4] = wba13[i4];
			wba31[i4] = wba32[i4];
			wba32[i4] = wba33[i4];
			pba1[i4] = pba2[i4];
			pba2[i4] = pba3[i4];
		}
		if (idz < pml)
		{
			i5 = i;
			wt11[i5] = wt12[i5];
			wt12[i5] = wt13[i5];
			wt31[i5] = wt32[i5];
			wt32[i5] = wt33[i5];
			pt1[i5] = pt2[i5];
			pt2[i5] = pt3[i5];
		}
	}
}

/*__global__ void exchange2(int nx, int ny, int nz, int pml, float *u1, float *u2, float *u3,   
												 float *wl11,float *wl12,float *wl13,float *wl31,float *wl32,float *wl33,float *pl1,float *pl2,float *pl3,
												 float *wr11,float *wr12,float *wr13,float *wr31,float *wr32,float *wr33,float *pr1,float *pr2,float *pr3,
												 float *wb11,float *wb12,float *wb13,float *wb31,float *wb32,float *wb33,float *pb1,float *pb2,float*pb3, 
												 float *wf11,float *wf12,float *wf13,float *wf31,float *wf32,float *wf33,float *pf1,float *pf2,float*pf3,
												 float *wba11,float *wba12,float *wba13,float *wba31,float *wba32,float *wba33,float *pba1,float *pba2,float*pba3)
{
			int idx =  blockDim.y * blockIdx.y + threadIdx.y;
			int idy =  blockDim.x * blockIdx.x + threadIdx.x;
			int idz =  blockDim.z * blockIdx.z + threadIdx.z;
			int i, i1, i2, i3, i4,i6;
			i = idz*ny*nx+idx*ny+idy;
    	if(idx<nx && idy<ny && idz<nz)
    {
		  		u1[i] = u2[i];
		  		u2[i] = u3[i];
				 //pml-domain time exchange
			   	if(idx<pml)
			   {
						i1 = idz*ny*pml+idx*ny+idy;
			      wl11[i1] = wl12[i1];
			      wl12[i1] = wl13[i1];
			      wl31[i1] = wl32[i1];
			      wl32[i1] = wl33[i1];
			      pl1[i1] =  pl2[i1];
			      pl2[i1] =  pl3[i1];
			   }
			   if(idx>=nx-pml)
			  {
						i2 = idz*ny*pml+(idx-nx+pml)*ny+idy;			      
						wr11[i2] = wr12[i2];
			      wr12[i2] = wr13[i2];
			      wr31[i2] = wr32[i2];
			      wr32[i2] = wr33[i2];
			      pr1[i2] =  pr2[i2];
			      pr2[i2] =  pr3[i2];
			  }
				 	if(idy<pml)
			   {
						i3 = idz*nx*pml+idx*pml+idy;			      
						wf11[i3] = wf12[i3];
			      wf12[i3] = wf13[i3];
			      wf31[i3] = wf32[i3];
			      wf32[i3] = wf33[i3];
			      pf1[i3] =  pf2[i3];
			      pf2[i3] =  pf3[i3];
			   }
			   if(idy>=ny-pml)
			  {
					i4 = idz*nx*pml+idx*pml+idy-ny+pml;				      
					wba11[i4] = wba12[i4];
			      	wba12[i4] = wba13[i4];
			      	wba31[i4] = wba32[i4];
			      	wba32[i4] = wba33[i4];
			      	pba1[i4] =  pba2[i4];
			      	pba2[i4] =  pba3[i4];
			  }
				if(idz>=nz-pml)
			{
				 	i6 = (idz-nz+pml)*ny*nx+idx*ny+idy;			   	  
				 	wb11[i6] = wb12[i6];
			   	  	wb12[i6] = wb13[i6];
			   	  	wb31[i6] = wb32[i6];
			   	  	wb32[i6] = wb33[i6];
			   	  	pb1[i6] =  pb2[i6];
			   	  	pb2[i6] =  pb3[i6];
			}
	 }
}
*/

__global__ void wavefield_output(float *u2, float *d_u, float *d_record, int nx, int ny, int nz, int sz, int pml)
{
	int idx = blockDim.y * blockIdx.y + threadIdx.y;
	int idy = blockDim.x * blockIdx.x + threadIdx.x;
	int idz = blockDim.z * blockIdx.z + threadIdx.z;
	if (idx >= pml && idx < nx - pml && idy >= pml && idy < ny - pml && idz >= pml && idz < nz - pml)
		d_u[(idz - pml) * (ny - 2 * pml) * (nx - 2 * pml) + (idx - pml) * (ny - 2 * pml) + idy - pml] = u2[idz * ny * nx + idx * ny + idy];
	if (idz == sz && idx >= pml && idx < nx - pml && idy >= pml && idy < ny - pml)
		d_record[(idx - pml) * (ny - 2 * pml) + idy - pml] = u2[idz * ny * nx + idx * ny + idy];
}

void checkCUDAerror(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err)
	{
		fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(-1);
	}
}

void CHECK(cudaError_t a)
{
	if (cudaSuccess != a)
	{
		fprintf(stderr, "Cuda runtime error in line %d of file %s: %s \n", __LINE__, __FILE__, cudaGetErrorString(cudaGetLastError()));
		exit(-1);
	}
}

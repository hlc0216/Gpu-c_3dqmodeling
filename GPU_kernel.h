
#include<cuda_runtime.h>
#define M 10
#define eps 2.22e-17

#ifndef GPU_KERNEL_H
#define GPU_KERNEL_H


//__device__ float absMaxval_d(float *a, int nx, int ny, int nz);

__global__ void grad(float *u2, float *ux, float *uy, float *uz, float *d_c, int nx, int ny, int nz, float dx, float dy, float dz);

__global__ void scalar_operator(float uxyzMax, float *ux, float *uy, float *uz, float *d_epsilon,float *d_delta, float *S, int nx, int ny, int nz);

__global__ void wavefield_update(float *d_c, float *d_c2, float *d_dlr, float *d_ddlr, float *d_dtb, float *d_ddtb, float *d_dfb, 
                                 float *d_ddfb, float *d_epsilon,float *d_delta,float *d_vp, float dx, float dy, float dz,  float dt,
                                 int nx, int ny, int nz, int pml,int sz,float *ux, float *uy, float *uz,float *u1,  float *u3,float *u2, float *S, 
																 float *wl11,float *wl12,float *wl13,float *wl21,float *wl31,float *wl32,float *wl33,float *pl1,float *pl2,float *pl3,
                                 float *wr11,float *wr12,float *wr13,float *wr21,float *wr31,float *wr32,float *wr33,float *pr1,float *pr2,float *pr3,
                                 float *wt11,float *wt12,float *wt13,float *wt21,float *wt31,float *wt32,float *wt33,float *pt1,float *pt2,float *pt3,
                                 float *wb11,float *wb12,float *wb13,float *wb21,float *wb31,float *wb32,float *wb33,float *pb1,float *pb2,float *pb3, 
                                 float *wf11,float *wf12,float *wf13,float *wf21,float *wf31,float *wf32,float *wf33,float *pf1,float *pf2,float *pf3,
                                 float *wba11,float *wba12,float *wba13,float *wba21,float *wba31,float *wba32,float *wba33,float *pba1,float *pba2,float*pba3);

__global__ void wavefield_update1(float*d_R, float *d_c, float *d_c2, float *d_dlr, float *d_ddlr, float *d_dtb, float *d_ddtb, float *d_dfb, 
                                 float *d_ddfb, float *d_epsilon,float *d_delta,float *d_vp, float dx, float dy, float dz,  float dt,
                                 int nx, int ny, int nz, int pml,float *ux, float *uy, float *uz,float *u1,  float *u3,float *u2, float *S, 
								float *wl11,float *wl12,float *wl13,float *wl21,float *wl31,float *wl32,float *wl33,float *pl1,float *pl2,float *pl3,
                                 float *wr11,float *wr12,float *wr13,float *wr21,float *wr31,float *wr32,float *wr33,float *pr1,float *pr2,float *pr3,
                                 float *wt11,float *wt12,float *wt13,float *wt21,float *wt31,float *wt32,float *wt33,float *pt1,float *pt2,float *pt3,
                                 float *wf11,float *wf12,float *wf13,float *wf21,float *wf31,float *wf32,float *wf33,float *pf1,float *pf2,float*pf3,
                                 float *wba11,float *wba12,float *wba13,float *wba21,float *wba31,float *wba32,float *wba33,float *pba1,float *pba2,float*pba3);

__global__ void exchange(int nx, int ny, int nz, int pml, float *u1, float *u2, float *u3,   
												 float *wl11,float *wl12,float *wl13,float *wl31,float *wl32,float *wl33,float *pl1,float *pl2,float *pl3,
												 float *wr11,float *wr12,float *wr13,float *wr31,float *wr32,float *wr33,float *pr1,float *pr2,float *pr3,
												 float *wt11,float *wt12,float *wt13,float *wt31,float *wt32,float *wt33,float *pt1,float *pt2,float *pt3,
												 float *wb11,float *wb12,float *wb13,float *wb31,float *wb32,float *wb33,float *pb1,float *pb2,float *pb3, 
												 float *wf11,float *wf12,float *wf13,float *wf31,float *wf32,float *wf33,float *pf1,float *pf2,float *pf3,
												 float *wba11,float *wba12,float *wba13,float *wba31,float *wba32,float *wba33,float *pba1,float *pba2,float *pba3);

__global__ void addsource(float *d_source, float wavelet, float *u3, int nx, int ny, int nz);

__global__ void wavefield_output(float *u2, float *d_u, float *d_record, int nx, int ny, int nz, int sz, int pml);

__global__ void grad1(float *R, float *u2, float *ux, float *uy, float *uz, float *d_c, int nx, int ny, int nz, float dx, float dy, float dz);

//__global__ void grad2(float *R, float *u2, float *ux, float *uy, float *uz, float *d_c, int nx, int ny, int nz, float dx, float dy, float dz);

__global__ void scalar_operator1(float uxyzMax, float *ux, float *uy, float *uz, float *d_epsilon,float *d_delta, float *S, int nx, int ny, int nz);

/*__global__ void u_update_middle1(float *d_c2, float *d_epsilon,float *d_delta,float *d_vp, float dx, float dy, float dz,  float dt,
                                int nx, int ny, int nz, int pml, float *R, float *u1,  float *u2,float *u3, float *S);
																
__global__ void u_update_middle2(float *d_c2, float *d_epsilon,float *d_delta,float *d_vp, float dx, float dy, float dz,  float dt,
                                int nx, int ny, int nz, int pml, float *R, float *u1,  float *u2,float *u3, float *S);

__global__ void u_update_lpml(float *d_c2, float *d_dlr, float *d_ddlr, float *d_epsilon,float *d_delta, float *d_vp, float dx, 
															float dy,float dz, float dt, int nx, int ny, int nz, int pml, float *R, float *ux,float *u3,float *u2, float *S,
															float *wl11,float *wl12,float *wl13,float *wl21,float *wl31,float *wl32,float *wl33,float *pl1,float *pl2,float *pl3);
															
__global__ void u_update_lpml2(float *d_c2, float *d_dlr, float *d_ddlr, float *d_epsilon,float *d_delta, float *d_vp, float dx, 
															float dy,float dz, float dt, int nx, int ny, int nz, int pml, float *R, float *ux,float *u3,float *u2, float *S,
															float *wl11,float *wl12,float *wl13,float *wl21,float *wl31,float *wl32,float *wl33,float *pl1,float *pl2,float *pl3);	

__global__ void u_update_rpml(float *d_c2, float *d_dlr, float *d_ddlr, float *d_epsilon,float *d_delta, float *d_vp, float dx, 
															float dy,float dz, float dt, int nx, int ny, int nz, int pml, float *R,float *ux,float *u3,float *u2, float *S, 
															float *wr11,float *wr12,float *wr13,float *wr21,float *wr31,float *wr32,float *wr33,float *pr1,float *pr2,float *pr3);

__global__ void u_update_rpml2(float *d_c2, float *d_dlr, float *d_ddlr, float *d_epsilon,float *d_delta, float *d_vp, float dx, 
															float dy,float dz, float dt, int nx, int ny, int nz, int pml, float *R,float *ux,float *u3,float *u2, float *S, 
															float *wr11,float *wr12,float *wr13,float *wr21,float *wr31,float *wr32,float *wr33,float *pr1,float *pr2,float *pr3);

__global__ void u_update_tpml(float *d_c2, float *d_dtb, float *d_ddtb, float *d_epsilon,float *d_delta, float *d_vp, float dx, 
															float dy,float dz, float dt, int nx, int ny, int nz, int pml, float *uz,float *u3,float *u2, float *S,
															float *wt11,float *wt12,float *wt13,float *wt21,float *wt31,float *wt32,float *wt33,float *pt1,float *pt2,float *pt3);

__global__ void u_update_bpml(float *R, float *d_c2, float *d_dtb, float *d_ddtb, float *d_epsilon,float *d_delta, float *d_vp, float dx, 
															float dy,float dz, float dt, int nx, int ny, int nz, int pml, float *uz,float *u3,float *u2, float *S,
															 float *wb11,float *wb12,float *wb13,float *wb21,float *wb31,float *wb32,float *wb33,float *pb1,float *pb2,float*pb3);

__global__ void u_update_fpml(float *d_c2, float *d_dfb, float *d_ddfb, float *d_epsilon,float *d_delta, float *d_vp, float dx, 
															float dy,float dz, float dt, int nx, int ny, int nz, int pml, float *R, float *uy,float *u3,float *u2, float *S,
															float *wf11,float *wf12,float *wf13,float *wf21,float *wf31,float *wf32,float *wf33,float *pf1,float *pf2,float*pf3);

__global__ void u_update_fpml2(float *d_c2, float *d_dfb, float *d_ddfb, float *d_epsilon,float *d_delta, float *d_vp, float dx, 
															float dy,float dz, float dt, int nx, int ny, int nz, int pml, float *R, float *uy,float *u3,float *u2, float *S,
															float *wf11,float *wf12,float *wf13,float *wf21,float *wf31,float *wf32,float *wf33,float *pf1,float *pf2,float*pf3);

__global__ void u_update_bapml(float *d_c2, float *d_dfb, float *d_ddfb, float *d_epsilon,float *d_delta, float *d_vp, float dx, 
															 float dy,float dz, float dt, int nx, int ny, int nz, int pml, float *R, float *uy,float *u3,float *u2, float *S,
															 float *wba11,float *wba12,float *wba13,float *wba21,float *wba31,float *wba32,float *wba33,float *pba1,float *pba2,float*pba3);

__global__ void u_update_bapml2(float *d_c2, float *d_dfb, float *d_ddfb, float *d_epsilon,float *d_delta, float *d_vp, float dx, 
															 float dy,float dz, float dt, int nx, int ny, int nz, int pml, float *R, float *uy,float *u3,float *u2, float *S,
															 float *wba11,float *wba12,float *wba13,float *wba21,float *wba31,float *wba32,float *wba33,float *pba1,float *pba2,float*pba3);
*/
__global__ void exchange1(int nx, int ny, int nz, int pml, float *u1, float *u2, float *u3,   
												 float *wl11,float *wl12,float *wl13,float *wl31,float *wl32,float *wl33,float *pl1,float *pl2,float *pl3,
												 float *wr11,float *wr12,float *wr13,float *wr31,float *wr32,float *wr33,float *pr1,float *pr2,float *pr3,
												 float *wt11,float *wt12,float *wt13,float *wt31,float *wt32,float *wt33,float *pt1,float *pt2,float *pt3,
												 float *wf11,float *wf12,float *wf13,float *wf31,float *wf32,float *wf33,float *pf1,float *pf2,float*pf3,
												 float *wba11,float *wba12,float *wba13,float *wba31,float *wba32,float *wba33,float *pba1,float *pba2,float*pba3);

/*__global__ void exchange2(int nx, int ny, int nz, int pml, float *u1, float *u2, float *u3,   
												 float *wl11,float *wl12,float *wl13,float *wl31,float *wl32,float *wl33,float *pl1,float *pl2,float *pl3,
												 float *wr11,float *wr12,float *wr13,float *wr31,float *wr32,float *wr33,float *pr1,float *pr2,float *pr3,
												 float *wb11,float *wb12,float *wb13,float *wb31,float *wb32,float *wb33,float *pb1,float *pb2,float*pb3, 
												 float *wf11,float *wf12,float *wf13,float *wf31,float *wf32,float *wf33,float *pf1,float *pf2,float*pf3,
												 float *wba11,float *wba12,float *wba13,float *wba31,float *wba32,float *wba33,float *pba1,float *pba2,float*pba3);
*/
												 
void checkCUDAerror(const char *msg);
void CHECK(cudaError_t a);
#endif

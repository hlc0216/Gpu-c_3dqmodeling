#ifndef CPU_FUNCTION_H
#define CPU_FUNCTION_H

float absMaxval(float *a, int nx, int ny, int nz);

float absMaxval2d(float *a, int nx, int ny);

void Source(int sx, int sy, int sz, float fm, float amp, float alpha, float dt, float dx, float dy, 
            float dz, float t0, int nt, int nx, int ny, int nz, float *source, float *wavelet);
						
void extend_model(float *in, float *out, int nx, int ny, int nz, int pml);

void fdcoff1(float *c);

void fdcoff2(float *c);

void pmlcoff(int pml, float vpmax, float dx, float dy, float dz, float *dlr, 
             float *ddlr, float *dtb, float *ddtb, float *dfb, float *ddfb);
             
//void readmodel3d(const char *filename,float *a,int ny,int nx,int nz,int fny,int fnx,int fnz,int vny,int vnx,int vnz,int nbd);

void readmodel3d(const char *filename,float *a,int ny,int nx,int nz,int fny,int fnx,int fnz,int vny,int vnx,int vnz) ;

void trans_z(float *in, int nx, int ny, int nz);

void trans2d(float *in, float *out, int nx, int nz);

void trans3d(float *in, float *out, int nx, int ny, int nz);
//hlc 添加的函数
// void grad(float *u2, float *ux, float *uy, float *uz, float *d_c,  int nx, int ny, int nz, float dx, float dy, float dz);
// void addsource(float *d_source, float wavelet, float *u3, int nx, int ny, int nz);
// void scalar_operator(float uxyzMax, float *ux, float *uy, float *uz, float *d_epsilon,float *d_delta, float *S, int nx, int ny, int nz);
// void wavefield_update1(float *d_R, float *d_c, float *d_c2, float *d_dlr, float *d_ddlr, float *d_dtb, float *d_ddtb, float *d_dfb, 
//                                  float *d_ddfb, float *d_epsilon,float *d_delta,float *d_vp, float dx, float dy, float dz,  float dt,
//                                  int nx, int ny, int nz, int pml,float *ux, float *uy, float *uz,float *u1,  float *u3,float *u2, float *S, 
// 								float *wl11,float *wl12,float *wl13,float *wl21,float *wl31,float *wl32,float *wl33,float *pl1,float *pl2,float *pl3,
//                                  float *wr11,float *wr12,float *wr13,float *wr21,float *wr31,float *wr32,float *wr33,float *pr1,float *pr2,float *pr3,
//                                  float *wt11,float *wt12,float *wt13,float *wt21,float *wt31,float *wt32,float *wt33,float *pt1,float *pt2,float *pt3,
//                                  float *wf11,float *wf12,float *wf13,float *wf21,float *wf31,float *wf32,float *wf33,float *pf1,float *pf2,float*pf3,
//                                  float *wba11,float *wba12,float *wba13,float *wba21,float *wba31,float *wba32,float *wba33,float *pba1,float *pba2,float*pba3);
// void exchange1(int nx, int ny, int nz, int pml, float *u1, float *u2, float *u3,
// 					float *wl11, float *wl12, float *wl13, float *wl31, float *wl32, float *wl33, float *pl1, float *pl2, float *pl3,
// 					float *wr11, float *wr12, float *wr13, float *wr31, float *wr32, float *wr33, float *pr1, float *pr2, float *pr3,
// 					float *wt11, float *wt12, float *wt13, float *wt31, float *wt32, float *wt33, float *pt1, float *pt2, float *pt3,
// 					float *wf11, float *wf12, float *wf13, float *wf31, float *wf32, float *wf33, float *pf1, float *pf2, float *pf3,
// 					float *wba11, float *wba12, float *wba13, float *wba31, float *wba32, float *wba33, float *pba1, float *pba2, float *pba3);
#endif

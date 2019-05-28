/************ The program is writed by Lun Ruan, 2018.10***********************/
/*******3D Modeling for pure qP wave equation from Xu,2015************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <malloc.h>
#include <cuda.h>
//#include <cuda_runtime.h>
//#include <device_functions.h>

#include "array_new.h"
#include "read_write.h"
#include "GPU_kernel.h"
#include "CPU_function.h"

#define M 10
#define eps  2.22e-17
#define Block_Sizex 8
#define Block_Sizey 8
#define Block_Sizez 8


void modeling3d(int nx, int ny, int nz, int nt, int ntsnap, float dx, float dy, float dz, float dt, int pml, int snapflag, int sx, int sy, int sz, 
                float *vp, float *epsilon, float *delta, float *source, float *wavelet, float *record, float *dlr,float *ddlr, float *dtb, float *ddtb, 
				float *dfb, float *ddfb, float *c, float *c2, const char *snap_file)							
{
		//time-assuming
		clock_t starttime, endtime;
		float timespent;
		starttime = clock();

		int device_num;
		cudaGetDeviceCount(&device_num);
		if(device_num > 0)
			cudaSetDevice(0);
		else
			cudaSetDevice(0);

		float uxMax, uyMax, uzMax, uxyzMax;
		int i,j,l,k;
		char snapname[100], snapname_S[100], snapxzname[100], snapyzname[100],snapxyname[100],
			snapSxzname[100], snapSyzname[100],snapSxyname[100];
		
		dim3 grid((ny+Block_Sizey-1)/Block_Sizey, (nx+Block_Sizex-1)/Block_Sizex, (nz+Block_Sizez-1)/Block_Sizez);
		dim3 block(Block_Sizey, Block_Sizex, Block_Sizez);
		
		//allocate host memory
		float	*snap = array1d((nx-2*pml)*(ny-2*pml)*(nz-2*pml)), *snapxz = array1d((nx-2*pml)*(nz-2*pml)), 
				*snapyz = array1d((ny-2*pml)*(nz-2*pml)), *snapxy = array1d((ny-2*pml)*(nx-2*pml)),
				*snapS = array1d(nx*ny*nz), *snapSxz = array1d(nx*nz), *snapSyz = array1d(ny*nz), *snapSxy = array1d(ny*nx),
				*h_ux = array1d(nx*ny*nz), *h_uy = array1d(nx*ny*nz), *h_uz = array1d(nx*ny*nz),*h_u2 = array1d(nx*ny*nz);
		
		/******* allocate device memory *****/
		float	*d_vp, *d_epsilon,*d_delta,*d_c,*d_c2,*d_dlr,*d_ddlr,*d_dtb,*d_ddtb,*d_dfb,
				*d_ddfb, *d_source, *S, *u1, *u2, *u3, *ux, *uy, *uz,*d_record, *d_u,				
				*wl11, *wl12, *wl13, *wl21, *wl31, *wl32, *wl33, *pl1,*pl2,*pl3,
				*wr11, *wr12, *wr13, *wr21, *wr31, *wr32, *wr33, *pr1,*pr2,*pr3,
				*wt11, *wt12, *wt13, *wt21, *wt31, *wt32, *wt33, *pt1,*pt2,*pt3,
				*wb11, *wb12, *wb13, *wb21, *wb31, *wb32, *wb33, *pb1,*pb2,*pb3,
				*wf11, *wf12, *wf13, *wf21, *wf31, *wf32, *wf33, *pf1,*pf2,*pf3,
				*wba11, *wba12, *wba13, *wba21, *wba31, *wba32, *wba33, *pba1,*pba2,*pba3;

		/*打印cpu参数*/
		for(i=0;i<nt;i++){
			printf("wavelet=%4.3f ",wavelet[i]);
		}
		printf("\n");
		for(i=0;i<pml;i++){
			printf("dlr=%4.3f ",dlr[i]);
		}
		printf("\n");
		for(i=0;i<pml;i++){
			printf("ddlr=%4.3f ",ddlr[i]);
		}
		printf("\n");
		for(i=0;i<pml;i++){
			printf("dtb=%4.3f ",dtb[i]);
		}
		printf("\n");
		for(i=0;i<pml;i++){
			printf("ddtb=%4.3f ",ddtb[i]);
		}
		printf("\n");
		for(i=0;i<pml;i++){
			printf("dfb=%4.3f ",dfb[i]);
		}
		printf("\n");
		for(i=0;i<pml;i++){
			printf("ddfb=%4.3f ",ddfb[i]);
		}
		printf("\n");
		for(i=(M/2)*(M/2+1)+1;i<(M/2)*(M/2+1)+6;i++){
			printf("c2=%4.3f ",c2[i]);
		}
		printf("\n");
		
		cudaMalloc(&d_vp, nx*ny*nz*sizeof(float));
		cudaMalloc(&d_epsilon, nx*ny*nz*sizeof(float));
		cudaMalloc(&d_delta, nx*ny*nz*sizeof(float));
		cudaMalloc(&d_c, (M/2+1)*(M/2+1)*sizeof(float));
		cudaMalloc(&d_c2, (M/2+1)*(M/2+1)*sizeof(float));
		cudaMalloc(&d_dlr,pml*sizeof(float));
		cudaMalloc(&d_ddlr, pml*sizeof(float));
		cudaMalloc(&d_dtb, pml*sizeof(float));
		cudaMalloc(&d_ddtb, pml*sizeof(float));
		cudaMalloc(&d_dfb, pml*sizeof(float));
		cudaMalloc(&d_ddfb, pml*sizeof(float));
		cudaMalloc(&d_source, nx*ny*nz*sizeof(float));
		cudaMalloc(&S, nx*ny*nz*sizeof(float));
		cudaMalloc(&u1, nx*ny*nz*sizeof(float));
		cudaMalloc(&u2, nx*ny*nz*sizeof(float));
		cudaMalloc(&u3, nx*ny*nz*sizeof(float));
		cudaMalloc(&ux, nx*ny*nz*sizeof(float));
		cudaMalloc(&uy, nx*ny*nz*sizeof(float));
		cudaMalloc(&uz, nx*ny*nz*sizeof(float));
	//	cudaMalloc(&d_record, (nx-2*pml)*(ny-2*pml)*nt*sizeof(float));
	//	cudaMalloc(&d_u, (nx-2*pml)*(ny-2*pml)*(nz-2*pml)*sizeof(float));
	
		cudaMalloc(&wr11, pml*ny*nz*sizeof(float));
		cudaMalloc(&wr12, pml*ny*nz*sizeof(float));
		cudaMalloc(&wr13, pml*ny*nz*sizeof(float));
		cudaMalloc(&wr21, pml*ny*nz*sizeof(float));
		cudaMalloc(&wr31, pml*ny*nz*sizeof(float));
		cudaMalloc(&wr32, pml*ny*nz*sizeof(float));
		cudaMalloc(&wr33, pml*ny*nz*sizeof(float));
		cudaMalloc(&pr1, pml*ny*nz*sizeof(float));
		cudaMalloc(&pr2, pml*ny*nz*sizeof(float));
		cudaMalloc(&pr3, pml*ny*nz*sizeof(float));

		cudaMalloc(&wl11, pml*ny*nz*sizeof(float));
		cudaMalloc(&wl12, pml*ny*nz*sizeof(float));
		cudaMalloc(&wl13, pml*ny*nz*sizeof(float));
		cudaMalloc(&wl21, pml*ny*nz*sizeof(float));
		cudaMalloc(&wl31, pml*ny*nz*sizeof(float));
		cudaMalloc(&wl32, pml*ny*nz*sizeof(float));
		cudaMalloc(&wl33, pml*ny*nz*sizeof(float));
		cudaMalloc(&pl1, pml*ny*nz*sizeof(float));
		cudaMalloc(&pl2, pml*ny*nz*sizeof(float));
		cudaMalloc(&pl3, pml*ny*nz*sizeof(float));

		cudaMalloc(&wt11, pml*nx*ny*sizeof(float));
		cudaMalloc(&wt12, pml*nx*ny*sizeof(float));
		cudaMalloc(&wt13, pml*nx*ny*sizeof(float));
		cudaMalloc(&wt21, pml*nx*ny*sizeof(float));
		cudaMalloc(&wt31, pml*nx*ny*sizeof(float));
		cudaMalloc(&wt32, pml*nx*ny*sizeof(float));
		cudaMalloc(&wt33, pml*nx*ny*sizeof(float));
		cudaMalloc(&pt1, pml*nx*ny*sizeof(float));
		cudaMalloc(&pt2, pml*nx*ny*sizeof(float));
		cudaMalloc(&pt3, pml*nx*ny*sizeof(float));

		cudaMalloc(&wb11, pml*nx*ny*sizeof(float));
		cudaMalloc(&wb12, pml*nx*ny*sizeof(float));
		cudaMalloc(&wb13, pml*nx*ny*sizeof(float));
		cudaMalloc(&wb21, pml*nx*ny*sizeof(float));
		cudaMalloc(&wb31, pml*nx*ny*sizeof(float));
		cudaMalloc(&wb32, pml*nx*ny*sizeof(float));
		cudaMalloc(&wb33, pml*nx*ny*sizeof(float));
		cudaMalloc(&pb1, pml*nx*ny*sizeof(float));
		cudaMalloc(&pb2, pml*nx*ny*sizeof(float));
		cudaMalloc(&pb3, pml*nx*ny*sizeof(float));
		
		cudaMalloc(&wf11, pml*nx*nz*sizeof(float));
		cudaMalloc(&wf12, pml*nx*nz*sizeof(float));
		cudaMalloc(&wf13, pml*nx*nz*sizeof(float));
		cudaMalloc(&wf21, pml*nx*nz*sizeof(float));
		cudaMalloc(&wf31, pml*nx*nz*sizeof(float));
		cudaMalloc(&wf32, pml*nx*nz*sizeof(float));
		cudaMalloc(&wf33, pml*nx*nz*sizeof(float));
		cudaMalloc(&pf1, pml*nx*nz*sizeof(float));
		cudaMalloc(&pf2, pml*nx*nz*sizeof(float));
		cudaMalloc(&pf3, pml*nx*nz*sizeof(float));

		cudaMalloc(&wba11, pml*nx*nz*sizeof(float));
		cudaMalloc(&wba12, pml*nx*nz*sizeof(float));
		cudaMalloc(&wba13, pml*nx*nz*sizeof(float));
		cudaMalloc(&wba21, pml*nx*nz*sizeof(float));
		cudaMalloc(&wba31, pml*nx*nz*sizeof(float));
		cudaMalloc(&wba32, pml*nx*nz*sizeof(float));
		cudaMalloc(&wba33, pml*nx*nz*sizeof(float));
		cudaMalloc(&pba1, pml*nx*nz*sizeof(float));
		cudaMalloc(&pba2, pml*nx*nz*sizeof(float));
		cudaMalloc(&pba3, pml*nx*nz*sizeof(float));
		
		
	//intialized memory
	/*	cudaMemset(S, 0, nx*ny*nz*sizeof(float));
		cudaMemset(u1, 0, nx*ny*nz*sizeof(float));
		cudaMemset(u2, 0, nx*ny*nz*sizeof(float));
		cudaMemset(u3, 0, nx*ny*nz*sizeof(float));
		cudaMemset(ux, 0, nx*ny*nz*sizeof(float));
		cudaMemset(uy, 0, nx*ny*nz*sizeof(float));
		cudaMemset(uz, 0, nx*ny*nz*sizeof(float));

		cudaMemset(wr11, 0, pml*ny*nz*sizeof(float));
		cudaMemset(wr12, 0, pml*ny*nz*sizeof(float));
		cudaMemset(wr13, 0, pml*ny*nz*sizeof(float));
		cudaMemset(wr21, 0, pml*ny*nz*sizeof(float));
		cudaMemset(wr31, 0, pml*ny*nz*sizeof(float));
		cudaMemset(wr32, 0, pml*ny*nz*sizeof(float));
		cudaMemset(wr33, 0, pml*ny*nz*sizeof(float));
		cudaMemset(pr1, 0, pml*ny*nz*sizeof(float));
		cudaMemset(pr2, 0, pml*ny*nz*sizeof(float));
		cudaMemset(pr3, 0, pml*ny*nz*sizeof(float));

		cudaMemset(wl11, 0, pml*ny*nz*sizeof(float));
		cudaMemset(wl12, 0, pml*ny*nz*sizeof(float));
		cudaMemset(wl13, 0, pml*ny*nz*sizeof(float));
		cudaMemset(wl21, 0, pml*ny*nz*sizeof(float));
		cudaMemset(wl31, 0, pml*ny*nz*sizeof(float));
		cudaMemset(wl32, 0, pml*ny*nz*sizeof(float));
		cudaMemset(wl33, 0, pml*ny*nz*sizeof(float));
		cudaMemset(pl1, 0, pml*ny*nz*sizeof(float));
		cudaMemset(pl2, 0, pml*ny*nz*sizeof(float));
		cudaMemset(pl3, 0, pml*ny*nz*sizeof(float));

		cudaMemset(wt11, 0, pml*nx*ny*sizeof(float));
		cudaMemset(wt12, 0, pml*nx*ny*sizeof(float));
		cudaMemset(wt13, 0, pml*nx*ny*sizeof(float));
		cudaMemset(wt21, 0, pml*nx*ny*sizeof(float));
		cudaMemset(wt31, 0, pml*nx*ny*sizeof(float));
		cudaMemset(wt32, 0, pml*nx*ny*sizeof(float));
		cudaMemset(wt33, 0, pml*nx*ny*sizeof(float));
		cudaMemset(pt1, 0, pml*nx*ny*sizeof(float));
		cudaMemset(pt2, 0, pml*nx*ny*sizeof(float));
		cudaMemset(pt3, 0, pml*nx*ny*sizeof(float));

		cudaMemset(wb11, 0, pml*nx*ny*sizeof(float));
		cudaMemset(wb12, 0, pml*nx*ny*sizeof(float));
		cudaMemset(wb13, 0, pml*nx*ny*sizeof(float));
		cudaMemset(wb21, 0, pml*nx*ny*sizeof(float));
		cudaMemset(wb31, 0, pml*nx*ny*sizeof(float));
		cudaMemset(wb32, 0, pml*nx*ny*sizeof(float));
		cudaMemset(wb33, 0, pml*nx*ny*sizeof(float));
		cudaMemset(pb1, 0, pml*nx*ny*sizeof(float));
		cudaMemset(pb2, 0, pml*nx*ny*sizeof(float));
		cudaMemset(pb3, 0, pml*nx*ny*sizeof(float));
		
		cudaMemset(wf11, 0, pml*nx*nz*sizeof(float));
		cudaMemset(wf12, 0, pml*nx*nz*sizeof(float));
		cudaMemset(wf13, 0, pml*nx*nz*sizeof(float));
		cudaMemset(wf21, 0, pml*nx*nz*sizeof(float));
		cudaMemset(wf31, 0, pml*nx*nz*sizeof(float));
		cudaMemset(wf32, 0, pml*nx*nz*sizeof(float));
		cudaMemset(wf33, 0, pml*nx*nz*sizeof(float));
		cudaMemset(pf1, 0, pml*nx*nz*sizeof(float));
		cudaMemset(pf2, 0, pml*nx*nz*sizeof(float));
		cudaMemset(pf3, 0, pml*nx*nz*sizeof(float));

		cudaMemset(wba11, 0, pml*nx*nz*sizeof(float));
		cudaMemset(wba12, 0, pml*nx*nz*sizeof(float));
		cudaMemset(wba13, 0, pml*nx*nz*sizeof(float));
		cudaMemset(wba21, 0, pml*nx*nz*sizeof(float));
		cudaMemset(wba31, 0, pml*nx*nz*sizeof(float));
		cudaMemset(wba32, 0, pml*nx*nz*sizeof(float));
		cudaMemset(wba33, 0, pml*nx*nz*sizeof(float));
		cudaMemset(pba1, 0, pml*nx*nz*sizeof(float));
		cudaMemset(pba2, 0, pml*nx*nz*sizeof(float));
		cudaMemset(pba3, 0, pml*nx*nz*sizeof(float));
	*/	
		cudaMemcpy(d_vp, vp, nx*ny*nz*sizeof(float), cudaMemcpyHostToDevice);	
		cudaMemcpy(d_epsilon, epsilon,  nx*ny*nz*sizeof(float), cudaMemcpyHostToDevice);		
		cudaMemcpy(d_delta, delta, nx*ny*nz*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_c, c, (M/2+1)*(M/2+1)*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_c2, c2, (M/2+1)*(M/2+1)*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_source, source, nx*ny*nz*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_dlr, dlr,  pml*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_ddlr, ddlr,  pml*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_dtb, dtb,   pml*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_ddtb, ddtb,  pml*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_dfb, dfb,   pml*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_ddfb, ddfb,  pml*sizeof(float), cudaMemcpyHostToDevice);
		
		for(k=0;k<nt;k++)
   		{
  			if(k%100==0)
				printf("nt = %d\n",k);
				
			grad<<<grid,block>>>(u2, ux, uy, uz, d_c, nx, ny, nz, dx, dy, dz);
			
			cudaMemcpy(h_ux, ux, nx*ny*nz*sizeof(float), cudaMemcpyDeviceToHost);
			cudaMemcpy(h_uy, uy, nx*ny*nz*sizeof(float), cudaMemcpyDeviceToHost);
			cudaMemcpy(h_uz, uz, nx*ny*nz*sizeof(float), cudaMemcpyDeviceToHost);

			uxMax = absMaxval(h_ux, nx, ny, nz);
			uyMax = absMaxval(h_uy, nx, ny, nz);
			uzMax = absMaxval(h_uz, nx, ny, nz);
			uxyzMax = max(uxMax, uyMax);
			uxyzMax = max(uxyzMax, uzMax);
			//打印uxyzMax
			printf("uxyzMax=%4.3f\n",uxyzMax);
			//calculating S operators
  	  		scalar_operator<<<grid,block>>>(uxyzMax, ux, uy, uz, d_epsilon, d_delta, S, nx, ny, nz);
  	 	    //打印S GPU端(见GPU_kernel.cu)
  	  		//calculating wavefield using FD method
  	  		wavefield_update<<<grid,block>>>(d_c, d_c2, d_dlr, d_ddlr, d_dtb, d_ddtb, d_dfb, d_ddfb, d_epsilon,d_delta,
  	                                    	d_vp, dx, dy, dz, dt, nx, ny, nz, pml, sz, ux, uy, uz, u1, u3, u2, S,
											wl11, wl12, wl13, wl21, wl31, wl32, wl33, pl1, pl2, pl3,
											wr11, wr12, wr13, wr21, wr31, wr32, wr33, pr1, pr2, pr3,
											wt11, wt12, wt13, wt21, wt31, wt32, wt33, pt1, pt2, pt3,
											wb11, wb12, wb13, wb21, wb31, wb32, wb33, pb1, pb2, pb3,
											wf11, wf12, wf13, wf21, wf31, wf32, wf33, pf1, pf2, pf3,
											wba11, wba12, wba13, wba21, wba31, wba32, wba33, pba1, pba2, pba3);
  	  		addsource<<<grid,block>>>(d_source, wavelet[k], u3, nx, ny, nz);
			
  	 		exchange<<<grid,block>>>(nx, ny, nz, pml, u1, u2, u3, 
									wl11, wl12, wl13, wl31, wl32, wl33, pl1, pl2, pl3,
									wr11, wr12, wr13, wr31, wr32, wr33, pr1, pr2, pr3,
									wt11, wt12, wt13, wt31, wt32, wt33, pt1, pt2, pt3,
									wb11, wb12, wb13, wb31, wb32, wb33, pb1, pb2, pb3,
									wf11, wf12, wf13, wf31, wf32, wf33, pf1, pf2, pf3,
									wba11, wba12, wba13, wba31, wba32, wba33, pba1, pba2, pba3);
		 	
     		// seismic fullwavefield and record
            //wavefield_output<<<grid,block>>>(u2, d_u, &d_record[k*(nx-2*pml)*(ny-2*pml)], nx, ny, nz, sz, pml);	
	 		
			 cudaMemcpy(h_u2, u2, nx*ny*nz*sizeof(float), cudaMemcpyDeviceToHost);
			//打印 h_u2
			// for(int i=0;i<nz;i++){
			// 	for(int j=0;j<nx;j++){
			// 		for(int k=0;k<ny;k++){
			// 			if(h_u2[i*nx*ny+j*ny+k]>0.0001 || h_u2[i*nx*ny+j*ny+k]<-0.0001)
			// 				printf("h_u2[xxx]=%4.3f ",h_u2[i*nx*ny+j*ny+k]);
			// 		}
					
			// 	}
				
			// }

	 		for(i=pml;i<nx-pml;i++)
	 			for(j=pml;j<ny-pml;j++)
	 			{
	 				record[k*(nx-2*pml)*(ny-2*pml)+(i-pml)*(ny-2*pml)+j-pml] = h_u2[sz*nx*ny+i*ny+j];
	 			}
	 	
		
      		if(snapflag ==1 && k%ntsnap==0)
     		{
				sprintf(snapname,"%s%d.dat", snap_file, k);
				sprintf(snapxzname,"%s_xz%d.dat", snap_file, k);
				sprintf(snapyzname,"%s_yz%d.dat", snap_file, k);
				sprintf(snapxyname,"%s_xy%d.dat", snap_file, k);
    //  	 	cudaMemcpy(snap, d_u, (nz-2*pml)*(nx-2*pml)*(ny-2*pml)*sizeof(float), cudaMemcpyDeviceToHost);
    		 	for(i=pml;i<nz-pml;i++)
    		 		for(j=pml;j<nx-pml;j++)
	 			    	for(l=pml;l<ny-pml;l++)
	 			   				snap[(i-pml)*(nx-2*pml)*(ny-2*pml)+(j-pml)*(ny-2*pml)+l-pml] = h_u2[i*nx*ny+j*ny+l]; 
	 				 
    		
      	 		writefile_3d(snapname, snap, nz-2*pml, nx-2*pml, ny-2*pml);
				for(i=0;i<nz-2*pml;i++)
					for(j=0;j<nx-2*pml;j++)
						for(l=0;l<ny-2*pml;l++)
						{
									if(l==(ny-2*pml-1)/2)
									{
											snapxz[i*(nx-2*pml)+j] = snap[i*(nx-2*pml)*(ny-2*pml)+j*(ny-2*pml)+l];										
									}
									
									if(j==(nx-2*pml-1)/2)
									{
											snapyz[i*(ny-2*pml)+l] = snap[i*(nx-2*pml)*(ny-2*pml)+j*(ny-2*pml)+l];									
									}
									
									if(i==(nz-2*pml-1)/2)
									{
											snapxy[j*(ny-2*pml)+l] = snap[i*(nx-2*pml)*(ny-2*pml)+j*(ny-2*pml)+l];											
									}
							
						}		
				writefile_2d(snapxzname, snapxz, nz-2*pml, nx-2*pml);
				writefile_2d(snapyzname, snapyz, nz-2*pml, ny-2*pml);
				writefile_2d(snapxyname, snapxy, nx-2*pml, ny-2*pml);
					}

			//	printf("%f\n",absMaxval(snap, nx-2*pml, ny-2*pml, nz-2*pml));
				
		/*		sprintf(snapname_S,"%s_S%d.dat", snap_file, k);
				sprintf(snapSxzname,"%s_Sxz%d.dat", snap_file, k);
				sprintf(snapSyzname,"%s_Syz%d.dat", snap_file, k);
				sprintf(snapSxyname,"%s_Sxy%d.dat", snap_file, k);
				cudaMemcpy(snapS, S, nx*ny*nz*sizeof(float), cudaMemcpyDeviceToHost);
      	 		writefile_3d(snapname_S, snapS, nz, nx, ny); 
				for(i=0;i<nz;i++)
					for(j=0;j<nx;j++)
						for(l=0;l<ny;l++)
						{
									if(l==(ny-1)/2)
									{
											snapSxz[i*nx+j] = snapS[i*nx*ny+j*ny+l];										
									}
									
									if(j==(nx-1)/2)
									{
											snapSyz[i*ny+l] = snapS[i*nx*ny+j*ny+l];									
									}
									
									if(i==(nz-1)/2)
									{
											snapSxy[j*ny+l] = snapS[i*nx*ny+j*ny+l];											
									}
							
						}		
				writefile_2d(snapSxzname, snapSxz, nz, nx);
				writefile_2d(snapSyzname, snapSyz, nz, ny);
				writefile_2d(snapSxyname, snapSxy, nx, ny);*/
     	//	}	
  	}
	 
//	cudaMemcpy(record, d_record, nt*(nx-2*pml)*(ny-2*pml)*sizeof(float), cudaMemcpyDeviceToHost);

	//free device memory
 	cudaFree(d_vp);cudaFree(d_epsilon);cudaFree(d_delta);cudaFree(d_c);cudaFree(d_c2);
 	cudaFree(d_dlr);cudaFree(d_ddlr);cudaFree(d_dtb);cudaFree(d_ddtb);cudaFree(d_dfb);cudaFree(d_ddfb);
 	cudaFree(d_source);cudaFree(S);cudaFree(u1);cudaFree(u2);cudaFree(u3);
	cudaFree(ux);cudaFree(uy);cudaFree(uz);cudaFree(d_record);cudaFree(d_u);

 	cudaFree(wl11);cudaFree(wl12);cudaFree(wl13);cudaFree(wl21);
 	cudaFree(wl31);cudaFree(wl32);cudaFree(wl33);cudaFree(pl1);
 	cudaFree(pl2);cudaFree(pl3);
 	cudaFree(wr11);cudaFree(wr12);cudaFree(wr13);cudaFree(wr21);
 	cudaFree(wr31);cudaFree(wr32);cudaFree(wr33);cudaFree(pr1);
 	cudaFree(pr2);cudaFree(pr3);
 	cudaFree(wt11);cudaFree(wt12);cudaFree(wt13);cudaFree(wt21);
 	cudaFree(wt31);cudaFree(wt32);cudaFree(wt33);cudaFree(pt1);
 	cudaFree(pt2);cudaFree(pt3);
 	cudaFree(wb11);cudaFree(wb12);cudaFree(wb13);cudaFree(wb21);
 	cudaFree(wb31);cudaFree(wb32);cudaFree(wb33);cudaFree(pb1);
 	cudaFree(pb2);cudaFree(pb3);
	cudaFree(wf11);cudaFree(wf12);cudaFree(wf13);cudaFree(wf21);
 	cudaFree(wf31);cudaFree(wf32);cudaFree(wf33);cudaFree(pf1);
 	cudaFree(pf2);cudaFree(pf3);
 	cudaFree(wba11);cudaFree(wba12);cudaFree(wba13);cudaFree(wba21);
 	cudaFree(wba31);cudaFree(wba32);cudaFree(wba33);cudaFree(pba1);
 	cudaFree(pba2);cudaFree(pba3);

	free(h_ux); free(h_uy); free(h_uz); free(snap); free(snapxz);free(snapyz);free(snapxy);
	free(snapS);free(snapSxz);free(snapSyz);free(snapSxy);
	
	endtime = clock();
	timespent=(float)(endtime-starttime)/CLOCKS_PER_SEC;
	printf("Singshot modeling  time-assuming is %f s.\n",timespent);

}

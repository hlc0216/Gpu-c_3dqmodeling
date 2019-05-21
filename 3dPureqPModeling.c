/************ The program is writed by Lun Ruan, 2018.10***********************/
/*******3D Modeling for pure qP wave equation from Xu,2015************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <malloc.h>

#include "array_new.h"
#include "read_write.h"
#include "CPU_function.h"
#include "3dPureqPModeling_single_shot.h"

#define pi 3.1415926535898
#define M 10



int main(int argc, char *argv[])
{
	/*	int Nx0, Nz0, pml, nt, h , nshot, dshot, multishot, 
				muteflag, snapflag1, snapflag2, sx0, sz0, ntsnap,
				Nx1, Nz1, nx, nz, i, j, k, ishot, sx, sz;
		float dx, dz, dt, fm, t0, amp, alpha, vpmax;
		char vp_file[100], eps_file[100], del_file[100], recfull_file[100], 
				 recdir_file[100], recmut_file[100], snap_file1[100], snap_file2[100],buff[100];
		
		/********* input file ***********/
	/*	FILE *fp = NULL;
		fp = fopen("parameter.txt","r");
	  if (fp == NULL)
	  {
			 printf("parameter.txt is not exist !\n");
			 fclose(fp);
			 exit(0);
	  }
		
		fgets(buff,100,fp);
		fscanf(fp,"%f", &dt);fscanf(fp,"%d",&nt);fscanf(fp,"%d",&ntsnap);
		fgets(buff,100,fp);fgets(buff,100,fp);
		fscanf(fp,"%d", &Nx0);fscanf(fp,"%d", &Nz0);
		fgets(buff,100,fp);fgets(buff,100,fp);
		fscanf(fp,"%f", &dx);fscanf(fp,"%f", &dz);	
		fgets(buff,100,fp);fgets(buff,100,fp);
		fscanf(fp,"%f", &fm); 
		fgets(buff,100,fp);fgets(buff,100,fp);
		fscanf(fp,"%f", &amp);fscanf(fp,"%f", &alpha);	
		fgets(buff,100,fp);fgets(buff,100,fp);
		fscanf(fp,"%d", &sx0);fscanf(fp,"%d", &sz0);
		fgets(buff,100,fp);fgets(buff,100,fp);
		fscanf(fp,"%d", &nshot);fscanf(fp,"%d", &dshot); 
		fgets(buff,100,fp);fgets(buff,100,fp);
		fscanf(fp,"%d", &h); 
		fgets(buff,100,fp);fgets(buff,100,fp);
		fscanf(fp,"%d", &pml);
		fgets(buff,100,fp);fgets(buff,100,fp);
		fscanf(fp,"%s", vp_file);fscanf(fp,"%s", eps_file);fscanf(fp,"%s", del_file);
		fgets(buff,100,fp);fgets(buff,100,fp);
		fscanf(fp,"%s", recfull_file);fscanf(fp,"%s", recdir_file);fscanf(fp,"%s", recmut_file);
		fgets(buff,100,fp);fgets(buff,100,fp);
		fscanf(fp,"%s", snap_file2);fscanf(fp,"%s", snap_file1);
		fgets(buff,100,fp);fgets(buff,100,fp);
		fscanf(fp,"%d", &multishot);
		fgets(buff,100,fp);fgets(buff,100,fp);
		fscanf(fp,"%d", &muteflag);
		fgets(buff,100,fp);fgets(buff,100,fp);
		fscanf(fp,"%d", &snapflag2);
		fgets(buff,100,fp);fgets(buff,100,fp);
		fscanf(fp,"%d", &snapflag1);
		fclose(fp);
		
		
		Nx1 = Nx0 + h; Nz1 = Nz0; nx = h + 2*pml; nz = Nz1 + 2*pml; t0 = 1.0/fm;
*/
		/******** parameters input***********/
	/*	int Nx0 = 251, Ny0 = 251, Nz0 = 251, pml = 50, nt = 2501, hx = 251, hy = 251, nxshot = 1, nyshot = 1, dxshot = 10, dyshot = 10, multishot = 1, muteflag = 0, 
				snapflag1 = 0, snapflag2 = 0, sx0 = (Nx0-1)/2, sy0 = (Ny0-1)/2, sz0 = 0, ntsnap = 100;		
		float dx = 15.0, dy = 15.0, dz = 15.0, dt = 0.0008, fm = 20, t0 = 1.0/fm, amp = 1, alpha = 1;
		
		int nx = hx+2*pml, ny = hy+2*pml, nz = Nz0+2*pml;
		int i, j, k, ixshot,iyshot,  sx, sy, sz, sx1, sy1, sz1;
		float vpmax;
		
		/********* input file ***********/
	/*	const char *vp_file, *eps_file, *del_file;
		vp_file = "fault_model/vel.dat";
		eps_file = "fault_model/eps.dat";
		del_file = "fault_model/eps.dat";
		
		/*********  output file **********/
	/*	const char *recfull_file, *recdir_file, *recmut_file, *snap_file1, *snap_file2, *recfilexz, *recfileyz;
		recfull_file = "output/fault_model/Record_fullwave.dat";
		recdir_file = "output/fault_model/Record_direct.dat";
		recmut_file = "output/fault_model/Record_mute.dat";
		recfilexz = "output/fault_model/recordxz.dat";
		recfileyz = "output/fault_model/recordyz.dat";
		snap_file1 = "output/fault_model/snapshot_direct";
		snap_file2= "output/fault_model/snapshot_fullwave";
	
	
		/******** parameters input***********/
		int Nx0 = 101, Ny0 = 101, Nz0 = 101, pml = 50, nt = 501, hx = 101, hy = 101, nxshot = 1, nyshot = 1, dxshot = 10, dyshot = 10, multishot = 1, muteflag = 1, 
				snapflag1 = 0, snapflag2 = 1, sx0 = (Nx0-1)/2, sy0 = (Ny0-1)/2, sz0 = 0, ntsnap = 100;		
		float dx = 10.0, dy = 10.0, dz = 10.0, dt = 0.001, fm = 30, t0 = 1.0/fm, amp = 1, alpha = 1;
		
		int nx = hx+2*pml, ny = hy+2*pml, nz = Nz0+2*pml;
		int i, j, k, ixshot,iyshot, sx, sy, sz, sx1, sy1, sz1;
		float vpmax;
		
		/********* input file ***********/
		const char *vp_file, *eps_file, *del_file;
		vp_file = "twolayer/vp.dat";
		eps_file = "twolayer/epsilon.dat";
		del_file = "twolayer/delta.dat";
		
		/*********  output file **********/
		const char *recfull_file, *recdir_file, *recmut_file, *snap_file1, *snap_file2, *recfilexz, *recfileyz;
		recfull_file = "output/twolayer/Record_fullwave.dat";
		recdir_file = "output/twolayer/Record_direct.dat";
		recmut_file = "output/twolayer/Record_mute.dat";
		recfilexz = "output/twolayer/recordxz.dat";
		recfileyz = "output/twolayer/recordyz.dat";
		snap_file1 = "output/twolayer/snapshot_direct";
		snap_file2= "output/twolayer/snapshot_fullwave";
		
		/******allocate host memory *********/
		float *vp0 = array1d(Nx0*Ny0*Nz0), *epsilon0 = array1d(Nx0*Ny0*Nz0), *delta0 = array1d(Nx0*Ny0*Nz0),
					*vp1  = array1d(hx*hy*Nz0), *epsilon1  = array1d(hx*hy*Nz0), *delta1  = array1d(hx*hy*Nz0),
					*vp2 = array1d(Nx0*Ny0*Nz0), *epsilon2 = array1d(Nx0*Ny0*Nz0), *delta2 = array1d(Nx0*Ny0*Nz0),
					*vp  = array1d(nx*ny*nz),                *epsilon  = array1d(nx*ny*nz),                  *delta  = array1d(nx*ny*nz),
					*vp_d  = array1d(nx*ny*nz),              *epsilon_d  = array1d(nx*ny*nz),                *delta_d  = array1d(nx*ny*nz),
	        *source = array1d(nx*ny*nz),             *wavelet = array1d(nt),
	        *record_fullwave = array1d(hx*hy*nt),*record_direct = array1d(hx*hy*nt), *record_mute = array1d(hx*hy*nt),
					*dlr = array1d(pml), *ddlr = array1d(pml), *dtb = array1d(pml), *ddtb = array1d(pml),*dfb = array1d(pml), *ddfb = array1d(pml),
					*recordxz = array1d(nt*hx), *recordyz = array1d(nt*hy);
				
		
	  /********* transform y-z direction **********/
	/*  readfile_3d("../vel2.rsf@", vp0, Ny0, Nx0, Nz0);
	  readfile_3d("eps3.rsf@", epsilon0, Ny0, Nx0, Nz0);
	  readfile_3d("eps3.rsf@", delta0, Ny0, Nx0, Nz0);
	  
	  trans3d(vp0, vp2, Nx0, Ny0, Nz0);
	  trans3d(epsilon0, epsilon2, Nx0, Ny0, Nz0);
	  trans3d(delta0, delta2, Nx0, Ny0, Nz0);
	  
	  writefile_3d(vp_file, vp2, Ny0, Nx0, Nz0);
	  writefile_3d(eps_file, epsilon2, Ny0, Nx0, Nz0);
	  writefile_3d(del_file, delta2, Ny0, Nx0, Nz0);*/
	
		for(i=0;i<Nz0;i++)
	  	for(j=0;j<Nx0;j++)
	  		for(k=0;k<Ny0;k++)
	  		{
	  				if(i<(Nz0-1)/2)
	  				{
								vp2[i*Nx0*Ny0+j*Ny0+k] = 2500;
								epsilon2[i*Nx0*Ny0+j*Ny0+k] = 0.0;
								delta2[i*Nx0*Ny0+j*Ny0+k] = 0.0;
						}	
						else
						{
								vp2[i*Nx0*Ny0+j*Ny0+k] = 3500;
								epsilon2[i*Nx0*Ny0+j*Ny0+k] = 0.0;
								delta2[i*Nx0*Ny0+j*Ny0+k] = 0.0;
						}
	  		}
	  
	  writefile_3d(vp_file, vp2, Nz0, Nx0, Ny0);
	  writefile_3d(eps_file, epsilon2, Nz0, Nx0, Ny0);
	  writefile_3d(del_file, delta2, Nz0, Nx0, Ny0);
	
		//first-order center difference cofficient
	  float *c = array1d((M/2+1)*(M/2+1));
		fdcoff1(c);
	 
	  //second-order center difference cofficient
 		float *c2 = array1d((M/2+1)*(M/2+1));
 		fdcoff2(c2);
		

		 //initialize file
		FILE *fp;
		fp = fopen(recfull_file,"wb");
		fclose(fp);
		fp = fopen(recmut_file,"wb");
		fclose(fp);

		 //modeling-info-diaplay
		printf("\n****************** Parameters *********************\n");
		printf("ModelSize = %d * %d * %d \n", Nx0, Ny0, Nz0);
		printf("dx = %f dy = %f dz = %f nt = %d dt = %f fm = %f\n",dx, dy, dz, nt, dt, fm);
		printf("sx0 = %d sy0 = %d  sz0 = %d offsetx = %d offsety = %d nshot = %d dxshot = %d dyshot = %d\n", sx0, sy0, sz0, hx, hy, nxshot*nyshot, dxshot, dyshot);
		printf("****************************************************\n\n");
		printf("            Staring to Modeling !\n\n");
		
		
		if(multishot == 1)
	 {	
				//direct-wave model
			/*	for(i=0;i<nz;i++)
					 for(j=0;j<nx;j++)
						  for(k=0;k<ny;k++)
					   {																	
								vp_d[i*nx*ny+j*ny+k] = 1490;
								epsilon_d[i*nx*ny+j*ny+k] = 0.0;
								delta_d[i*nx*ny+j*ny+k] = 0.0;																			
						 }
				*/
				for(i=0;i<nz;i++)
					 for(j=0;j<nx;j++)
						  for(k=0;k<ny;k++)
					   {																	
								vp_d[i*nx*ny+j*ny+k] = 2500;
								epsilon_d[i*nx*ny+j*ny+k] = 0.0;
								delta_d[i*nx*ny+j*ny+k] = 0.0;																			
						 }
				
				
			 //modeling direct-wave
			 if(muteflag == 1)
			{
					printf("/************* Modeling directwave ***************/\n");
					
					vpmax = (1 + 2*absMaxval(epsilon_d, nx, ny, nz))*absMaxval(vp_d, nx, ny, nz);
					
				  pmlcoff(pml, vpmax, dx, dy, dz, dlr, ddlr, dtb, ddtb, dfb, ddfb);
				  
				  sx1 = (hx-1)/2+pml;
				  sy1 = (hy-1)/2+pml;
				  sz1 = pml;
					
					Source(sx1, sy1, sz1, fm, amp, alpha, dt, dx, dy, dz, t0, nt, nx, ny, nz, source, wavelet);
					
					modeling3d(nx, ny, nz, nt, ntsnap, dx, dy, dz, dt, pml, snapflag1, sx1, sy1, sz1, vp_d, epsilon_d, delta_d, 
										source, wavelet, record_direct, dlr, ddlr, dtb, ddtb, dfb, ddfb, c, c2, snap_file1);

					writefile_3d(recdir_file, record_direct, nt, hx, hy);		
					
				/*	for(i=0;i<hx;i++)
						 	for(j=0;j<hy;j++)
								for(k=0;k<nt;k++)
								{
										if(j==sy1-pml)
											recordxz[k*hx+i] = record_direct[k*hx*hy+i*hy+j];

										if(i==sx1-pml)
											recordyz[k*hy+j] = record_direct[k*hx*hy+i*hy+j]; 

								}
					
						writefile_2d(recfilexz , recordxz, nt, hx);
						writefile_2d(recfileyz , recordyz, nt, hy);*/
					
			 }
			 
			 
			 
				printf("/************* Multi-shots Modeling  ***************/\n");
		
			//shot loop  (first-y direction(fast dimention) and second-x direction (slow dimention))
		for(ixshot=0;ixshot<nxshot;ixshot++)
		   for(iyshot=0;iyshot<nyshot;iyshot++)
			 {
				 printf("shot=%d\n", ixshot*nyshot+iyshot);
				 
				 sx = sx0 + ixshot * dxshot;
				 sy = sy0 + iyshot * dyshot;
				 sz = sz0;
				 
				 //single-shot getting model-parameters 
				 readmodel3d(vp_file, vp1, hy, hx, Nz0, sy-(hy-1)/2, sx-(hx-1)/2, 0, Ny0, Nx0, Nz0);
				 readmodel3d(eps_file, epsilon1, hy, hx, Nz0, sy-(hy-1)/2, sx-(hx-1)/2, 0, Ny0, Nx0, Nz0);
				 readmodel3d(del_file, delta1, hy, hx, Nz0, sy-(hy-1)/2, sx-(hx-1)/2, 0, Ny0, Nx0, Nz0);
				 
		//		 writefile_3d("vptest.dat", vp1, Ny0, Nx0, Nz0);
	  		 
				 
				for(i=0;i<Nz0;i++)
					 for(j=0;j<hx;j++)
						  for(k=0;k<hy;k++)
					   {	
					   		if(i<(Nz0-1)/2)
					   		{
					   				vp1[i*hx*hy+j*hy+k] = 2500;
										epsilon1[i*hx*hy+j*hy+k] = 0.0;
										delta1[i*hx*hy+j*hy+k] = 0.0;	
					   		}
					   		else
					   		{
					   				vp1[i*hx*hy+j*hy+k] = 3500;
										epsilon1[i*hx*hy+j*hy+k] = 0.0;
										delta1[i*hx*hy+j*hy+k] = 0.0;	
					   		}																
																										
						 }
						
		/*		for(i=0;i<nz;i++)
					 for(j=0;j<nx;j++)
						  for(k=0;k<ny;k++)
					   {																	
								vp[i*nx*ny+j*ny+k] = 3000;
								epsilon[i*nx*ny+j*ny+k] = 0.0;
								delta[i*nx*ny+j*ny+k] = 0.0;																			
						 }*/
				
								
				 extend_model(vp1, vp, nx, ny, nz, pml);
				 extend_model(epsilon1, epsilon, nx, ny, nz, pml);
				 extend_model(delta1, delta, nx, ny, nz, pml);
				
				 //PML ATTENUATION COEFFICIENT 
				 vpmax = (1 + 2*absMaxval(epsilon, nx, ny, nz))*absMaxval(vp, nx, ny, nz);
					
				 pmlcoff(pml, vpmax, dx, dy, dz, dlr, ddlr, dtb, ddtb, dfb, ddfb);
					
				 sx1 = (hx-1)/2+pml;
				 sy1 = (hy-1)/2+pml;
				 sz1 = pml;
				 
				 Source(sx1, sy1, sz1, fm, amp, alpha, dt, dx, dy, dz, t0, nt, nx, ny, nz, source, wavelet);
					
				 modeling3d(nx, ny, nz, nt, ntsnap, dx, dy, dz, dt, pml, snapflag2, sx1, sy1, sz1, vp, epsilon, delta, 
										source, wavelet, record_fullwave, dlr, ddlr, dtb, ddtb, dfb, ddfb, c, c2, snap_file2);
				
				// Mute directwave 
				for(i=0;i<hx;i++)
					for(j=0;j<hy;j++)
						for(k=0;k<nt;k++)
								record_mute[k*hx*hy+i*hy+j] = record_fullwave[k*hx*hy+i*hy+j]-record_direct[k*hx*hy+i*hy+j];
									
				writefile2_3d(recfull_file, record_fullwave, nt, hx, hy);
				writefile2_3d(recmut_file, record_mute, nt, hx, hy);
				
				if(ixshot == 0 && iyshot == 0)
				{
					for(i=0;i<hx;i++)
						 	for(j=0;j<hy;j++)
								for(k=0;k<nt;k++)
								{
										if(j==sy1-pml)
											recordxz[k*hx+i] = record_mute[k*hx*hy+i*hy+j];

										if(i==sx1-pml)
											recordyz[k*hy+j] = record_mute[k*hx*hy+i*hy+j]; 

								}
					
						writefile_2d(recfilexz , recordxz, nt, hx);
						writefile_2d(recfileyz , recordyz, nt, hy);
						
				/*	 for(i=0;i<hx;i++)
						 	for(j=0;j<hy;j++)
								for(k=0;k<nt;k++)
								{
										if(j==sy1-pml)
											recordxz[k*hx+i] = record_fullwave[k*hx*hy+i*hy+j];

										if(i==sx1-pml)
											recordyz[k*hy+j] = record_fullwave[k*hx*hy+i*hy+j]; 

								}
					
						writefile_2d("output/twolayer/recordfullwavexz.dat" , recordxz, nt, hx);
						writefile_2d("output/twolayer/recordfullwaveyz.dat" , recordyz, nt, hy);*/
				}
		 }
   }
	  else if(multishot == 0)
	 {
		  printf("/************* single-shot Modeling ***************/\n");
		  nx = 	Nx0+2*pml;
		  ny = 	Ny0+2*pml;
		  nz = 	Nz0+2*pml;
		  
		  float *vp_s = array1d(nx*ny*nz), *epsilon_s = array1d(nx*ny*nz), *delta_s = array1d(nx*ny*nz),
					*record_singleshot = array1d(nx*ny*nt), *source1 = array1d(nx*ny*nz), *recordxz_s = array1d(nx*nt),
					*recordyz_s = array1d(ny*nt);
			
					
			extend_model(vp0, vp_s, nx, ny, nz, pml);
			extend_model(epsilon0, epsilon_s, nx, ny, nz, pml);
			extend_model(delta0, delta_s, nx, ny, nz, pml);
		
			vpmax = (1 + 2*absMaxval(epsilon_s, nx, ny, nz))*absMaxval(vp_s, nx, ny, nz);
			pmlcoff(pml, vpmax, dx, dy, dz, dlr, ddlr, dtb, ddtb, dfb, ddfb);
			Source(sx0+pml, sy0+pml, sz0+pml, fm, amp, alpha, dt, dx, dy, dz, t0, nt, nx, ny, nz, source1, wavelet);
	
			modeling3d(nx, ny, nz, nt, ntsnap, dx, dy, dz, dt, pml, snapflag2, sx0+pml, sy0+pml, sz0+pml, vp_s, epsilon_s, delta_s, 
										source1, wavelet, record_singleshot, dlr, ddlr, dtb, ddtb, dfb, ddfb, c, c2, snap_file2);
							
			writefile_3d(recfull_file, record_singleshot, nt, nx-2*pml, ny-2*pml);	
			
			for(i=0;i<nx-2*pml;i++)
			 	for(j=0;j<ny-2*pml;j++)
					for(k=0;k<nt;k++)
					{
							if(j==sy0+pml)
								recordxz_s[k*(nx-2*pml)+i] = record_direct[k*(nx-2*pml)*(ny-2*pml)+i*(ny-2*pml)+j];

							if(i==sx0+pml)
								recordyz_s[k*(ny-2*pml)+j] = record_direct[k*(nx-2*pml)*(ny-2*pml)+i*(ny-2*pml)+j]; 

					}
		
			writefile_2d(recfilexz , recordxz_s, nt, nx-2*pml);
			writefile_2d(recfileyz , recordyz_s, nt, ny-2*pml);
			
			free(vp_s);free(epsilon_s);free(delta_s);free(record_singleshot);free(source1);free(recordxz_s);free(recordyz_s); 	 
	 }
	 

	 
	 
	//free  memory
  free(vp) ; free(epsilon) ; free(delta);
  free(vp0); free(epsilon0); free(delta0);
	free(vp1) ; free(epsilon1) ; free(delta1);
	free(vp2) ; free(epsilon2) ; free(delta2);
	free(vp_d) ; free(epsilon_d) ; free(delta_d);
  free(source);free(wavelet);
  free(dlr);free(ddlr);free(dtb);free(ddtb);free(dfb);free(ddfb);free(c);free(c2);
  free(record_mute);free(record_fullwave);free(record_direct);

	
		

	return 0;

}

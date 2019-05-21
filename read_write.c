#include<stdio.h>
#include "read_write.h"

void readfile_2d(const char *filename, float *a, int nz, int nx)
{
	  int i,j;
	  FILE *fp;
	  fp = fopen(filename,"rb+");
	  for(i=0;i<nz;i++)
	     for(j=0;j<nx;j++)
		        fread(&a[i*nx+j],sizeof(float),1,fp);
		fclose(fp);

}

void readfile_3d(const char *filename, float *a, int nz, int nx, int ny)
{
	  int i,j,k;
	  FILE *fp;
	  fp = fopen(filename,"rb+");
	  for(i=0;i<nz;i++)
	     for(j=0;j<nx;j++)
					for(k=0;k<ny;k++)
		         fread(&a[i*nx*ny+j*ny+k],sizeof(float),1,fp);
		fclose(fp);

}

void writefile_2d(const char *filename, float *a, int nz, int nx)
{
	  int i,j;
	  FILE *fp;
	  fp = fopen(filename,"wb+");
	  for(i=0;i<nz;i++)
	     for(j=0;j<nx;j++)
		        fwrite(&a[i*nx+j],sizeof(float),1,fp);
		fclose(fp);

}

void writefile_3d(const char *filename, float *a, int nz, int nx, int ny)
{
	  int i,j,k;
	  FILE *fp;
	  fp = fopen(filename,"wb+");
	  for(i=0;i<nz;i++)
	     for(j=0;j<nx;j++)
					for(k=0;k<ny;k++)
		         fwrite(&a[i*nx*ny+j*ny+k],sizeof(float),1,fp);
		fclose(fp);

}

void writefile2_2d(const char *filename, float *a, int nz, int nx)
{
	  int i,j;
	  FILE *fp;
	  fp = fopen(filename,"ab+");
	  for(i=0;i<nz;i++)
	     for(j=0;j<nx;j++)
		        fwrite(&a[i*nx+j],sizeof(float),1,fp);
		fclose(fp);

}

void writefile2_3d(const char *filename, float *a, int nz, int nx, int ny)
{
	  int i,j,k;
	  FILE *fp;
	  fp = fopen(filename,"ab+");
	  for(i=0;i<nz;i++)
	     for(j=0;j<nx;j++)
					for(k=0;k<ny;k++)
		         fwrite(&a[i*nx*ny+j*ny+k],sizeof(float),1,fp);
		fclose(fp);

}




#include <math.h>
#include <stdio.h>
#include <malloc.h>
#include "CPU_function.h"
#include <time.h>
#define pi 3.1415926535898
#define M 10
#define eps 2.22e-17
//Max
float absMaxval(float *a, int nx, int ny, int nz)
{
	int i, j, k;
	float Max = 0.0;
	for (i = 0; i < nz; i++)
		for (j = 0; j < nx; j++)
			for (k = 0; k < ny; k++)
			{
				if (fabs(a[i * nx * ny + j * ny + k]) >= Max)
					Max = fabs(a[i * nx * ny + j * ny + k]);
			}
	return Max;
}

float absMaxval2d(float *a, int nx, int ny)
{
	int i, j;
	float Max = 0.0;
	for (i = 0; i < nx; i++)
		for (j = 0; j < ny; j++)
			if (fabs(a[i * ny + j]) >= Max)
				Max = fabs(a[i * ny + j]);

	return Max;
}

//Source
void Source(int sx, int sy, int sz, float fm, float amp, float alpha, float dt, float dx, float dy,
						float dz, float t0, int nt, int nx, int ny, int nz, float *source, float *wavelet)
{
	int i, j, k;
	float temp, t;

	for (i = 0; i < nz; i++)
		for (j = 0; j < nx; j++)
			for (k = 0; k < ny; k++)
			{
				source[i * nx * ny + j * ny + k] = amp * exp(-pow(alpha, 2) * (pow((j - sx) * dx, 2) + pow((k - sy) * dy, 2) + pow((i - sz) * dz, 2)));

				/*	if(i==sz && j==sx && k==sy)
							 source[i*nx*ny+j*ny+k] = 1.0;
						else
							 source[i*nx*ny+j*ny+k] = 0.0;*/
			}

	for (k = 0; k < nt; k++)
	{
		t = k * dt;
		temp = pow(pi * fm * (t - t0), 2);
		wavelet[k] = (1 - 2 * temp) * exp(-temp);

		//	if(k<100)
		//	printf("%f\n",wavelet[k]);
	}
}

//extend-pml-domain
void extend_model(float *in, float *out, int nx, int ny, int nz, int pml)
{
	int i, j, k;
	//middle
	for (i = pml; i < nz - pml; i++)
		for (j = pml; j < nx - pml; j++)
			for (k = pml; k < ny - pml; k++)
				out[i * nx * ny + j * ny + k] = in[(i - pml) * (nx - 2 * pml) * (ny - 2 * pml) + (j - pml) * (ny - 2 * pml) + k - pml];

	//z-direction-extend
	for (i = 0; i < pml; i++)
		for (j = pml; j < nx - pml; j++)
			for (k = pml; k < ny - pml; k++)
				out[i * nx * ny + j * ny + k] = out[pml * nx * ny + j * ny + k];

	for (i = nz - pml; i < nz; i++)
		for (j = pml; j < nx - pml; j++)
			for (k = pml; k < ny - pml; k++)
				out[i * nx * ny + j * ny + k] = out[(nz - pml - 1) * nx * ny + j * ny + k];

	//x-direction-extend
	for (i = 0; i < nz; i++)
		for (j = 0; j < pml; j++)
			for (k = pml; k < ny - pml; k++)
				out[i * nx * ny + j * ny + k] = out[i * nx * ny + pml * ny + k];

	for (i = 0; i < nz; i++)
		for (j = nx - pml; j < nx; j++)
			for (k = pml; k < ny - pml; k++)
				out[i * nx * ny + j * ny + k] = out[i * nx * ny + (nx - pml - 1) * ny + k];

	//y-direction-extend
	for (i = 0; i < nz; i++)
		for (j = 0; j < nx; j++)
			for (k = 0; k < pml; k++)
				out[i * nx * ny + j * ny + k] = out[i * nx * ny + j * ny + pml];

	for (i = 0; i < nz; i++)
		for (j = 0; j < nx; j++)
			for (k = ny - pml; k < ny; k++)
				out[i * nx * ny + j * ny + k] = out[i * nx * ny + j * ny + ny - pml - 1];
}

//一阶导数中心差分系数
void fdcoff1(float *c)
{
	c[7] = 0.5;
	c[13] = 0.667;
	c[14] = -0.083;
	c[19] = 0.75;
	c[20] = -0.15;
	c[21] = 0.0167;
	c[25] = 0.8;
	c[26] = -0.2;
	c[27] = 0.038;
	c[28] = -0.0036;
	c[31] = 0.833;
	c[32] = -0.2381;
	c[33] = 0.059;
	c[34] = -0.0099;
	c[35] = 0.00079;
}

//二阶导数中心差分系数
void fdcoff2(float *c)
{
	c[7] = 1.0;
	c[13] = 4.0 / 3.0;
	c[14] = -1.0 / 12.0;
	c[19] = 1.5;
	c[20] = -0.15;
	c[21] = 1.0 / 90.0;
	c[25] = 1.6;
	c[26] = -0.2;
	c[27] = 8.0 / 315.0;
	c[28] = -1.0 / 560;
	c[31] = 1.6667;
	c[32] = -0.2381;
	c[33] = 0.0397;
	c[34] = -0.0050;
	c[35] = 0.0003;
}

void pmlcoff(int pml, float vpmax, float dx, float dy, float dz, float *dlr,
						 float *ddlr, float *dtb, float *ddtb, float *dfb, float *ddfb) //top-bottom, left-right, forward-backward
{
	float R = 0.00001, widlr = pml * dx, widtb = pml * dz, widfb = pml * dy, widx, widz, widy;
	int i;
	for (i = 0; i < pml; i++)
	{
		widx = (i + 1) * dx;
		dlr[i] = 4 * vpmax / (2 * widlr) * pow(widx / widlr, 3) * logf(1 / R);
		ddlr[i] = 6 * vpmax * widx * widx / pow(widlr, 4) * logf(1 / R);

		widz = (i + 1) * dz;
		dtb[i] = 4 * vpmax / (2 * widtb) * pow(widz / widtb, 3) * logf(1 / R);
		ddtb[i] = 6 * vpmax * widz * widz / pow(widtb, 4) * logf(1 / R);

		widy = (i + 1) * dy;
		dfb[i] = 4 * vpmax / (2 * widfb) * pow(widy / widfb, 3) * logf(1 / R);
		ddfb[i] = 6 * vpmax * widy * widy / pow(widfb, 4) * logf(1 / R);
	}
}

/*void readmodel3d(const char *filename,float *a,int ny,int nx,int nz,int fny,int fnx,int fnz,int vny,int vnx,int vnz,int nbd) 
{
    int i,j,k,im,jm,km,iy,ix;
	int padny=2*nbd+ny;
	int padnx=2*nbd+nx;
	int padnz=2*nbd+nz;
   	FILE *fp;
	long offset;

    fp = fopen(filename,"rb+");
//	fseek(fp,offset,SEEK_SET);
    for(i=0;i<nz;i++)
    {
				im=i+nbd;
    		for(j=0;j<nx;j++)
    	 {
					jm=j+nbd;
		      offset=((long((fnz+i))*vnx+fnx+j)*vny+fny)*sizeof(float);
		    	fseek(fp,offset,SEEK_SET);
		     	for(k=0;k<ny;k++)
		        	fread(&a[im*(nx+2*nbd)*(ny+2*nbd)+jm*(ny+2*nbd)+k+nbd],sizeof(float),1,fp);
				}
		}
    fclose(fp);
    
    for(i=nbd; i<nbd+nz; i++)
			for(j=nbd; j<nbd+nx; j++)
				for(k=0; k<nbd; k++)
						a[i*(nx+2*nbd)*(ny+2*nbd)+j*(ny+2*nbd)+k] = a[i*(nx+2*nbd)*(ny+2*nbd)+j*(ny+2*nbd)+nbd];  

    for(i=nbd; i<nbd+nz; i++)
    		for(j=nbd; j<nbd+nx; j++)
    			for(k=0; k<nbd; k++)
        		//a[i][j][k+nbd+ny]=a[i][j][nbd+ny-1];
        		a[i*(nx+2*nbd)*(ny+2*nbd)+j*(ny+2*nbd)+k+nbd+ny] = a[i*(nx+2*nbd)*(ny+2*nbd)+j*(ny+2*nbd)+nbd+ny-1];
  
		for(i=nbd; i<nbd+nz; i++)
    		for(k=0; k<padny; k++)
    			for(j=0; j<nbd; j++)
     //   		a[i][j][k]=a[i][nbd][k];
        			a[i*(nx+2*nbd)*(ny+2*nbd)+j*(ny+2*nbd)+k] = a[i*(nx+2*nbd)*(ny+2*nbd)+nbd*(ny+2*nbd)+k];

    for(i=nbd; i<nbd+nz; i++)
				for(k=0; k<padny; k++)
					for(j=0; j<nbd; j++)
	    //			a[i][j+nbd+nx][k]=a[i][nbd+nx-1][k];
	    				a[i*(nx+2*nbd)*(ny+2*nbd)+(j+nbd+nx)*(ny+2*nbd)+k] = a[i*(nx+2*nbd)*(ny+2*nbd)+(j+nbd+nx-1)*(ny+2*nbd)+k];

		for(j=0; j<padnx; j++)
			for(k=0; k<padny; k++)
				for(i=0; i<nbd; i++)
	   	//		 a[i][j][k]=a[nbd][j][k];
	   				a[i*(nx+2*nbd)*(ny+2*nbd)+j*(ny+2*nbd)+k] = a[nbd*(nx+2*nbd)*(ny+2*nbd)+j*(ny+2*nbd)+k];  

		for(j=0; j<padnx; j++)
			for(k=0; k<padny; k++)
				for(i=0; i<nbd; i++)
	   // 		a[i+nbd+nz][j][k]=a[nbd+nz-1][j][k];
	   		a[(i+nbd+nz)*(nx+2*nbd)*(ny+2*nbd)+j*(ny+2*nbd)+k] = a[(nbd+nz-1)*(nx+2*nbd)*(ny+2*nbd)+j*(ny+2*nbd)+k];  

}
*/

void readmodel3d(const char *filename, float *a, int ny, int nx, int nz, int fny, int fnx, int fnz, int vny, int vnx, int vnz)
{
	int i, j, k, im, jm;
	FILE *fp;
	long offset;
	float *temp = (float *)malloc(sizeof(float) * nx * ny * nz);
	printf("start open file:%s\n", filename);
	fp = fopen(filename, "rb+");
	if (fp != NULL)
		printf("%s文件已经打开", filename);
	for (i = 0; i < ny; i++)
		for (j = 0; j < nx; j++)
		{
			offset = (long)((((fny + i)) * vnx + fnx + j) * vnz + fnz) * sizeof(float);
			fseek(fp, offset, SEEK_SET);
			for (k = 0; k < nz; k++)
				fread(&temp[i * nx * nz + j * nz + k], sizeof(float), 1, fp);
		}
	fclose(fp);
	printf("nx=%d ny= %d nz=%d \n", nx, ny, nz);
	for (i = 0; i < nz; i++)
		for (j = 0; j < nx; j++)
			for (k = 0; k < ny; k++)
			{
				a[i * nx * ny + j * ny + k] = temp[k * nx * nz + j * nz + i];
			}
	free(temp);
}

/*void trans2d(float *in, float *out, int nx, int nz)
{
	int i, j;
	for(i=0;i<nz;i++)
		for(j=0;j<nx;j++)
			out[i*nx+j] = in[j*nz+i];
}

void trans3d(float *in, float *out, int nx, int ny, int nz)
{
	int i, j, k;
	for(i=0;i<nz;i++)
		for(j=0;j<nx;j++)
			for(k=0;k<ny;k++)
				out[i*nx*ny+j*ny+k] = in[k*nx*nz+j*nz+i];
}*/

void trans_z(float *in, int nx, int ny, int nz)
{
	int i1, i2, j, l;
	float temp;
	for (j = 0; j < nx; j++)
		for (l = 0; l < ny; l++)
		{
			i1 = 0;
			i2 = nz - 1;
			while (i1 < i2)
			{
				temp = in[i2 * nx * ny + j * ny + l];
				in[i2 * nx * ny + j * ny + l] = in[i1 * nx * ny + j * ny + l];
				in[i1 * nx * ny + j * ny + l] = temp;
				i1++;
				i2--;
			}
		}
}
/*
//-----------------------------------------------------hlc修改添加的5个---------------------------------------------------------------------------------------
void grad(float *u2, float *ux, float *uy, float *uz, float *d_c,
					int nx, int ny, int nz, float dx, float dy, float dz)
{
	int m, N = M / 2;
	for (int i = M / 2; i < nz - M / 2; i++)
	{
		for (int j = M / 2; j < nx - M / 2; j++)
		{
			for (int k = M / 2; k < ny - M / 2; k++)
			{
				float Ux = 0.0, Uy = 0.0, Uz = 0.0;
				for (m = 1; m < N + 1; m++)
				{
					Ux = Ux + d_c[N * (M / 2 + 1) + m] * (u2[i * ny * nx + (j + m) * ny + k] - u2[i * ny * nx + (j - m) * ny + k]) / dx;
					Uy = Uy + d_c[N * (M / 2 + 1) + m] * (u2[i * ny * nx + j * ny + k + m] - u2[i * ny * nx + j * ny + k - m]) / dy;
					Uz = Uz + d_c[N * (M / 2 + 1) + m] * (u2[(i + m) * ny * nx + j * ny + k] - u2[(i - m) * ny * nx + j * ny + k]) / dz;
				}
				ux[i * ny * nx + j * ny + k] = Ux;
				uy[i * ny * nx + j * ny + k] = Uy;
				uz[i * ny * nx + j * ny + k] = Uz;
			}
		}
	}
}
void addsource(float *d_source, float wavelet, float *u3, int nx, int ny, int nz)
{
	for (int i = 0; i < nz; i++)
	{
		for (int j = 0; j < nx; j++)
		{
			for (int k = 0; k < ny; k++)
			{
				u3[i * ny * nx + j * ny + k] += d_source[i * ny * nx + j * ny + k] * wavelet;
			}
		}
	}
}
void scalar_operator(float uxyzMax, float *ux, float *uy, float *uz, float *d_epsilon, float *d_delta, float *S, int nx, int ny, int nz)
{
	float nxvector, nyvector, nzvector, mode;
	int idx, idy, idz;
	for (idz = 0; idz < nz; idz++)
	{
		for (idx = 0; idx < nx; idx++)
		{
			for (idy = 0; idy < ny; idy++)
			{
				S[idz * ny * nx + idx * ny + idy] = 1.0;
				if (idx >= 5 && idx < nx - 5 && idy >= 5 && idy < ny - 5 && idz >= 5 && idz < nz - 5)
				{
					mode = sqrt(ux[idz * ny * nx + idx * ny + idy] * ux[idz * ny * nx + idx * ny + idy] + uy[idz * ny * nx + idx * ny + idy] * uy[idz * ny * nx + idx * ny + idy] + uz[idz * ny * nx + idx * ny + idy] * uz[idz * ny * nx + idx * ny + idy]);
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
				}
			}
		}
	}
}
void wavefield_update1(float *d_R, float *d_c, float *d_c2, float *d_dlr, float *d_ddlr, float *d_dtb, float *d_ddtb, float *d_dfb,
											 float *d_ddfb, float *d_epsilon, float *d_delta, float *d_vp, float dx, float dy, float dz, float dt,
											 int nx, int ny, int nz, int pml, float *ux, float *uy, float *uz, float *u1, float *u3, float *u2, float *S,
											 float *wl11, float *wl12, float *wl13, float *wl21, float *wl31, float *wl32, float *wl33, float *pl1, float *pl2, float *pl3,
											 float *wr11, float *wr12, float *wr13, float *wr21, float *wr31, float *wr32, float *wr33, float *pr1, float *pr2, float *pr3,
											 float *wt11, float *wt12, float *wt13, float *wt21, float *wt31, float *wt32, float *wt33, float *pt1, float *pt2, float *pt3,
											 float *wf11, float *wf12, float *wf13, float *wf21, float *wf31, float *wf32, float *wf33, float *pf1, float *pf2, float *pf3,
											 float *wba11, float *wba12, float *wba13, float *wba21, float *wba31, float *wba32, float *wba33, float *pba1, float *pba2, float *pba3)
{
	int idx, idy, idz;
	for (idz = M / 2; idz < nz - M / 2; idz++)
	{
		for (idx = M / 2; idx < nx - M / 2; idx++)
		{
			for (idy = M / 2; idy < ny - M / 2; idy++)
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
	}
}
void exchange1(int nx, int ny, int nz, int pml, float *u1, float *u2, float *u3,
							 float *wl11, float *wl12, float *wl13, float *wl31, float *wl32, float *wl33, float *pl1, float *pl2, float *pl3,
							 float *wr11, float *wr12, float *wr13, float *wr31, float *wr32, float *wr33, float *pr1, float *pr2, float *pr3,
							 float *wt11, float *wt12, float *wt13, float *wt31, float *wt32, float *wt33, float *pt1, float *pt2, float *pt3,
							 float *wf11, float *wf12, float *wf13, float *wf31, float *wf32, float *wf33, float *pf1, float *pf2, float *pf3,
							 float *wba11, float *wba12, float *wba13, float *wba31, float *wba32, float *wba33,
							 float *pba1, float *pba2, float *pba3)
{
	int idx, idy, idz;
	int i, i1, i2, i3, i4, i5;
	// i1 = idz * ny * pml + idx * ny + idy;
	// i2 = idz * ny * pml + (idx - nx + pml) * ny + idy;
	// i3 = idz * nx * pml + idx * pml + idy;
	// i4 = idz * nx * pml + idx * pml + idy - ny + pml;
	// i5 = i;
	for (idz = 0; idz < nz; idz++)
	{
		for (idx = 0; idx < nx; idx++)
		{
			for (idy = 0; idy < ny; idy++)
			{
				i = idz * ny * nx + idx * ny + idy;
				u1[i] = u2[i];
				u2[i] = u3[i];
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
	}
}
*/
/*void check(cudaError_t a)
{
	 if(cudaSuccess != a)
	{
		fprintf(stderr, "Cuda runtime error in line %d of file %s: %s \n", __LINE__, __FILE__, cudaGetErrorString(cudaGetLastError()));
		exit(-1);
	}		     
}
*/

/*
void Outputrecord(float *record, int nt, int nx, float dt, char buff[40], int Out_flag)
{
	int it,ix;
	FILE *fp=NULL;
	if (Out_flag==1)
	{
		float *temp =(float *)malloc(nt*sizeof(float));
		fp=fopen(buff,"wb");
		if (fp==NULL)
		{
			printf("The file %s open failed !\n",buff);
		}
		short int header[120];
		for (it=0;it<120;it++)
		{
			header[it]=0;
		}
		header[57]=(short int)(nt);
		header[58]=(short int)(dt*1000000.0);         // dt
		header[104]=(short int)(nx);
		for (ix=0;ix<nx;ix++)
		{
			header[0]=ix+1;
			fwrite(header,2,120,fp);

			for (it=0; it<nt; it++)
				temp[it] = record[it*nx+ix];

			fwrite(temp,sizeof(float),nt,fp);
		}
		fclose(fp);
		free(temp);
	}
	else
	{
		fp=fopen(buff,"wb");
		if (fp==NULL)
		{
			printf("The file %s open failed !\n",buff);
		}
		short int header[120];
		for (it=0;it<120;it++)
		{
			header[it]=0;
		}
		header[57]=(short int)(nt);
		header[58]=(short int)(dt*1000000.0);         // dt
		header[104]=(short int)(nx);
		for (ix=0;ix<nx;ix++)
		{
			header[0]=ix+1;
			fwrite(header,2,120,fp);
			fwrite(&record[ix*nt],sizeof(float),nt,fp);
		}
		fclose(fp);
	} 
}

void Inputrecord(float *record, int nt, int nx, char buff[40],int In_flag)
{
	int ix,it;
	FILE *fp=NULL;
	if (In_flag==1)
	{
		float *temp =(float *)malloc(nt*sizeof(float));
		fp=fopen(buff,"rb");
		if (fp==NULL)
		{
			printf("The file %s open failed !\n",buff);
		}
		for (ix=0;ix<nx;ix++)
		{
			fseek(fp,240L,1);
			fread(temp,sizeof(float),nt,fp);
			for (it=0; it <nt; it++)
				record[it*nx+ix] = temp[it];
		}
		fclose(fp);
		free(temp);
	} 
	else
	{
		fp=fopen(buff,"rb");
		if (fp==NULL)
		{
			printf("The file %s open failed !\n",buff);
		}
		for (ix=0;ix<nx;ix++)
		{
			fseek(fp,240L,1);
			fread(&record[ix*nt],sizeof(float),nt,fp);
		}
		fclose(fp);
	}	
}*/

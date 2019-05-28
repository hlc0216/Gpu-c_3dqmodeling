#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <malloc.h>

#include "array_new.h"
#include "read_write.h"
#include "CPU_function.h"

#define M 10
#define eps 2.22e-17
#define Block_Sizex 8
#define Block_Sizey 8
#define Block_Sizez 8
void modeling3d(int nx, int ny, int nz, int nt, int ntsnap, float dx, float dy, float dz, float dt, int pml, int snapflag, int sx, int sy, int sz,
                float *vp, float *epsilon, float *delta, float *source, float *wavelet, float *record, float *dlr, float *ddlr, float *dtb, float *ddtb,
                float *dfb, float *ddfb, float *c, float *c2, const char *snap_file)
{
    clock_t starttime, endtime;
    float timespent;
    starttime = clock();
    float uxMax, uyMax, uzMax, uxyzMax;
    int i, j, l, k;
    char snapname[100], snapname_S[100], snapxzname[100], snapyzname[100], snapxyname[100],
        snapSxzname[100], snapSyzname[100], snapSxyname[100];
    //allocate host memory
    float *snap = array1d((nx - 2 * pml) * (ny - 2 * pml) * (nz - 2 * pml)), *snapxz = array1d((nx - 2 * pml) * (nz - 2 * pml)),
          *snapyz = array1d((ny - 2 * pml) * (nz - 2 * pml)), *snapxy = array1d((ny - 2 * pml) * (nx - 2 * pml)),
          *snapS = array1d(nx * ny * nz), *snapSxz = array1d(nx * nz), *snapSyz = array1d(ny * nz), *snapSxy = array1d(ny * nx);
    //   *h_ux = array1d(nx * ny * nz), *h_uy = array1d(nx * ny * nz), *h_uz = array1d(nx * ny * nz), *h_u2 = array1d(nx * ny * nz);
    /******* allocate device memory *****/
    float *S = (float *)malloc(nx * ny * nz * sizeof(float)),
          *u1 = (float *)malloc(nx * ny * nz * sizeof(float)), *u2 = (float *)malloc(nx * ny * nz * sizeof(float)), *u3 = (float *)malloc(nx * ny * nz * sizeof(float)), *ux = (float *)malloc(nx * ny * nz * sizeof(float)), *uy = (float *)malloc(nx * ny * nz * sizeof(float)), *uz = (float *)malloc(nx * ny * nz * sizeof(float)), *u,
          *wl11 = (float *)malloc(pml * ny * nz * sizeof(float)), *wl12 = (float *)malloc(pml * ny * nz * sizeof(float)), *wl13 = (float *)malloc(pml * ny * nz * sizeof(float)), *wl21 = (float *)malloc(pml * ny * nz * sizeof(float)), *wl31 = (float *)malloc(pml * ny * nz * sizeof(float)), *wl32 = (float *)malloc(pml * ny * nz * sizeof(float)), *wl33 = (float *)malloc(pml * ny * nz * sizeof(float)),
          *wr11 = (float *)malloc(pml * ny * nz * sizeof(float)), *wr12 = (float *)malloc(pml * ny * nz * sizeof(float)), *wr13 = (float *)malloc(pml * ny * nz * sizeof(float)), *wr21 = (float *)malloc(pml * ny * nz * sizeof(float)), *wr31 = (float *)malloc(pml * ny * nz * sizeof(float)), *wr32 = (float *)malloc(pml * ny * nz * sizeof(float)), *wr33 = (float *)malloc(pml * ny * nz * sizeof(float)),
          *wt11 = (float *)malloc(pml * ny * nz * sizeof(float)), *wt12 = (float *)malloc(pml * ny * nz * sizeof(float)), *wt13 = (float *)malloc(pml * ny * nz * sizeof(float)), *wt21 = (float *)malloc(pml * ny * nz * sizeof(float)), *wt31 = (float *)malloc(pml * ny * nz * sizeof(float)), *wt32 = (float *)malloc(pml * ny * nz * sizeof(float)), *wt33 = (float *)malloc(pml * ny * nz * sizeof(float)),
          *wb11 = (float *)malloc(pml * ny * nz * sizeof(float)), *wb12 = (float *)malloc(pml * ny * nz * sizeof(float)), *wb13 = (float *)malloc(pml * ny * nz * sizeof(float)), *wb21 = (float *)malloc(pml * ny * nz * sizeof(float)), *wb31 = (float *)malloc(pml * ny * nz * sizeof(float)), *wb32 = (float *)malloc(pml * ny * nz * sizeof(float)), *wb33 = (float *)malloc(pml * ny * nz * sizeof(float)),
          *wf11 = (float *)malloc(pml * ny * nz * sizeof(float)), *wf12 = (float *)malloc(pml * ny * nz * sizeof(float)), *wf13 = (float *)malloc(pml * ny * nz * sizeof(float)), *wf21 = (float *)malloc(pml * ny * nz * sizeof(float)), *wf31 = (float *)malloc(pml * ny * nz * sizeof(float)), *wf32 = (float *)malloc(pml * ny * nz * sizeof(float)), *wf33 = (float *)malloc(pml * ny * nz * sizeof(float)),
          *pl1 = (float *)malloc(pml * ny * nz * sizeof(float)), *pl2 = (float *)malloc(pml * ny * nz * sizeof(float)), *pl3 = (float *)malloc(pml * ny * nz * sizeof(float)),
          *pr1 = (float *)malloc(pml * ny * nz * sizeof(float)), *pr2 = (float *)malloc(pml * ny * nz * sizeof(float)), *pr3 = (float *)malloc(pml * ny * nz * sizeof(float)),
          *pt1 = (float *)malloc(pml * ny * nz * sizeof(float)), *pt2 = (float *)malloc(pml * ny * nz * sizeof(float)), *pt3 = (float *)malloc(pml * ny * nz * sizeof(float)),
          *pb1 = (float *)malloc(pml * ny * nz * sizeof(float)), *pb2 = (float *)malloc(pml * ny * nz * sizeof(float)), *pb3 = (float *)malloc(pml * ny * nz * sizeof(float)),
          *pf1 = (float *)malloc(pml * ny * nz * sizeof(float)), *pf2 = (float *)malloc(pml * ny * nz * sizeof(float)), *pf3 = (float *)malloc(pml * ny * nz * sizeof(float)),
          *wba11 = (float *)malloc(pml * nx * nz * sizeof(float)), *wba12 = (float *)malloc(pml * nx * nz * sizeof(float)), *wba13 = (float *)malloc(pml * nx * nz * sizeof(float)),
          *wba21 = (float *)malloc(pml * nx * nz * sizeof(float)), *wba31 = (float *)malloc(pml * nx * nz * sizeof(float)), *wba32 = (float *)malloc(pml * nx * nz * sizeof(float)), *wba33 = (float *)malloc(pml * nx * nz * sizeof(float)),
          *pba1 = (float *)malloc(pml * nx * nz * sizeof(float)), *pba2 = (float *)malloc(pml * nx * nz * sizeof(float)), *pba3 = (float *)malloc(pml * nx * nz * sizeof(float));
    for (k = 0; k < nt; k++)
    {
        if (k % 100 == 0)
            printf("nt = %d\n", k);
        grad(u2, ux, uy, uz, c, nx, ny, nz, dx, dy, dz);
        uxMax = absMaxval(ux, nx, ny, nz);
        uyMax = absMaxval(uy, nx, ny, nz);
        uzMax = absMaxval(uz, nx, ny, nz);

        uxyzMax = (uxMax > uyMax) ? uxMax : uyMax;
        uxyzMax = (uxyzMax > uzMax) ? uxyzMax : uzMax; //开销很大
                                                       //打印uxyzMax
        printf("uxyzMax=%4.3f\n", uxyzMax);
        //calculating S operators
        scalar_operator(uxyzMax, ux, uy, uz, epsilon, delta, S, nx, ny, nz);
        //calculating wavefield using FD method
        wavefield_update(c, c2, dlr, ddlr, dtb, ddtb, dfb, ddfb, epsilon, delta,
                         vp, dx, dy, dz, dt, nx, ny, nz, pml, sz, ux, uy, uz, u1, u3, u2, S,
                         wl11, wl12, wl13, wl21, wl31, wl32, wl33, pl1, pl2, pl3,
                         wr11, wr12, wr13, wr21, wr31, wr32, wr33, pr1, pr2, pr3,
                         wt11, wt12, wt13, wt21, wt31, wt32, wt33, pt1, pt2, pt3,
                         wb11, wb12, wb13, wb21, wb31, wb32, wb33, pb1, pb2, pb3,
                         wf11, wf12, wf13, wf21, wf31, wf32, wf33, pf1, pf2, pf3,
                         wba11, wba12, wba13, wba21, wba31, wba32, wba33, pba1, pba2, pba3);
        addsource(source, wavelet[k], u3, nx, ny, nz);

        exchange(nx, ny, nz, pml, u1, u2, u3,
                 wl11, wl12, wl13, wl31, wl32, wl33, pl1, pl2, pl3,
                 wr11, wr12, wr13, wr31, wr32, wr33, pr1, pr2, pr3,
                 wt11, wt12, wt13, wt31, wt32, wt33, pt1, pt2, pt3,
                 wb11, wb12, wb13, wb31, wb32, wb33, pb1, pb2, pb3,
                 wf11, wf12, wf13, wf31, wf32, wf33, pf1, pf2, pf3,
                 wba11, wba12, wba13, wba31, wba32, wba33, pba1, pba2, pba3);

        for (i = pml; i < nx - pml; i++)
            for (j = pml; j < ny - pml; j++)
            {
                record[k * (nx - 2 * pml) * (ny - 2 * pml) + (i - pml) * (ny - 2 * pml) + j - pml] = u2[sz * nx * ny + i * ny + j];
            }

        if (snapflag == 1 && k % ntsnap == 0)
        {
            sprintf(snapname, "%s%d.dat", snap_file, k);
            sprintf(snapxzname, "%s_xz%d.dat", snap_file, k);
            sprintf(snapyzname, "%s_yz%d.dat", snap_file, k);
            sprintf(snapxyname, "%s_xy%d.dat", snap_file, k);
            //  	 	cudaMemcpy(snap, d_u, (nz-2*pml)*(nx-2*pml)*(ny-2*pml)*sizeof(float), cudaMemcpyDeviceToHost);
            for (i = pml; i < nz - pml; i++)
                for (j = pml; j < nx - pml; j++)
                    for (l = pml; l < ny - pml; l++)
                        snap[(i - pml) * (nx - 2 * pml) * (ny - 2 * pml) + (j - pml) * (ny - 2 * pml) + l - pml] = u2[i * nx * ny + j * ny + l];

            writefile_3d(snapname, snap, nz - 2 * pml, nx - 2 * pml, ny - 2 * pml);
            for (i = 0; i < nz - 2 * pml; i++)
                for (j = 0; j < nx - 2 * pml; j++)
                    for (l = 0; l < ny - 2 * pml; l++)
                    {
                        if (l == (ny - 2 * pml - 1) / 2)
                        {
                            snapxz[i * (nx - 2 * pml) + j] = snap[i * (nx - 2 * pml) * (ny - 2 * pml) + j * (ny - 2 * pml) + l];
                        }

                        if (j == (nx - 2 * pml - 1) / 2)
                        {
                            snapyz[i * (ny - 2 * pml) + l] = snap[i * (nx - 2 * pml) * (ny - 2 * pml) + j * (ny - 2 * pml) + l];
                        }

                        if (i == (nz - 2 * pml - 1) / 2)
                        {
                            snapxy[j * (ny - 2 * pml) + l] = snap[i * (nx - 2 * pml) * (ny - 2 * pml) + j * (ny - 2 * pml) + l];
                        }
                    }
            writefile_2d(snapxzname, snapxz, nz - 2 * pml, nx - 2 * pml);
            writefile_2d(snapyzname, snapyz, nz - 2 * pml, ny - 2 * pml);
            writefile_2d(snapxyname, snapxy, nx - 2 * pml, ny - 2 * pml);
        }
    }
    

 	free(wl11);free(wl12);free(wl13);free(wl21);
 	free(wl31);free(wl32);free(wl33);free(pl1);
 	free(pl2);free(pl3);
 	free(wr11);free(wr12);free(wr13);free(wr21);
 	free(wr31);free(wr32);free(wr33);free(pr1);
 	free(pr2);free(pr3);
 	free(wt11);free(wt12);free(wt13);free(wt21);
 	free(wt31);free(wt32);free(wt33);free(pt1);
 	free(pt2);free(pt3);
 	free(wb11);free(wb12);free(wb13);free(wb21);
 	free(wb31);free(wb32);free(wb33);free(pb1);
 	free(pb2);free(pb3);
	free(wf11);free(wf12);free(wf13);free(wf21);
 	free(wf31);free(wf32);free(wf33);free(pf1);
 	free(pf2);free(pf3);
 	free(wba11);free(wba12);free(wba13);free(wba21);
 	free(wba31);free(wba32);free(wba33);free(pba1);
 	free(pba2);free(pba3);

	free(ux); free(uy); free(uz); free(snap); free(snapxz);free(snapyz);free(snapxy);
	free(snapS);free(snapSxz);free(snapSyz);free(snapSxy);
	
	endtime = clock();
	timespent=(float)(endtime-starttime)/CLOCKS_PER_SEC;
	printf("Singshot modeling  time-assuming is %f s.\n",timespent);
}

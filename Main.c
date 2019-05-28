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
    int Nx0, Ny0, Nz0, pml, nt, hx, hy, nxshot, nyshot, dxshot, dyshot, multishot, muteflag, snapflag1, snapflag2,
        sx0, sy0, sz0, ntsnap, Nx1, Ny1, Nz1, nx, ny, nz, i, j, k, ixshot, iyshot, sx, sy, sz, sx1, sy1, sz1,
        gx0, gy0, gz0;
    float dx, dy, dz, dt, fm, t0, amp, alpha, vpmax, vpd, epsd, deld;
    char vp_file[100], eps_file[100], del_file[100], recfull_file[100], recfilexz[100], recfileyz[100], recdir_file[100],
        recmut_file[100], snap_file1[100], snap_file2[100], buff[100];
    FILE *fp = NULL;
    fp = fopen("./input/parameter.txt", "r");
    if (fp == NULL)
    {
        printf("parameter.txt is not exist !\n");
        fclose(fp);
        exit(0);
    }
    fgets(buff, 100, fp);
    fscanf(fp, "%f", &dt);
    fscanf(fp, "%d", &nt);
    fscanf(fp, "%d", &ntsnap);
    printf("dt= %f  nt=%d ntsnap=%d\n", dt, nt, ntsnap);
    fgets(buff, 100, fp);
    fgets(buff, 100, fp);
    fscanf(fp, "%d", &Nx0);
    fscanf(fp, "%d", &Ny0);
    fscanf(fp, "%d", &Nz0);
    printf("Nx0=%d ny0=%d nz0=%d\n", Nx0, Ny0, Nz0);
    fgets(buff, 100, fp);
    fgets(buff, 100, fp);
    fscanf(fp, "%f", &dx);
    fscanf(fp, "%f", &dy);
    fscanf(fp, "%f", &dz);
    printf("dx=%f,dy=%f dz=%f\n", dx, dy, dz);
    fgets(buff, 100, fp);
    fgets(buff, 100, fp);
    fscanf(fp, "%f", &fm);
    fscanf(fp, "%f", &amp);
    fscanf(fp, "%f", &alpha);
    printf("fm=%f amp=%f alpha=%f\n", fm, amp, alpha);
    fgets(buff, 100, fp);
    fgets(buff, 100, fp);
    fscanf(fp, "%d", &sx0);
    fscanf(fp, "%d", &sy0);
    fscanf(fp, "%d", &sz0);
    printf("sx0=%d sy0=%d sz0=%d\n", sx0, sy0, sz0);
    fgets(buff, 100, fp);
    fgets(buff, 100, fp);
    fscanf(fp, "%d", &gx0);
    fscanf(fp, "%d", &gy0);
    fscanf(fp, "%d", &gz0);
    printf("gx0=%d,gy0=%d gz0=%d\n", gx0, gy0, gz0);
    fgets(buff, 100, fp);
    fgets(buff, 100, fp);
    fscanf(fp, "%d", &nxshot);
    fscanf(fp, "%d", &nyshot);
    fscanf(fp, "%d", &dxshot);
    fscanf(fp, "%d", &dyshot);
    printf("nxshot=%d nyshot=%d dxshot=%d dyshot =%d\n", nxshot, nyshot, dxshot, dyshot);
    fgets(buff, 100, fp);
    fgets(buff, 100, fp);
    fscanf(fp, "%d", &hx);
    fscanf(fp, "%d", &hy);
    printf("hx=%d hy=%d\n", hx, hy);
    fgets(buff, 100, fp);
    fgets(buff, 100, fp);
    fscanf(fp, "%d", &pml);
    printf("pml=%d\n", pml);
    fgets(buff, 100, fp);
    fgets(buff, 100, fp);
    fscanf(fp, "%s", vp_file);
    fscanf(fp, "%s", eps_file);
    fscanf(fp, "%s", del_file);
    printf("vp_file=%s eps_file=%s del_file=%s\n", vp_file, eps_file, del_file);
    //fgets(buff,100,fp);fgets(buff,100,fp);
    //fscanf(fp,"%f", &vpd);fscanf(fp,"%f", &epsd);fscanf(fp,"%f", &deld);
    //printf("vpd=%f,epsd=%f deld=%f\n",vpd,epsd,deld);
    fgets(buff, 100, fp);
    fgets(buff, 100, fp);
    fscanf(fp, "%s", recfull_file);
    fscanf(fp, "%s", recdir_file);
    fscanf(fp, "%s", recmut_file);
    printf("recfull_file=%s recdir_file=%s recmut_file=%s\n", recfull_file, recdir_file, recmut_file);
    fgets(buff, 100, fp);
    fgets(buff, 100, fp);
    fscanf(fp, "%s", recfilexz);
    fscanf(fp, "%s", recfileyz);
    printf("recfilexz=%s recfileyz=%s\n", recfilexz, recfileyz);
    fgets(buff, 100, fp);
    fgets(buff, 100, fp);
    fscanf(fp, "%s", snap_file1);
    fscanf(fp, "%s", snap_file2);
    printf("snap_file=%s snap_file2=%s\n", snap_file1, snap_file2);
    fgets(buff, 100, fp);
    fgets(buff, 100, fp);
    fscanf(fp, "%d", &multishot);
    printf("multishot=%d\n", multishot);
    fgets(buff, 100, fp);
    fgets(buff, 100, fp);
    fscanf(fp, "%d", &muteflag);
    printf("muteflag=%d\n", muteflag);
    fgets(buff, 100, fp);
    fgets(buff, 100, fp);
    fscanf(fp, "%d", &snapflag2);
    printf("snapflag2=%d\n", snapflag2);
    fgets(buff, 100, fp);
    fgets(buff, 100, fp);
    fscanf(fp, "%d", &snapflag1);
    printf("snapflag1=%d\n", snapflag1);
    fclose(fp);
    printf("recfilexz=%s\n", recfilexz);
    nx = hx + 2 * pml, ny = hy + 2 * pml, nz = Nz0 + 2 * pml;
    t0 = 1.0 / fm;

    /******allocate host memory *********/
    float *vp0 = array1d(Nx0 * Ny0 * Nz0), *epsilon0 = array1d(Nx0 * Ny0 * Nz0), *delta0 = array1d(Nx0 * Ny0 * Nz0),
          *vp1 = array1d(hx * hy * Nz0), *epsilon1 = array1d(hx * hy * Nz0), *delta1 = array1d(hx * hy * Nz0),
          *vp2 = array1d(Nx0 * Ny0 * Nz0), *epsilon2 = array1d(Nx0 * Ny0 * Nz0), *delta2 = array1d(Nx0 * Ny0 * Nz0),
          *vp = array1d(nx * ny * nz), *epsilon = array1d(nx * ny * nz), *delta = array1d(nx * ny * nz),
          *vp_d = array1d(nx * ny * nz), *epsilon_d = array1d(nx * ny * nz), *delta_d = array1d(nx * ny * nz),
          *source = array1d(nx * ny * nz), *wavelet = array1d(nt),
          *record_fullwave = array1d(hx * hy * nt), *record_direct = array1d(hx * hy * nt), *record_mute = array1d(hx * hy * nt),
          *dlr = array1d(pml), *ddlr = array1d(pml), *dtb = array1d(pml), *ddtb = array1d(pml), *dfb = array1d(pml), *ddfb = array1d(pml),
          *recordxz = array1d(nt * hx), *recordyz = array1d(nt * hy);
    //float *test = array1d(nt);

    //first-order center difference cofficient
    float *c = array1d((M / 2 + 1) * (M / 2 + 1));
    fdcoff1(c);

    //second-order center difference cofficient
    float *c2 = array1d((M / 2 + 1) * (M / 2 + 1));
    fdcoff2(c2);

    //info display
    printf("\n****************** Parameters *********************\n");
    printf("ModelSize = %d * %d * %d \n", Nx0, Ny0, Nz0);
    printf("dx = %f dy = %f dz = %f nt = %d dt = %f fm = %f\n", dx, dy, dz, nt, dt, fm);
    printf("sx0 = %d sy0 = %d  sz0 = %d offsetx = %d offsety = %d nshot = %d dxshot = %d dyshot = %d\n", sx0, sy0, sz0, hx, hy, nxshot * nyshot, dxshot, dyshot);
    printf("****************************************************\n\n");
    printf("            Start to Modeling !\n\n");
    //Nx0, Ny0, Nz0:模型大小
    //dx, dy, dz,：计算区域边长，
    // dt,nt, fm：时间采样间隔，采样点数，子波频率
    //sx0, sy0, sz0：第一炮模型位置
    //hx,hy:炮位置偏移
    //nxshot ,nyshot :x方向炮数,y方向炮数
    //dxshot,dyshot :x方向炮间隔，y方向炮间隔

    if (muteflag == 1) //modeling direct-wave
    {
        printf("/************* Modeling directwave ***************/\n");
        vpmax = (1 + 2 * absMaxval(epsilon_d, nx, ny, nz)) * absMaxval(vp_d, nx, ny, nz);
        pmlcoff(pml, vpmax, dx, dy, dz, dlr, ddlr, dtb, ddtb, dfb, ddfb);
        sx1 = (hx - 1) / 2 + pml;
        sy1 = (hy - 1) / 2 + pml;
        sz1 = pml;
        Source(sx1, sy1, sz1, fm, amp, alpha, dt, dx, dy, dz, t0, nt, nx, ny, nz, source, wavelet);
        modeling3d(nx, ny, nz, nt, ntsnap, dx, dy, dz, dt, pml, snapflag1, sx1, sy1, sz1, vp_d, epsilon_d, delta_d,
                   source, wavelet, record_direct, dlr, ddlr, dtb, ddtb, dfb, ddfb, c, c2, snap_file1);
        writefile_3d(recdir_file, record_direct, nt, hx, hy);
    }
    printf("/************* Multi-shots Modeling  ***************/\n");
    //shot loop  (first-y direction(fast dimention) and second-x direction (slow dimention))
    printf("nxshort=%d", nxshot);
    for (ixshot = 0; ixshot < nxshot; ixshot++)
        for (iyshot = 0; iyshot < nyshot; iyshot++)
        {
            printf("shot=%d\n", ixshot * nyshot + iyshot);
            sx = sx0 + ixshot * dxshot;
            sy = sy0 + iyshot * dyshot;
            sz = sz0;
            //single-shot getting model-parameters
            readmodel3d(vp_file, vp1, hy, hx, Nz0, sy - (hy - 1) / 2, sx - (hx - 1) / 2, 0, Ny0, Nx0, Nz0);
            readmodel3d(eps_file, epsilon1, hy, hx, Nz0, sy - (hy - 1) / 2, sx - (hx - 1) / 2, 0, Ny0, Nx0, Nz0);
            readmodel3d(del_file, delta1, hy, hx, Nz0, sy - (hy - 1) / 2, sx - (hx - 1) / 2, 0, Ny0, Nx0, Nz0);

            extend_model(vp1, vp, nx, ny, nz, pml);
            extend_model(epsilon1, epsilon, nx, ny, nz, pml);
            extend_model(delta1, delta, nx, ny, nz, pml);

            //PML ATTENUATION COEFFICIENT
            vpmax = (1 + 2 * absMaxval(epsilon, nx, ny, nz)) * absMaxval(vp, nx, ny, nz);
            pmlcoff(pml, vpmax, dx, dy, dz, dlr, ddlr, dtb, ddtb, dfb, ddfb);
            sx1 = (hx - 1) / 2 + pml;
            sy1 = (hy - 1) / 2 + pml;
            sz1 = pml;
            Source(sx1, sy1, sz1, fm, amp, alpha, dt, dx, dy, dz, t0, nt, nx, ny, nz, source, wavelet);
            modeling3d(nx, ny, nz, nt, ntsnap, dx, dy, dz, dt, pml, snapflag2, sx1, sy1, sz1, vp, epsilon, delta,
                       source, wavelet, record_fullwave, dlr, ddlr, dtb, ddtb, dfb, ddfb, c, c2, snap_file2);
            // Mute directwave
            for (i = 0; i < hx; i++)
            {
                for (j = 0; j < hy; j++)
                    {
                        for (k = 0; k < nt; k++)
                        {
                           record_mute[k * hx * hy + i * hy + j] = record_fullwave[k * hx * hy + i * hy + j] - record_direct[k * hx * hy + i * hy + j];
                        }
                    }
            }
                
            writefile2_3d(recfull_file, record_fullwave, nt, hx, hy);
            writefile2_3d(recmut_file, record_mute, nt, hx, hy);
            
            if (ixshot == 0 && iyshot == 0)
            {
                for (i = 0; i < hx; i++)
                    for (j = 0; j < hy; j++)
                        for (k = 0; k < nt; k++)
                        {
                            if (j == sy1 - pml)
                                recordxz[k * hx + i] = record_mute[k * hx * hy + i * hy + j];

                            if (i == sx1 - pml)
                                recordyz[k * hy + j] = record_mute[k * hx * hy + i * hy + j];
                        }
                writefile_2d(recfilexz, recordxz, nt, hx);
                writefile_2d(recfileyz, recordyz, nt, hy);

                for (i = 0; i < hx; i++)
                    for (j = 0; j < hy; j++)
                        for (k = 0; k < nt; k++)
                        {
                            if (j == sy1 - pml)
                                recordxz[k * hx + i] = record_fullwave[k * hx * hy + i * hy + j];

                            if (i == sx1 - pml)
                                recordyz[k * hy + j] = record_fullwave[k * hx * hy + i * hy + j];
                        }
                writefile_2d(recfilexz, recordxz, nt, hx);
                writefile_2d(recfileyz, recordyz, nt, hy);
            }
        }
    //free memory
    free(vp);free(epsilon); free(delta); free(vp0); free(epsilon0);
    free(delta0);free(vp1); free(epsilon1); free(delta1); free(vp2);
    free(epsilon2);free(delta2);free(vp_d);free(epsilon_d);free(delta_d); free(source);
    free(wavelet); free(dlr); free(ddlr); free(dtb); free(ddtb); free(dfb); free(ddfb); 
    free(c);free(c2);free(record_mute); free(record_fullwave);free(record_direct);

    return 0;
}
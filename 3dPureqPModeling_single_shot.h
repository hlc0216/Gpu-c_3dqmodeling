#ifndef QPMODELING3D_H
#define QPMODELING3D_H

void modeling3d(int nx, int ny, int nz, int nt, int ntsnap, float dx, float dy, float dz, float dt, int pml, int snapflag, int sx, int sy, int sz, 
                float *vp, float *epsilon, float *delta, float *source, float *wavelet, float *record, float *dlr,float *ddlr, float *dtb, float *ddtb, 
				float *dfb, float *ddfb, float *c, float *c2, const char *snap_file);


#endif

#ifndef READ_WRITE_H
#define READ_WRITE_H

void readfile_2d(const char *filename, float *a, int nz, int nx);
void readfile_3d(const char *filename, float *a, int nz, int nx, int ny);
void writefile_2d(const char *filename, float *a, int nz, int nx);
void writefile_3d(const char *filename, float *a, int nz, int nx, int ny);
void writefile2_2d(const char *filename, float *a, int nz, int nx);
void writefile2_3d(const char *filename, float *a, int nz, int nx, int ny);


#endif
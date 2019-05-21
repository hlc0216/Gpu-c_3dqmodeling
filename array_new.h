#ifndef ARRAY_NEW_H
#define ARRAY_NEW_H

float *array1d(int x);
float **array2d(int z,int x);
float ***array3d(int z,int x,int y);
void  free2d(float **p,int z);
void  free3d(float ***p,int z,int x);

#endif
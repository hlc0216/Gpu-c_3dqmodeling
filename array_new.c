/******** 2D:  CreatAray  follow  x as  fast dimension, then z. eg: a[nz][nx] **********/
/*********3D:  CreatAray  follow  y as  first fast dimension, x as second fast dimension , then z.  eg: a[nz][nx][ny] **********/

#include <malloc.h>
#include "array_new.h"



float *array1d(int x)
{
  float *p;
  int i;
  p=(float*)malloc(sizeof(float)*x);
  for(i=0;i<x;i++)
     p[i]=0.0;
  return p;
}

float **array2d(int z,int x)
{
   float **p;
   int i,j;
   p=(float**)malloc(sizeof(float*)*z);
   for(i=0;i<z;i++)
      p[i]=(float*)malloc(sizeof(float)*x);
   for(i=0;i<z;i++)
      for(j=0;j<x;j++)
         p[i][j]=0.0;
   return p;
}

float ***array3d(int z,int x,int y)
{
    float ***p;
    int i,j,k;
    p=(float ***)malloc(sizeof(float**)*z);
    for(i=0;i<z;i++)
    {
        p[i]=(float **)malloc(sizeof(float*)*x);
        for(j=0;j<x;j++)
        {
            p[i][j] =(float*)malloc(sizeof(float)*y);
        }
    }
    for(i=0;i<z;i++)
      for(j=0;j<x;j++)
        for(k=0;k<y;k++)
           p[i][j][k]=0.0;
    return p;
}

void free2d(float **p,int z)
{
	int i;
	for (i=0; i<z; i++)
		free(p[i]);
	free(p);
}

void  free3d(float ***p,int z,int x)
{
	int i,j;
	for(i=0;i<z;i++)
	{
		for(j=0;j<x;j++)
			free(p[i][j]);
		free(p[i]);
	}
	free(p);
}

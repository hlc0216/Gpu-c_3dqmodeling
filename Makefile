CUDA_INTSTALL_PATH = /usr/local/cuda-9.0

GCC = g++
NVCC = nvcc
#LFLAGS = -lm
CFLAgS = -Wall -O3

# CULDFLAGS = -L/usr/local/cuda-9.0/lib64
# LIB = -lcudart #-lcurand 

CFILES = 3dPureqPModeling1.0.c array_new.c CPU_function.c read_write.c
CUFILES = 3dPureqPModeling_single_shot.cu GPU_kernel.cu
OBJECTS = 3dPureqPModeling1.0.o array_new.o CPU_function.o read_write.o 3dPureqPModeling_single_shot.o GPU_kernel.o  
EXENAME = 3dModeling

all:
	$(GCC) -c $(CFILES)  
	$(NVCC) -c $(CUFILES)
	$(NVCC) $(OBJECTS) -o $(EXENAME) 
	rm -f *.o *~
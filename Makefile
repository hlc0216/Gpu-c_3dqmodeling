CUDA_INTSTALL_PATH = /usr/local/cuda-9.0

GCC = g++

#LFLAGS = -lm
CFLAgS = -Wall -O3

CFILES = Main.c array_new.c CPU_function.c read_write.c modeling3D.c
OBJECTS = Main.o array_new.o CPU_function.o read_write.o modeling3D.o  
EXENAME = 3dModeling

all:
	$(GCC) -c $(CFILES)  
	$(GCC) $(OBJECTS) -o $(EXENAME) 
	rm -f *.o *~
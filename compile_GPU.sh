#!/bin/bash
#GPU -version
rm 3dqpmodeling
g++ -c -g array_new.c -o array_new.o 
g++ -c -g read_write.c -o read_write.o 
g++ -c -g CPU_function.c -o CPU_function.o 
g++ -c -g 3dPureqPModeling1.0.c -o 3dPureqPModeling1.0.o
nvcc -c -g GPU_kernel.cu -o GPU_kernel.o 
nvcc -c -g 3dPureqPModeling_single_shot.cu  -o 3dPureqPModeling_single_shot.o

nvcc -o 3dqpmodeling 3dPureqPModeling1.0.o 3dPureqPModeling_single_shot.o GPU_kernel.o CPU_function.o read_write.o array_new.o





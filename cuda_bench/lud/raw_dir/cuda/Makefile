CC = gcc

CUDA_DIR = /usr/local/cuda
CUDA_LIB_DIR := $(CUDA_DIR)/lib
SDK_DIR = /usr/local/cuda/samples/
CUDA_SDK_PATH := $(SDK_DIR)
CC_FLAG = -lmpfr -lgmp

NVCC = $(CUDA_DIR)/bin/nvcc

SRC_DD = lud_double.cu
EXE_DD = lud_double

DEFS += \
		-DGPU_TIMER \
		$(SPACE)

NVCCFLAGS += -I../common \
			 -O3 \
			 -use_fast_math \
			 -arch=sm_35 \
			 -lm \
			 $(SPACE)

CFLAGS += -I../common \
					-I/usr/include/cuda \
		  -O3 \
		  -Wall \
		  $(SPACE)

# Add source files here
EXECUTABLE  := lud_cuda
# Cuda source files (compiled with cudacc)
CUFILES     := lud_kernel.cu
# C/C++ source files (compiled with gcc / c++)
CCFILES     := lud.c lud_cuda.c ../common/common.c

OBJS = ../common/common.o lud.o lud_kernel.o

.PHONY: all clean 
all : $(EXECUTABLE) $(EXE_DD)

$(EXE_DD): $(SRC_DD)
	$(NVCC) -Xptxas -v -O3 --gpu-architecture=compute_35 --gpu-code=compute_35 lud_double.cu -o lud_double -I$(CUDA_SDK_PATH)/common/inc  -L$(CUDA_SDK_PATH)/lib $(CUTIL_LIB) $(CC_FLAG)

.c.o : 
	$(NVCC) $(KERNEL_DIM) $(NVCCFLAGS) $(DEFS) -o $@ -c $<

%.o:	%.cu 
	$(NVCC) $(KERNEL_DIM) $(NVCCFLAGS) $(DEFS) -o $@ -c $<


$(EXECUTABLE) : $(OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@  $? $(CC_FLAG)

clean:
	rm -f $(EXECUTABLE) $(OBJS) *.linkinfo

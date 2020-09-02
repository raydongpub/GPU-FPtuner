#include ../../common/make.config

CUDA_DIR = /usr/local/cuda
CUDA_LIB_DIR := $(CUDA_DIR)/lib
SDK_DIR = /usr/local/cuda/samples/
CUDA_SDK_PATH := $(SDK_DIR)

CC := $(CUDA_DIR)/bin/nvcc

INCLUDE := $(CUDA_DIR)/include

CC_FLAG = -lmpfr -lgmp

SRC = gaussian.cu
EXE = gaussian

SRC_DD = gaussian_dd.cu
EXE_DD = gaussian_dd

SRC_D = gaussian_double.cu
EXE_D = gaussian_double


all: $(EXE_D) $(EXE_DD)

$(EXE_D): $(SRC_D)
	$(CC) -Xptxas -v -O3 -w  --gpu-architecture=compute_35 --gpu-code=compute_35 gaussian_double.cu -o gaussian_double -I$(CUDA_SDK_PATH)/common/inc  -L$(CUDA_SDK_PATH)/lib $(CUTIL_LIB) $(CC_FLAG)

$(EXE_DD): $(SRC_DD)
	nvcc -Xptxas -v -O3 -w --gpu-architecture=compute_35 --gpu-code=compute_35 gaussian_dd.cu -o gaussian_dd -I$(CUDA_SDK_PATH)/common/inc  -L$(CUDA_SDK_PATH)/lib $(CUTIL_LIB) $(CC_FLAG)

$(EXE): $(SRC)
	$(CC) $(KERNEL_DIM) $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) $(CC_FLAG) 

clean:
	rm -f gaussian
	rm -f gaussian_double

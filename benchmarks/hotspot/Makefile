CUDA_DIR = /usr/local/cuda
CUDA_LIB_DIR := $(CUDA_DIR)/lib
SDK_DIR = /usr/local/cuda/samples
CUDA_SDK_PATH := $(SDK_DIR)
CC_FLAG = -lm -lmpfr -lgmp -lqd

CC := $(CUDA_DIR)/bin/nvcc

C_FLAG =

INCLUDE := $(CUDA_DIR)/include

SRC = hotspot.cu

EXE = hotspot

SRC_DD = hotspot_dd.cu
EXE_DD = hotspot_dd

SRC_D = hotspot_double.cu
EXE_D = hotspot_double

all: $(EXE_DD) $(EXE_D)

$(EXE_D): $(SRC_D)
	nvcc $(C_FLAG) -Xptxas="-v" -O3 -w --gpu-architecture=compute_35 --gpu-code=compute_35 hotspot_double.cu -o hotspot_double -I$(CUDA_SDK_PATH)/common/inc  -L$(CUDA_SDK_PATH)/lib $(CC_FLAG)

$(EXE_DD): $(SRC_DD)
	nvcc $(C_FLAG) -Xptxas="-v" -O3 -w --gpu-architecture=compute_35 --gpu-code=compute_35 hotspot_dd.cu -o hotspot_dd -I$(CUDA_SDK_PATH)/common/inc  -L$(CUDA_SDK_PATH)/lib $(CC_FLAG)

$(EXE): $(SRC)
	$(CC) $(KERNEL_DIM) $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) $(CC_FLAG)

enum: $(SRC)
	$(CC) $(KERNEL_DIM) -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) 

debug: $(SRC)
	$(CC) $(KERNEL_DIM) -g $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) 

debugenum: $(SRC)
	$(CC) $(KERNEL_DIM) -g -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) 

clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo result.txt $(EXE_D) $(EXE_DD)

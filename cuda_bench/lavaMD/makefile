
# Example
# target: dependencies
	# command 1
	# command 2
          # .
          # .
          # .
	# command n
ifdef OUTPUT
override OUTPUT = -DOUTPUT
endif
CUDA_DIR = /usr/local/cuda
CUDA_LIB_DIR := $(CUDA_DIR)/lib
SDK_DIR = /usr/local/cuda/samples
CUDA_SDK_PATH := $(SDK_DIR)
CC_FLAG = -lm -lqd -lmpfr -lgmp

C_FLAG =

C_C = gcc
OMP_LIB = -lgomp
OMP_FLAG = -fopenmp

CUD_C = $(CUDA_DIR)/bin/nvcc
# OMP_FLAG = 	-Xcompiler paste_one_here
CUDA_FLAG = -arch sm_35


SRC_DD = lava_dd.cu
EXE_DD = lava_dd
SRC_D = lava_double.cu
EXE_D = lava_double

all: $(EXE_D) $(EXE_DD) 
#a.out


$(EXE_DD): $(SRC_DD)
	$(CUD_C) $(C_FLAG) -Xptxas -v -w -O3 --gpu-architecture=compute_35 --gpu-code=compute_35 lava_dd.cu -o lava_dd -I$(CUDA_SDK_PATH)/common/inc  -L$(CUDA_SDK_PATH)/lib $(CUTIL_LIB) $(CC_FLAG)
$(EXE_D): $(SRC_D)
	$(CUD_C) $(C_FLAG) -Xptxas -v -w -O3 --gpu-architecture=compute_35 --gpu-code=compute_35 lava_double.cu -o lava_double -I$(CUDA_SDK_PATH)/common/inc  -L$(CUDA_SDK_PATH)/lib $(CUTIL_LIB) $(CC_FLAG)

# link objects (binaries) together
a.out:		main.o \
			./kernel/kernel_gpu_cuda_wrapper.o \
			./util/num/num.o \
			./util/timer/timer.o \
			./util/device/device.o
	$(C_C)	$(KERNEL_DIM) main.o \
			./kernel/kernel_gpu_cuda_wrapper.o \
			./util/num/num.o \
			./util/timer/timer.o \
			./util/device/device.o \
			-lm \
			-L/usr/local/cuda/lib64 \
			-lcuda -lcudart \
			$(OMP_LIB) \
			-o lavaMD $(CC_FLAG)

# compile function files into objects (binaries)
main.o:		main.h \
			main.c \
			./kernel/kernel_gpu_cuda_wrapper.h \
			./kernel/kernel_gpu_cuda_wrapper.cu \
			./util/num/num.h \
			./util/num/num.c \
			./util/timer/timer.h \
			./util/timer/timer.c \
			./util/device/device.h \
			./util/device/device.cu
	$(C_C)	$(KERNEL_DIM) $(OUTPUT) main.c \
			-c \
			-o main.o \
			-O3

./kernel/kernel_gpu_cuda_wrapper.o:	./kernel/kernel_gpu_cuda_wrapper.h \
									./kernel/kernel_gpu_cuda_wrapper.cu
	$(CUD_C) $(KERNEL_DIM)						./kernel/kernel_gpu_cuda_wrapper.cu \
									-c \
									-o ./kernel/kernel_gpu_cuda_wrapper.o \
									-O3 \
									$(CUDA_FLAG)

./util/num/num.o:	./util/num/num.h \
					./util/num/num.c
	$(C_C)			./util/num/num.c \
					-c \
					-o ./util/num/num.o \
					-O3

./util/timer/timer.o:	./util/timer/timer.h \
						./util/timer/timer.c
	$(C_C)				./util/timer/timer.c \
						-c \
						-o ./util/timer/timer.o \
						-O3

./util/device/device.o:	./util/device/device.h \
						./util/device/device.cu
	$(CUD_C)			./util/device/device.cu \
						-c \
						-o ./util/device/device.o \
						-O3

# delete all object and executable files
clean:
	rm	*.o \
		lavaMD $(EXE_D) $(EXE_DD)
#		./kernel/*.o \
#		./util/num/*.o \
#		./util/timer/*.o \
#		./util/device/*.o \

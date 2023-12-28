# ----- Make Macros -----

CC = CC -target-accel=nvidia80 -fopenmp

NVCC = nvcc
NVCCFLAGS = -lineinfo -O3 -std=c++14 -gencode arch=compute_80,code=sm_80 -ccbin=CC -Xcompiler -fopenmp -Xptxas="-v" 

LD_FLAGS = -L${NCCL_DIR}/lib -lnccl

TARGETS = pyComm
OBJECTS = pyComm.cpp 

# ----- Make Rules -----

all:	$(TARGETS)

%.o : %.cpp
	${CC} $< -c -o $@

%.o : %.cu
	${NVCC} ${NVCCFLAGS} $< -c -o $@

pyComm: $(OBJECTS)
	${CC} -O3 -Wall -shared -std=c++11 -fPIC $$(python3 -m pybind11 --includes) $(OBJECTS) -o pyComm.so

clean:
	rm -f $(TARGETS) *.o *.bin core *.html *.xml

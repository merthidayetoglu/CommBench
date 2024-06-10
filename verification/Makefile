# ----- Make Macros -----

GASNET_DIR = /global/homes/m/merth/CommBench/gasnet/release

CC = CC -target-accel=nvidia80 -fopenmp -I${GASNET_DIR}/include -I${GASNET_DIR}/include/ofi-conduit/

NVCC = nvcc
NVCCFLAGS = -lineinfo -O3 -std=c++14 -gencode arch=compute_80,code=sm_80 -ccbin=CC -Xcompiler -fopenmp -Xptxas="-v"

LD_FLAGS = -L${NCCL_DIR}/lib -lnccl -L${GASNET_DIR}/lib -lgasnet-ofi-par -L/opt/cray/libfabric/1.15.2.0/lib64 -lfabric -L/opt/cray/pe/pmi/default/lib/ -lpmi -lhwloc -lhugetlbfs
# LD_FLAGS = -lnccl

TARGETS = CommBench
OBJECTS = main.o 

# ----- Make Rules -----

all:	$(TARGETS)

%.o : %.cpp
	${CC} $< -c -o $@

%.o : %.cu
	${NVCC} ${NVCCFLAGS} $< -c -o $@

CommBench: $(OBJECTS)
	$(CC) -o $@ $(OBJECTS) $(LD_FLAGS)

clean:
	rm -f $(TARGETS) *.o *.o.* *.txt *.bin core *.html *.xml

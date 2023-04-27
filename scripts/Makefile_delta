# ----- Make Macros -----

NVCC = nvcc
NVCCFLAGS = -lineinfo -O3 -std=c++14 -gencode arch=compute_80,code=sm_80 -ccbin=mpicxx -Xcompiler -fopenmp -Xptxas="-v" 

LD_FLAGS = -ccbin=mpicxx -Xcompiler -fopenmp -lnccl

TARGETS = CommBench
OBJECTS = main.o 

# ----- Make Rules -----

all:	$(TARGETS)

%.o : %.cpp
	${NVCC} ${NVCCFLAGS} $< -c -o $@

CommBench: $(OBJECTS)
	$(NVCC) -o $@ $(OBJECTS) $(LD_FLAGS)

clean:
	rm -f $(TARGETS) *.o *.o.* *.txt *.bin core *.html *.xml

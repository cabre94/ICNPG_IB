APPS=tabulacion

N ?= 1024 

all: ${APPS}

%: %.cu
	nvcc -o $@ $<
clean:
	rm -f ${APPS}

submit: clean all
	qsub jobGPU $(N)

submitwatch: clean all
	qsub jobGPU $(N); watch qstat


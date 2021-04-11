APPS=hello

all: ${APPS}

%: %.cu
	nvcc -O2 -arch=sm_20 -o $@ $<
clean:
	rm -f ${APPS}

run:	all
	./hello

submit:	all
	qsub jobGPU

submitwatch:	all
	qsub jobGPU; watch qstat

SHELL:=/bin/bash	# Esto es para poder correr el comando module

APPS=queplacasoy

all: ${APPS}

%: %.cu
	# Si lo corre en una compu personal deberia sacar estas dos lineas
	. /usr/share/Modules/init/bash
	module load cuda; \
	nvcc -o $@ $<
	
clean:
	rm -f ${APPS}

run:	all
	./queplacasoy

submit:	all
	qsub jobGPU; watch qstat

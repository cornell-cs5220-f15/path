#
# To build with a different compiler / on a different platform, use
#     make PLATFORM=xxx
#
# where xxx is
#     icc = Intel compilers
#     gcc = GNU compilers
#     clang = Clang compiler (OS X default)
#
# Or create a Makefile.in.xxx of your own!
#

PLATFORM=icc
include Makefile.in.$(PLATFORM)

# === Variables

SRCS   = $(shell ls *-{omp,mpi,hybrid}.c)
EXES   = $(SRCS:.c=.x)
RUNS   = $(addprefix run-,$(basename $(EXES)))
AMPLS  = $(addprefix ampl-,$(basename $(EXES)))
SWEEPS = $(addprefix sweep-,$(basename $(EXES)))
SPRAYS = $(addprefix spray-,$(basename $(EXES)))
OBJS   = mt19937p.o

SWEEPS := $(filter-out sweep-fw-mpi, $(SWEEPS))
SWEEPS := $(filter-out sweep-fw-hybrid, $(SWEEPS))
SPRAYS := $(filter-out spray-fw-mpi, $(SPRAYS))
SPRAYS := $(filter-out spray-fw-hybrid, $(SPRAYS))

N = 2400 # -n value passed to main program
P = 24   # -n value passed to mpirun
SWEEP_MIN = 480
SWEEP_MAX = 2400
SWEEP_STEP = 480

# === Defaults

.PHONY: default all
default: all
all:   $(EXES)
run:   $(RUNS)
ampl:  $(AMPLS)
sweep: $(SWEEPS)
spray: $(SPRAYS)

# === Executables
#
# We have many algorithms (e.g. Floyd-Warshal and repeated squares) and many
# forms of parallelization (e.g. OpenMP, OpenMPI, hybrid). This leads to a lot
# of different combinations and a lot of different executables. Instead of
# having a set of Makefile rules for each executable, we can use some Makefile
# tricks to reduce the number of rules.

# omp
%-omp.x: %-omp.o $(OBJS)
	$(CC) $(OMP_CFLAGS) $^ -o $@

%-omp.o: %-omp.c
	$(CC) -c $(OMP_CFLAGS) $<

run-%-omp: %-omp.x
	qsub omp.pbs -N $*-omp "-vEXE=$*-omp.x,N=$(N)"

sweep-%-omp: %-omp.x
	qsub omp-sweep.pbs -N $*-omp "-vSWEEP_MIN=$(SWEEP_MIN),SWEEP_MAX=$(SWEEP_MAX),SWEEP_STEP=$(SWEEP_STEP),EXE=$*-omp.x,N=$$i";

spray-%-omp: %-omp.x
	for (( i = $(SWEEP_MIN); i <= $(SWEEP_MAX); i += $(SWEEP_STEP) )); do \
		qsub omp.pbs -N $*-omp "-vEXE=$*-omp.x,N=$$i"; \
	done \

ampl-%-omp: %-omp.x
	qsub omp.pbs -N $*-omp "-vEXE=$*-omp.x,N=$(N),AMPL="

# hybrid
%-hybrid.x: %-hybrid.o $(OBJS)
	$(MPICC) $(HYBRID_CFLAGS) $^ -o $@

%-hybrid.o: %-hybrid.c
	$(MPICC) -c $(HYBRID_CFLAGS) $<

run-%-hybrid: %-hybrid.x
	qsub hybrid.pbs -N $*-hybrid "-vP=$(P),EXE=$*-hybrid.x,N=$(N)"

sweep-%-hybrid: %-hybrid.x
	qsub hybrid-sweep.pbs -N $*-hybrid "-vSWEEP_MIN=$(SWEEP_MIN),SWEEP_MAX=$(SWEEP_MAX),SWEEP_STEP=$(SWEEP_STEP),P=$(P),EXE=$*-hybrid.x,N=$$i";

spray-%-hybrid: %-hybrid.x
	for (( i = $(SWEEP_MIN); i <= $(SWEEP_MAX); i += $(SWEEP_STEP) )); do \
		qsub hybrid.pbs -N $*-hybrid "-vP=$(P),EXE=$*-hybrid.x,N=$$i"; \
	done \

ampl-%-hybrid: %-hybrid.x
	qsub hybrid.pbs -N $*-hybrid "-vP=$(P),EXE=$*-hybrid.x,N=$(N),AMPL="

# mpi
%-mpi.x: %-mpi.o $(OBJS)
	$(MPICC) $(MPI_CFLAGS) $^ -o $@

%-mpi.o: %-mpi.c
	$(MPICC) -c $(MPI_CFLAGS) $<

run-%-mpi: %-mpi.x
	qsub mpi.pbs -N $*-mpi "-vP=$(P),EXE=$*-mpi.x,N=$(N)"

sweep-%-mpi: %-mpi.x
	qsub mpi-sweep.pbs -N $*-mpi "-vSWEEP_MIN=$(SWEEP_MIN),SWEEP_MAX=$(SWEEP_MAX),SWEEP_STEP=$(SWEEP_STEP),P=$(P),EXE=$*-mpi.x,N=$$i";

spray-%-mpi: %-mpi.x
	for (( i = $(SWEEP_MIN); i <= $(SWEEP_MAX); i += $(SWEEP_STEP) )); do \
		qsub mpi.pbs -N $*-mpi "-vP=$(P),EXE=$*-mpi.x,N=$$i"; \
	done \

ampl-%-mpi: %-mpi.x
	qsub mpi.pbs -N $*-mpi "-vP=$(P),EXE=$*-mpi.x,N=$(N),AMPL="

# generic
mtp.o: mt19937p.c
	$(CC) -mmic -c $(PHI_FLAGS) $< -o $@

%.o: %.c
	$(CC) -c $(CFLAGS) $<

# phi compiling (of hybrid)
phi: phi.MIC

phi.MIC: path-mic.o mtp.o
	$(MPICC) -mmic $(PHI_FLAGS) $^ -o $@

path-mic.o: path-mpi-omp.c
	$(MPICC) -mmic -c $(PHI_FLAGS) $< -o $@

# checking and plotting
.PHONY: check csv plot
csv:
	python results/pbs_to_csv.py *.o[1-9]*

check: csv
	python results/check_results.py rs-omp.csv *.csv

plot: csv
	python results/plot.py *.csv

# === Miscellaneous

# Watch the status of your qsubbed jobs in reverse chronological order.
.PHONY: watch
watch:
	watch -d -n 1 'qstat | tac'

# http://blog.jgc.org/2015/04/the-one-line-you-should-add-to-every.html
print-%:
	@echo $*=$($*)

# === Documentation

main.pdf: README.md path.md
	pandoc $^ -o $@

path.md: path.c
	ldoc -o $@ $^

# === Cleanup and tarball

.PHONY: clean
clean:
	rm -f *.o*
	rm -f *.x
	rm -f *.csv
	rm -rf *.x-ampl

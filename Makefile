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

SRCS = $(shell ls *-{omp,mpi,hybrid}.c)
EXES = $(SRCS:.c=.x)
OBJS = mt19937p.o

# === Defaults

.PHONY: default all
default: all
all: $(EXES)

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

# hybrid
%-hybrid.x: %-hybrid.o $(OBJS)
	$(MPICC) $(HYBRID_CFLAGS) $^ -o $@

%-hybrid.o: %-hybrid.c
	$(MPICC) -c $(HYBRID_CFLAGS) $<

# mpi
%-mpi.x: %-mpi.o $(OBJS)
	$(MPICC) $(MPI_CFLAGS) $^ -o $@

%-mpi.o: %-mpi.c
	$(MPICC) -c $(MPI_CFLAGS) $<

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
	rm -f *.o
	rm -f *.x

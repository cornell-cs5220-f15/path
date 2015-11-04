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

.PHONY: omp mpi hybrid clean realclean


# === Executables

# #### Floyd-Warshall Algorithm
fwomp: fw-omp.x

fw-omp.x: fw-omp.o mt19937p.o
	$(CC) $(OMP_CFLAGS) $^ -o $@

fw-omp.o: fw-omp.c
	$(CC) -c $(OMP_CFLAGS) $<

fwmpi: fw-mpi.x

fw-mpi.x: fw-mpi.o mt19937p.o
	$(MPICC) $(OMP_CFLAGS) $^ -o $@

fw-mpi.o: fw-mpi.c
	$(MPICC) -c $(OMP_CFLAGS) $<


# phi compiling (of hybrid)
phi: phi.MIC

phi.MIC: path-mic.o mtp.o
	$(MPICC) -mmic $(PHI_FLAGS) $^ -o $@

path-mic.o: path-mpi-omp.c
	$(MPICC) -mmic -c $(PHI_FLAGS) $< -o $@

mtp.o: mt19937p.c
	$(CC) -mmic -c $(PHI_FLAGS) $< -o $@

#hybrid MPI-OMP compiling
hybrid: hybrid.x

hybrid.x: path-mpi-omp.o mt19937p.o
	$(MPICC) $(OMP_CFLAGS) $^ -o $@

path-mpi-omp.o: path-mpi-omp.c
	$(MPICC) -c $(OMP_CFLAGS) $<

#OMP Compiling
omp: omp.x

omp.x: path.o mt19937p.o
	$(CC) $(OMP_CFLAGS) $^ -o $@

path.o: path.c
	$(CC) -c $(OMP_CFLAGS) $<

#MPI Compiling
mpi: mpi.x

mpi.x: path-mpi.o mt19937p.o
	$(MPICC) $(MPI_CFLAGS) $^ -o $@

path-mpi.o: path-mpi.c
	$(MPICC) -c $(MPI_CFLAGS) $<

# General .o compiling
%.o: %.c
	$(CC) -c $(CFLAGS) $<


# === Documentation

main.pdf: README.md path.md
	pandoc $^ -o $@

path.md: path.c
	ldoc -o $@ $^


# === Cleanup and tarball

clean:
	rm -f *.o

realclean: clean
	rm -f path.x path-mpi.x path.md main.pdf

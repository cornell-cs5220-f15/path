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

.PHONY: exe clean realclean


# === Executables

exe: path.x path_original.x path_offload.x path_cannon.x

path.x: path.o mt19937p.o
	$(CC) $(OMP_CFLAGS) $^ -o $@

path.o: path.c
	$(CC) -c $(OMP_CFLAGS) $<

path_original.x: path_original.o mt19937p.o
	$(CC) $(OMP_CFLAGS) $^ -o $@

path_original.o: path_original.c
	$(CC) -c $(OMP_CFLAGS) $<

path_offload.x: path_offload.o mt19937p.o
	$(CC) $(OMP_CFLAGS) $(OFFLOADFLAGS) $^ -o $@

path_offload.o: path_offload.c
	$(CC) -c $(OMP_CFLAGS) $(OFFLOADFLAGS) $<

path_mpi.x: path_mpi.o mt19937p.o
	$(MPICC) $(MPI_CFLAGS) $^ -o $@

path_mpi.o: path_mpi.c
	$(MPICC) -c $(MPI_CFLAGS) $<
	
path_cannon.x: path_cannon.o mt19937p.o
	$(MPICC) $(MPI_CFLAGS) $^ -o $@

path_cannon.o: path_cannon.c
	$(MPICC) -c $(MPI_CFLAGS) $<

%.o: %.c
	$(CC) -c $(CFLAGS) $(OFFLOADFLAGS) $<


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

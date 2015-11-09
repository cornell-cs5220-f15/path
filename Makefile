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

exe: path.x path-mpi-node.x path-mpi-device.x

path.x: path.o mt19937p.o
	$(CC) $(OMP_CFLAGS) $^ -o $@

path.o: path.c
	$(CC) -c $(OMP_CFLAGS) $<

path-mpi-node.x: path-mpi-node.o mt19937p.o
	$(MPICC) $(MPI_CFLAGS) $^ -o $@

path-mpi-node.o: path-mpi.c
	$(MPICC) -c $(MPI_CFLAGS) -D_PARALLEL_NODE -o $@ $<

path-mpi-device.x: path-mpi-device.o mt19937p.o
	$(MPICC) $(MPI_CFLAGS) $^ -o $@

path-mpi-device.o: path-mpi.c
	$(MPICC) -c $(MPI_CFLAGS) -D_PARALLEL_DEVICE -o $@ $<

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
	rm -f path.o*
	rm -f path-mpi-node.o*
	rm -f path-mpi-device.o*
	rm -f *.x

realclean: clean
	rm -f path.x path-mpi-node.x path-mpi-device.x path.md main.pdf

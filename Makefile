PLATFORM=icc
include Makefile.in.$(PLATFORM)

# === Executables

.PHONY: exe exe-base exe-blocked exe-tiled

exe: path.x

exe-base: path-base.x

exe-blocked: path-blocked.x

exe-tiled: path-tiled.x

path.x: path.o mt19937p.o
	$(CC) $(OMP_CFLAGS) $^ -o $@

path-base.x: path-base.o mt19937p.o
	$(CC) $(OMP_CFLAGS) $^ -o $@

path-blocked.x: path-blocked.o mt19937p.o
	$(CC) $(OMP_CFLAGS) $^ -o $@

path-tiled.x: path-tiled.o mt19937p.o
	$(CC) $(OMP_CFLAGS) $^ -o $@

path.o: path.c
	$(CC) -c $(OMP_CFLAGS) $<

path-base.o: path-base.c
	$(CC) -c $(OMP_CFLAGS) $<

path-blocked.o: path-blocked.c
	$(CC) -c $(OMP_CFLAGS) $<

path-mpi.x: path-mpi.o mt19937p.o
	$(MPICC) $(MPI_CFLAGS) $^ -o $@

path-mpi.o: path-mpi.c
	$(MPICC) -c $(MPI_CFLAGS) $<

%.o: %.c
	$(CC) -c $(CFLAGS) $<


# === Profiling

.PHONY: maqao-cqa maqao-perf scan-build vtune-report

maqao-cqa: path.x
	( module load maqao ; \
	  maqao cqa ./path.x fct=main uarch=HASWELL > maqao-cqa.report )

maqao-perf: path.x
	( module load maqao ; \
	  maqao perf ./path.x fct=main uarch=HASWELL > maqao-perf.report )

scan-build:
	( module load llvm-analyzer ; \
	  scan-build -v --use-analyzer=/share/apps/llvm-3.7.0/bin/clang make )

vtune-report:
	amplxe-cl -R hotspots -report-output vtune-report.csv -format csv -csv-delimiter comma


# === Documentation

.PHONY: main.pdf path.md

main.pdf: README.md path.md
	pandoc $^ -o $@

path.md: path.c
	ldoc -o $@ $^

# === Cleanup and tarball

.PHONY: clean realclean

clean:
	rm -f *.o *.dump

realclean: clean
	rm -f path.x path-mpi.x path.md main.pdf

import os
import sys

HEADER = "n,p,time,check,omp_threads,mpi_threads"

def base(s):
    """ foo/bar/baz.py --> baz """
    return os.path.splitext(os.path.basename(s))[0]

def contents(filename):
    with open(filename, "r") as f:
        return f.read()

def collect(b, filenames):
    with open("{}.csv".format(b), "w") as out:
        s = "".join(contents(f) for f in filenames)
        out.write(HEADER + "\n")
        out.write(s)

def main(filenames):
    bases = {base(f) for f in filenames}
    for b in bases:
        collect(b, [f for f in filenames if base(f) == b])

if __name__ == "__main__":
    main(sys.argv[1:])

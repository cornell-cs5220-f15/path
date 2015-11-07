import csv
import matplotlib
import os
import sys

matplotlib.use("Agg") # http://webcache.googleusercontent.com/a/3054314
import matplotlib.pyplot as plt

def base(s):
    """ foo/bar/baz.py --> baz """
    return os.path.splitext(os.path.basename(s))[0]

def unzip(xys):
    xs, ys = zip(*xys)
    return (list(xs), list(ys))

def csv_to_dict(csv_filename):
    with open(csv_filename, "r") as f:
        reader = csv.DictReader(f)
        return [row for row in reader]

def plot(b, ds):
    ns, times = unzip(sorted((int(d["n"]), float(d["time"])) for d in ds))
    plt.plot(ns, times, label=b)

def main(csv_filenames):
    plt.figure()

    for csv_filename in csv_filenames:
        b = base(csv_filename)
        ds = csv_to_dict(csv_filename)
        plot(b, ds)

    plt.grid()
    plt.legend(loc="best")
    plt.xlabel("n")
    plt.ylabel("time")
    plt.savefig("plot.pdf")

if __name__ == "__main__":
    main(sys.argv[1:])

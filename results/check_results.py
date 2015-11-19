import csv
import os
import sys

def csv_to_dict(csv_filename):
    with open(csv_filename, "r") as f:
        reader = csv.DictReader(f)
        return [row for row in reader]

def base(s):
    """ foo/bar/baz.py --> baz """
    return os.path.splitext(os.path.basename(s))[0]

def index_by_n(ds):
    return {d["n"]: d for d in ds}

def check((ref, ref_base), (dev, dev_base)):
    ref = index_by_n(ref)
    dev = index_by_n(dev)
    for (n, d) in dev.iteritems():
        ref_check = ref[n]["check"].strip()
        dev_check = d["check"].strip()
        if ref_check != dev_check:
            print "ERROR: {}[{}] = {} != {} = {}[{}]".format(
                    ref_base, n, ref_check, dev_check, dev_base, n)

def main(csv_filenames):
    if len(csv_filenames) < 1:
        print "usage: python check_results.py reference.csv [out.csv]..."
        sys.exit(-1)

    ref_filename = csv_filenames.pop(0)
    ref = csv_to_dict(ref_filename)
    for f in csv_filenames:
        check((ref, base(ref_filename)), (csv_to_dict(f), base(f)))

if __name__ == "__main__":
    main(sys.argv[1:])

import os, re, csv


def collect(name):
    with open('results-{}.csv'.format(name), 'w') as out:
        writer = csv.writer(out, delimiter=',')
        writer.writerow(['n', 'p', 'time'])
        for fname in os.listdir(os.getcwd()):
            if re.match('{}.o[0-9]+'.format(name), fname):
                with open(fname) as f:
                    print fname
                    try:
                        for line in f:
                            par = line.split(', ')
                            if len(par) == 6:
                                p = par[-1][:-1]
                                n = par[0]
                                time = par[2]
                                writer.writerow([n, p, time])
                    except:
                        print par, found

# collect('rs-omp')
# collect('block-hybrid')
# collect('block-mpi')
collect('block-hybrid-weak')
collect('block-mpi-weak')

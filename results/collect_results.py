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


collect('rs-omp')
collect('rs-hybrid')
collect('rs-mpi')
collect('fw-omp')
collect('fw-hybrid')
collect('fw-mpi')
collect('block-omp')
collect('block-hybrid')
collect('block-mpi')
collect('weak_block-hybrid')
collect('weak_block-mpi')
collect('weak_rs-hybrid')
collect('weak_rs-mpi')

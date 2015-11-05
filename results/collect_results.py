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
                        found = 0
                        for line in f:
                            par = line.split()
                            if found == 0 and len(par) > 0 and par[0] == '==':
                                found = 1
                                p = par[2]
                                print p
                            elif found == 1 and len(par) > 0 and par[0] == 'n:':
                                found = 2
                                n = par[-1]
                            elif found == 2 and len(par) > 0 and par[0] == 'Time:':
                                time = par[-1]
                                found = 0
                                writer.writerow([n, p, time])
                    except:
                        print par, found

collect('omp')
collect('hybrid')
collect('mpi')

#!/usr/bin/env python

import os, re, csv

def check(basic, new):
    for file_name in os.listdir(os.getcwd()):
        if re.match('{}.o[0-9]+'.format(basic), file_name):
            print "basic: %s in %s" %(basic, file_name)
            basic_name = file_name
            basic_out = open(file_name, 'r')
            basic_result = {}
            found = 0
            for line in basic_out:
                par = line.split()
                if found == 0 and len(par) > 0 and par[0] == '==':
                    found = 1
                    p = par[2]
                elif found == 1 and len(par) > 0 and par[0] == 'n:':
                    found = 2
                    n = par[-1]
                elif found == 2 and len(par) > 0 and par[0] == 'Check:':
                    result = par[-1]
                    found = 0
                    basic_result[n] = result
            basic_out.close()

    for file_name in os.listdir(os.getcwd()):
        if re.match('{}.o[0-9]+'.format(new), file_name):
            print "new: %s in %s" %(new, file_name)
            new_name = file_name
            new_out = open(file_name, 'r')
            found = 0
            error = False
            for line in new_out:
                par = line.split()
                if found == 0 and len(par) > 0 and par[0] == '==':
                    found = 1
                    p = par[2]
                elif found == 1 and len(par) > 0 and par[0] == 'n:':
                    found = 2
                    n = par[-1]
                elif found == 2 and len(par) > 0 and par[0] == 'Check:':
                    result = par[-1]
                    found = 0
                    if result != basic_result[n]:
                        print "ERROR IS DETECTED"
                        print "%s in %s: n = %d, Check = %s" %(basic, basic_name, n, basic_result[n])
                        print "%s in %s: n = %d, Check = %s" %(new, new_name, n, result)
                        error = True
            new_out.close()

            if not error:    
                print "NO ERROR IS DETECTED"

check('omp', 'mpi')
check('omp', 'hybrid')

import sys
from math import *

def round(x,p):
    y = int(x)
    return (y/p)*p


if __name__=='__main__':
    p = int(sys.argv[1])
    print round(2000*p**(1/3.0),p)


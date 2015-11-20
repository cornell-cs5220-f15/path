import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import sys

rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

threads, timings = np.loadtxt('strong_scaling.csv', delimiter=',', usecols=(0,1), unpack=True)

serial_time = timings[0];
timings = np.divide(serial_time, timings)

plt.plot(threads, timings,'k')
plt.xlim([1,26])
plt.xlabel("Number of OMP threads")
plt.ylabel("Speedup over the serial implementation")
plt.savefig('strong_scaling.png', dpi=300)
plt.clf()
plt.cla()

threads, timings = np.loadtxt('weak_scaling.csv', delimiter=',', usecols=(0,1), unpack=True)
serial_time = timings[0];
timings = np.divide(serial_time, timings)

plt.plot(threads, timings,'k')
plt.xlim([1,20])
plt.xlabel("Number of OMP threads")
plt.ylabel("Efficiency")
plt.savefig('weak_scaling.png', dpi=300)
plt.clf()
plt.cla()

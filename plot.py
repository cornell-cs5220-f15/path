import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

time = pd.read_csv("time.txt", sep=' ', header=None)
time = time.as_matrix()
time = time[:,2]

length1 = len(time)
for i in range(1, length1):
	time[i] = time[0] / time[i]
time[0] = 1

time_weak = pd.read_csv("time_weak.txt", sep=' ', header=None)
time_weak = time_weak.as_matrix()
time_weak = time_weak[:,2]

length2 = len(time_weak)
for i in range(1, length2):
	time_weak[i] = time_weak[0] / time_weak[i]
time_weak[0] = 1

x = range(1,length1 + 1)
xx = range(1,length2 + 1)

plt.subplot(121)
plt.plot(x,time, linewidth=3)
plt.tick_params(axis='x', labelsize=40)
plt.tick_params(axis='y', labelsize=40)
plt.xlabel("number of threads", fontsize=40)
plt.ylabel("Speed up", fontsize=40)
plt.title("Strong scaling for n=2000",fontsize=40)

plt.subplot(122)
plt.plot(xx,time_weak, linewidth=3)
plt.tick_params(axis='x', labelsize=40)
plt.tick_params(axis='y', labelsize=40)
plt.xlabel("number of threads", fontsize=40)
plt.ylabel("Speed up", fontsize=40)
plt.title("Weak scaling",fontsize=40)
plt.show()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

time = pd.read_csv("time.txt", sep=' ', header=None)
time = time.as_matrix()
time = time[:,2]

length = len(time)
for i in range(1, length):
	time[i] = time[0] / time[i]
time[0] = 1
x = range(1,length + 1)
plt.plot(x,time, linewidth=3)
plt.tick_params(axis='x', labelsize=20)
plt.tick_params(axis='y', labelsize=20)
plt.xlabel("number of threads", fontsize=20)
plt.ylabel("Time for the whole process", fontsize=20)
plt.title("Weaking scaling for problem size = 2000",fontsize=20)
plt.show()

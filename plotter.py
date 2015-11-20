#densities = [0.01, 0.02, 0.05, 0.10, 0.20, 0.25, 0.30, 0.40, 0.50]
densities = [0.05]
ns = [100, 250, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000] 

outfile = open('results.csv', 'w')
for n in ns:
  for j in densities:
    for i in range(1, 25):
      f = open('path_' + str(i) + '_' + str(j) + '_' + str(n) + '.stats')
      for l in range(0,4):
        f.readline()
      time = float(f.readline().split(' ')[2])
      outfile.write(str(i) + ', ' + str(n) + ', ' + str(time) + '\n')
  outfile.write()

outfile.close()

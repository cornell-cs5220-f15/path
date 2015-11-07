import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def average_results(data, baseline=False):
    output = []
    for n in np.unique(data['n']):
        temp = data[data['n'] == n]
        if not baseline:
            for p in np.unique(temp['p']):
                temp2 = temp[temp['p'] == p]
                output.append((n, p, np.mean(temp2['time'])))
        else:
            output.append((n, np.mean(temp['time'])))
    if baseline:
        return np.array(output, dtype={'names':['n', 'time'], 'formats':['f4', 'f4']})
    else:
        return np.array(output, dtype={'names':['n', 'p', 'time'], 'formats':['f4', 'f4', 'f4']})

def create_graph(name, baseline_name, baseline_proc):
    results = np.genfromtxt('results-{}.csv'.format(name), delimiter=',', names=True)
    results = average_results(results)
    baseline = np.genfromtxt('results-{}.csv'.format(baseline_name), delimiter=',', names=True)
    baseline = average_results(baseline)
    baseline = baseline[baseline['p'] == baseline_proc]
    print baseline
    baseline.sort(order=['n'])
    N = baseline['n']
    baseline_y = baseline['time']
    colors = ['red', 'green', 'blue', 'yellow', 'black', 'magenta', 'cyan', 'brown', 'orange', 'gray', 'purple', 'pink']
    # strong scaling
    for i, n in enumerate(N):
        pts = results[results['n']==n]
        pts.sort(order=['n'])
        x = pts['p']
        y = baseline_y[i] / pts['time']
        plt.plot(x,y, color=colors[i], label=str(int(n)))
    plt.legend(bbox_to_anchor=(1.12,1),
              ncol=1)
    plt.xlabel('Threads')
    plt.ylabel('Speedup')
    plt.savefig('strong_{}_baseline-{}-{}.pdf'.format(name, baseline_name, baseline_proc))
    plt.savefig('strong_{}_baseline-{}-{}.png'.format(name, baseline_name, baseline_proc))
    plt.close()

    # weak scaling
    n = 500
    i = 0
    pts = results[(results['n']/np.sqrt(results['p']))==n]
    print n
    print pts
    pts.sort(order=['n'])
    x = pts['p']
    y = baseline_y[i] / pts['time']
    plt.plot(x,y, color=colors[i], label=str(n))

    plt.legend(bbox_to_anchor=(1.12,1),
              ncol=1)
    plt.xlabel('Threads')
    plt.ylabel('Speedup')
    plt.savefig('weak_{}_baseline-{}-{}.pdf'.format(name, baseline_name, baseline_proc))
    plt.savefig('weak_{}_baseline-{}-{}.png'.format(name, baseline_name, baseline_proc))
    plt.close()

create_graph('hybrid', 'hybrid', 1)
create_graph('hybrid', 'mpi', 1)
create_graph('hybrid', 'omp', 24)
create_graph('mpi', 'mpi', 1)
create_graph('mpi', 'omp', 24)


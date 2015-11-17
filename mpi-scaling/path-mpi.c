#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <sys/time.h>
#include "mt19937p.h"

//ldoc on

/**
 *
 * The value $l_{ij}^0$ is almost the same as the $(i,j)$ entry of
 * the adjacency matrix, except for one thing: by convention, the
 * $(i,j)$ entry of the adjacency matrix is zero when there is no
 * edge between $i$ and $j$; but in this case, we want $l_{ij}^0$
 * to be "infinite".  It turns out that it is adequate to make
 * $l_{ij}^0$ longer than the longest possible shortest path; if
 * edges are unweighted, $n+1$ is a fine proxy for "infinite."
 * The functions `infinitize` and `deinfinitize` convert back 
 * and forth between the zero-for-no-edge and $n+1$-for-no-edge
 * conventions.
 */

static inline void infinitize(int n, int* l)
{
    for (int i = 0; i < n*n; ++i)
        if (l[i] == 0)
            l[i] = n+1;
}

static inline void deinfinitize(int n, int* l)
{
    for (int i = 0; i < n*n; ++i)
        if (l[i] > n)
            l[i] = 0;
}

// Extract an n*n matrix from a padded m*m matrix, assuming m >= n
void unpad(int n, int m, int* graph, int* paddedgraph) {
    for (int i = 0; i < n; i++) {
        memcpy(graph + (i * n), paddedgraph + (i * m), n * sizeof(int));
    }
    _mm_free(paddedgraph);
}

// Pad an n*n matrix into an m*m matrix, assuming m >= n
int* pad(int n, int m, int* graph) {
    size_t paddedsize = m*m * sizeof(int);
    int* paddedgraph = (int*) _mm_malloc(m * m, sizeof(int));
    memset(paddedgraph, m+1, paddedsize);
    for (int i = 0; i < n; i++) {
        memcpy(paddedgraph + (i * m), graph + (i * n), n * sizeof(int));
    }
    return paddedgraph;
}

int receive_updates(int m, int root, int* recvupdates, int* graph) {
    int upsrecv;
    MPI_Bcast(&upsrecv, 1, MPI_INT, root, MPI_COMM_WORLD);
    MPI_Bcast(recvupdates, 3 * upsrecv, MPI_INT, root, MPI_COMM_WORLD);
    for (int i = 0; i < upsrecv; i++) {
        int x = recvupdates[i * 3];
        int y = recvupdates[i * 3 + 1];
        graph[x * m + y] = recvupdates[i * 3 + 2];
    }
    return upsrecv;
}

void shortest_paths(int n, int m, int c, int t, int* l)
{
    // Generate l_{ij}^0 from adjacency matrix representation
    infinitize(m, l);
    // Setting distances between the same node to 0
    for (int i = 0; i < n*n; i += n+1)
        l[i] = 0;

    int* graph = l;
    if (m != n) {
        // Need to pad
        graph = pad(n, m, l);
    }
    int* updates = (int*) _mm_malloc(3 * m * c, sizeof(int));

    // Send matrix to every worker
    MPI_Bcast(graph, m * m, MPI_INT, 0, MPI_COMM_WORLD);

    int done;
    do {
        done = 1;
        for (int i = 1; i <= t; i++) {
            done &= !receive_updates(m, i, updates, graph);
        }
    } while (!done);

    if (m != n) {
        unpad(n, m, l, graph);
    }
    deinfinitize(m, l);
    _mm_free(updates);
}

void shortest_paths_worker(int m, int c, int t, int id) {
    int* graph = (int*) _mm_malloc(m * m, sizeof(int));
    int* updates = (int*) _mm_malloc(3 * m * c, sizeof(int));
    int* recvupdates = (int*) _mm_malloc(3 * m * c, sizeof(int));
    MPI_Bcast(graph, m * m, MPI_INT, 0, MPI_COMM_WORLD);

    int done;
    do {
        done = 1;
        int nups = 0; // Number of updates

        // Optimize the assigned columns
        for (int j = 0; j < c; j++) {
            int x = (id - 1) * m * c + j;
            for (int y = 0; y < m; y++) {
                int hasupdate = 0;
                int newval = graph[x * m + y];
                for (int i = 0; i < m; i++) {
                    int dist = graph[x * m + i] + graph[i * m + y];
                    if (dist < newval) {
                        newval = dist;
                        hasupdate = 1;
                    }
                }
                if (hasupdate) {
                    graph[x * m + y] = newval;
                    done = 0;
                    updates[nups * 3] = x;
                    updates[nups * 3 + 1] = y;
                    updates[nups * 3 + 2] = newval;
                    nups++;
                }
            }
        }

        // Synchronize state with other workers
        for (int i = 1; i <= t; i++) {
            if (i == id) {
                MPI_Bcast(&nups, 1, MPI_INT, i, MPI_COMM_WORLD);
                MPI_Bcast(updates, 3 * nups, MPI_INT, i, MPI_COMM_WORLD);
            } else {
                done &= !receive_updates(m, i, recvupdates, graph);
            }
        }
    } while (!done);

    _mm_free(graph);
    _mm_free(updates);
    _mm_free(recvupdates);
}

/**
 * # The random graph model
 *
 * Of course, we need to run the shortest path algorithm on something!
 * For the sake of keeping things interesting, let's use a simple random graph
 * model to generate the input data.  The $G(n,p)$ model simply includes each
 * possible edge with probability $p$, drops it otherwise -- doesn't get much
 * simpler than that.  We use a thread-safe version of the Mersenne twister
 * random number generator in lieu of coin flips.
 */

int* gen_graph(int n, double p)
{
    int* l = _mm_malloc(n * n, sizeof(int));
    memset(l, 0, n * n * sizeof(int));
    struct mt19937p state;
    sgenrand(10302011UL, &state);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i)
            l[j*n+i] = (genrand(&state) < p);
        l[j*n+j] = 0;
    }
    return l;
}

/**
 * # Result checks
 *
 * Simple tests are always useful when tuning code, so I have included
 * two of them.  Since this computation doesn't involve floating point
 * arithmetic, we should get bitwise identical results from run to
 * run, even if we do optimizations that change the associativity of
 * our computations.  The function `fletcher16` computes a simple
 * [simple checksum][wiki-fletcher] over the output of the
 * `shortest_paths` routine, which we can then use to quickly tell
 * whether something has gone wrong.  The `write_matrix` routine
 * actually writes out a text representation of the matrix, in case we
 * want to load it into MATLAB to compare results.
 *
 * [wiki-fletcher]: http://en.wikipedia.org/wiki/Fletcher's_checksum
 */

int fletcher16(int* data, int count)
{
    int sum1 = 0;
    int sum2 = 0;
    for(int index = 0; index < count; ++index) {
          sum1 = (sum1 + data[index]) % 255;
          sum2 = (sum2 + sum1) % 255;
    }
    return (sum2 << 8) | sum1;
}

void write_matrix(const char* fname, int n, int* a)
{
    FILE* fp = fopen(fname, "w+");
    if (fp == NULL) {
        fprintf(stderr, "Could not open output file: %s\n", fname);
        exit(-1);
    }
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) 
            fprintf(fp, "%d ", a[j*n+i]);
        fprintf(fp, "\n");
    }
    fclose(fp);
}

/**
 * # The `main` event
 */

const char* usage =
    "path.x -- Parallel all-pairs shortest path on a random graph\n"
    "Flags:\n"
    "  - n -- number of nodes (200)\n"
    "  - p -- probability of including edges (0.05)\n"
    "  - i -- file name where adjacency matrix should be stored (none)\n"
    "  - o -- file name where output matrix should be stored (none)\n";

int main(int argc, char** argv)
{
    int n    = 200;            // Number of nodes
    double p = 0.05;           // Edge probability
    const char* ifname = NULL; // Adjacency matrix file name
    const char* ofname = NULL; // Distance matrix file name

    // Option processing
    extern char* optarg;
    const char* optstring = "hn:d:p:o:i:";
    int c;
    while ((c = getopt(argc, argv, optstring)) != -1) {
        switch (c) {
        case 'h':
            fprintf(stderr, "%s", usage);
            return -1;
        case 'n': n = atoi(optarg); break;
        case 'p': p = atof(optarg); break;
        case 'o': ofname = optarg;  break;
        case 'i': ifname = optarg;  break;
        }
    }

    int t, id;
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&t);
    MPI_Comm_rank(MPI_COMM_WORLD,&id);
    t -= 1; // Use t - 1 as one of the processes is root

    if (t > n) {
        fprintf(stderr, "Number of workers exceeds dimension of matrix!");
        return 1;
    }

    c = (int) ceil(((float) n) / ((float) t));
    int m = t * c;

    if (id == 0) {
        // Graph generation + output
        int* l = gen_graph(n, p);
        if (ifname)
            write_matrix(ifname,  n, l);

        struct timeval t0, t1;
        gettimeofday(&t0, NULL);
        shortest_paths(n, m, c, t, l); 
        gettimeofday(&t1, NULL);
        double elapsed = (t1.tv_sec-t0.tv_sec) + (t1.tv_usec-t0.tv_usec)*1e-6;

        printf("%d,%d,%g,%g,%X\n", t, n, elapsed, p, fletcher16(l, n*n));

        // Generate output file
        if (ofname)
            write_matrix(ofname, n, l);

        // Clean up
        _mm_free(l);
    } else {
        shortest_paths_worker(m, c, t, id);
    }
    
    MPI_Finalize();

    return 0;
}

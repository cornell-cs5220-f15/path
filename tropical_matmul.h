#include <mpi.h>

void tropical_matmul_partial(int *A, int *C, int offset, int width, int dim){
	for(int i = 0; i < dim; i ++){
		for(int j = offset; j < (offset + width); j ++){
				int Cij = C[i +j*dim];
			for(int k = 0; k < dim; k ++){
				int Aik = A[i + k*dim];
                int Akj = A[k + j*dim];
                    if(Aik + Akj < Cij){
                        Cij = Aik + Akj;
                    }
			}
			C[i + j*dim] = Cij;
		}
	}
}

void mpi_tropical_matmul(int size, int *A, int *C) {

	//Get the number of processes
	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	// Get the rank of the process
	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	//init memory and stuff

	for(int i = 0; i < size*size; i ++) A[i] = 1;

	int width = size/world_size;
	tropical_matmul_partial(A, C, world_rank*width, width, size);

	//printf("Thread: %d, Value: %d\n", world_rank, C[world_rank*size*width]);

	/*for(int i = 0; i < log2(world_rank); i ++){
	recepient = (world_rank + pow(2,i)) % 24;
	sender = (2*world_rank - pow(2,i)) % 24;

	MPI_Send(C, size*size, MPI_INT, (world_rank + 1 + 24) % world_size, 0, MPI_COMM_WORLD);
	MPI_Recv(C, size*size, MPI_INT, (world_rank - 1 + 24) % world_size, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	}*/

	MPI_Allgather(C+world_rank*size*width, size*width, MPI_INT, A, size*width, MPI_INT, MPI_COMM_WORLD);

	// Finalize the MPI environment.

}

static inline void infinitize(int n, int* l)
{
    for (int i = 0; i < n*n; ++i)
        if (l[i] == 0)
            l[i] = n+1;
}

static inline void deinfinitize(int n, int* l)
{
    for (int i = 0; i < n*n; ++i)
        if (l[i] == n+1)
            l[i] = 0;
}

void shortest_paths(int n, int* restrict l)
{
	MPI_Init(NULL, NULL);
    // Generate l_{ij}^0 from adjacency matrix representation
    infinitize(n, l);
    for (int i = 0; i < n*n; i += n+1)
        l[i] = 0;

    // Repeated squaring until nothing changes
    int* restrict lnew = (int*) calloc(n*n, sizeof(int));
    memcpy(lnew, l, n*n * sizeof(int));
    for (int i = 0; i < n; i ++ ) {
        mpi_tropical_matmul(n, l, lnew);
        memcpy(l, lnew, n*n * sizeof(int));
    }
    free(lnew);
    deinfinitize(n, l);
    MPI_Finalize();
}
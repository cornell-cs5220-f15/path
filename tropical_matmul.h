#include <mpi.h>
#include <stdlib.h>

void toBin(int value, int bitsCount, char* output)
{
    int i;
    output[bitsCount] = '\0';
    for (i = bitsCount - 1; i >= 0; --i, value >>= 1)
    {
        output[i] = (value & 1) + '0';
    }
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

void tropical_matmul_partial(int *A, int *C, int offset, int width, int dim){
	for(int i = 0; i < dim; i ++){
		for(int j = offset; j < (offset + width); j ++){
				int Cij = 0;
			for(int k = 0; k < dim; k ++){
				int Aik = A[i + k*dim];
                int Akj = A[k + j*dim];
                    if(Aik + Akj < Cij){
                        Cij = Aik + Akj;
                    }
                Cij += Aik*Akj;
			}
			C[i + j*dim] = Cij;
		}
	}
}

void mpi_tropical_matmul(int size, int *A, char* binary_representation, int length_binary) {
	MPI_Init(NULL, NULL);
    // Generate l_{ij}^0 from adjacency matrix representation
    infinitize(size, A);
    for (int i = 0; i < size*size; i += size+1)
        A[i] = 0;
    int* C = (int*) calloc(size*size, sizeof(int));

	//Get the number of processes
	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	// Get the rank of the process
	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	int width = size/world_size;

	//continually square A. TODO: binary representation. 
	for(int i = 0; i < 1; i ++){
		tropical_matmul_partial(A, C, world_rank*width, width, size);
		MPI_Allgather(C+world_rank*size*width, size*width, MPI_INT, A, size*width, MPI_INT, MPI_COMM_WORLD);
	}

	// Finalize the MPI environment.
	deinfinitize(size, A);

	//print matrix for debugging purposes
	// if(world_rank == 0){
	// 	for(int i = 0; i < size; i++){
	// 		for(int j = 0; j < size; j++){
	// 			printf(" %d ", A[i + j*size]);
	// 		}
	// 		printf("\n");
	// 	}
	// }

    MPI_Finalize();
}



void shortest_paths(int n, int* restrict l)
{

	//doing some annoying binary number manipulation, first cutting the binary number down and then flipping it
	char str[32];
    toBin(n, 32, str);
    int offset;
    for(int i = 0; i < 32; i ++){
    if(str[i] == '1'){ 
        offset = i;
        break;
        }
    }
    int length_bin = 32-offset;
    printf("%d\n", length_bin);
    char *binary_representation = (char*)malloc(length_bin*sizeof(char));
    for(int i = 0; i < length_bin; i ++){
        binary_representation[i] = str[offset + i];
    }

    for(int i = 0; i < length_bin/2; i ++){
    	char temp;
    	temp = binary_representation[i];
        binary_representation[i] = binary_representation[length_bin - i - 1];
        binary_representation[length_bin - i - 1] = temp;
    }

	mpi_tropical_matmul(n,l, binary_representation, length_bin);
}
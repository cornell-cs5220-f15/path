#include <mpi.h>
#include <stdlib.h>
#include <omp.h>

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
        if (l[i] >= n+1)
            l[i] = 0;
}

void tropical_matsquare_partial(int *A, int *C, int offset, int width, int dim){
	//do some copy optimization 
	int * Abuf = (int*)_mm_malloc(dim*sizeof(int),64);
	__assume_aligned(Abuf, 64);
	//__assume_aligned(A, 64);
    #pragma omp parallel for shared(A, C)
	for(int i = 0; i < dim; i ++){
		
		for(int n = 0; n < dim; n++) Abuf[n] =  A[i + n*dim];
		for(int j = offset; j < (offset + width); j ++){
			int Cij = dim*dim;
			
			for(int k = 0; k < dim; k ++){
				int Aik = Abuf[k];
               			int Akj = A[k + j*dim];
                   	 if(Aik + Akj < Cij){
                        Cij = Aik + Akj;
                    }
			}
			C[i + j*dim] = Cij;
		}
	}
}

void tropical_matmul_partial(int *A, int *B, int *C, int offset, int width, int dim){

    int *Abuf = (int*)_mm_malloc(dim*sizeof(int),64);

    __assume_aligned(Abuf, 64);
    //__assume_aligned(B, 64);
    #pragma omp parallel for shared(A, B, C)
    for(int i = 0; i < dim; i ++){
	
		for(int n = 0; n < dim; n++) Abuf[n] = A[i + n*dim];

        for(int j = offset; j < (offset + width); j ++){
                int Cij = dim*dim;
            for(int k = 0; k < dim; k ++){
                int Aik = Abuf[k];
                int Bkj = B[k + j*dim];
                    if(Aik + Bkj < Cij){
                        Cij = Aik + Bkj;
                    }
            }
            C[i + j*dim] = Cij;
        }
    }
}


void mpi_tropical_matmul(int size, int *A, char* binary_representation, int length_binary) {
	MPI_Init(NULL, NULL);
    double t0 = omp_get_wtime();
    // Generate l_{ij}^0 from adjacency matrix representation
    infinitize(size, A);
    for (int i = 0; i < size*size; i += size+1)
        A[i] = 0;
    int* C = (int*) calloc(size*size, sizeof(int));
    int *ANS = (int*)calloc(size*size, sizeof(int));


    //making ANS the identity matrix first
    for(int i = 0; i < size*size; i ++) ANS[i] = size*size;
    for(int i = 0; i < size; i ++) ANS[i + size*i] = 0;
	//Get the number of processes
	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	// Get the rank of the process
	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	int width = size/world_size;


	//continually square A. TODO: binary representation. 
	for(int i = 0; i < length_binary; i ++){
        if(binary_representation[i] == '0'){
            //don't do anything
        }
        else if(binary_representation[i] == '1'){
            //multiply the current 2-power of A by our answer i.e. ANS = A*ANS
            tropical_matmul_partial(A, ANS, C, world_rank*width, width, size);
            MPI_Allgather(C+world_rank*size*width, size*width, MPI_INT, ANS, size*width, MPI_INT, MPI_COMM_WORLD);
        }
        else{
            printf("invalid binary representation\n");
        }
		tropical_matsquare_partial(A, C, world_rank*width, width, size);
		MPI_Allgather(C+world_rank*size*width, size*width, MPI_INT, A, size*width, MPI_INT, MPI_COMM_WORLD);
	}

	// Finalize the MPI environment.
	deinfinitize(size, ANS);
    memcpy(A, ANS, size*size*sizeof(int));
    double t1 = omp_get_wtime();

    if(world_rank == 0){
        printf("n:     %d\n", size);
        printf("Time:  %g\n", t1-t0);
    }
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

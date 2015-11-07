#include <stdio.h>
#include <mpi.h>


int main(int argc, char** argv)
{
    int rank, size, i, local_sum = 0;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int next_buf[1], curr_buf[1];
    int prev;
    MPI_Request s_request, r_request;
    MPI_Status s_status, r_status;

    // Doing my work
    curr_buf[0] = rank;
    
    for (i=0; i<(size); i++) {
        // Send my work
        MPI_Isend(curr_buf, sizeof(curr_buf), MPI_INT, (rank + 1)%size, 77, MPI_COMM_WORLD, &s_request);

        MPI_Irecv(next_buf, sizeof(next_buf), MPI_INT, (size + rank - 1)%size, 77, MPI_COMM_WORLD, &r_request);
        local_sum += curr_buf[0];
        MPI_Wait (&r_request, &r_status);
        MPI_Wait (&s_request, &s_status);
        curr_buf[0] = next_buf[0];
    }
    
    printf("Total of %d: %d\n", rank, local_sum);
    MPI_Finalize();

    return 0;
}

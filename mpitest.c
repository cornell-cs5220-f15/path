#include "mpi.h"
#include <stdio.h>

int main(int argc, char *argv[]) {
    char hostname[MPI_MAX_PROCESSOR_NAME];

    int rc = MPI_Init(&argc, &argv);

    int numtasks, rank, len;
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Get_processor_name(hostname, &len);

    printf ("Number of tasks= %d My rank= %d Running on %s\n", numtasks, rank, hostname);

    MPI_Finalize();

    return 0;
}

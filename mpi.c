#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define M 3  // Number of rows in A
#define K 3  // Number of columns in A and rows in B
#define N 2  // Number of columns in B

void initialize_matrices(int *A, int *B) {

    A[0] = 1; A[1] = 2; A[2] = 3;
    A[3] = 4; A[4] = 5; A[5] = 6;
    A[6] = 7; A[7] = 8; A[8] = 9;


    B[0] = 1; B[1] = 4;
    B[2] = 2; B[3] = 5;
    B[4] = 3; B[5] = 6;
}

void print_matrix(int *C) {
    // Function to print matrix C
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", C[i * N + j]);
        }
        printf("\n");
    }
}

int main(int argc, char** argv) {
    int rank, size;

    //  MPI environment
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int *A = NULL; // Matrix A will be initialized only in the root process
    int *B = (int *)malloc(K * N * sizeof(int)); // Matrix B will be broadcast to all processes
    int *C = NULL; //  matrix C,
    int *local_A, *local_C; // Local portions of A and C for each process

    
    int *sendcounts = (int *)malloc(size * sizeof(int));
    int *displs = (int *)malloc(size * sizeof(int));
    int *recvcounts = (int *)malloc(size * sizeof(int));
    int *recvdispls = (int *)malloc(size * sizeof(int));

    
    int rows_per_proc = M / size;
    int remainder = M % size;

    int offset = 0;
    for (int i = 0; i < size; i++) {
        int rows = rows_per_proc + (i < remainder ? 1 : 0);
        sendcounts[i] = rows * K; // Number of elements to send to each process
        displs[i] = offset; // Displacement for each process
        offset += sendcounts[i];
    }


    offset = 0;
    for (int i = 0; i < size; i++) {
        int rows = sendcounts[i] / K;
        recvcounts[i] = rows * N; 
        recvdispls[i] = offset;
        offset += recvcounts[i];
    }

    int local_rows = sendcounts[rank] / K;
    local_A = (int *)malloc(local_rows * K * sizeof(int));
    local_C = (int *)malloc(local_rows * N * sizeof(int));

    if (rank == 0) {
        // Root process initializes matrices A and B
        A = (int *)malloc(M * K * sizeof(int));
        C = (int *)malloc(M * N * sizeof(int));
        initialize_matrices(A, B);
    }

    // Broadcast matrix B to all processes
    MPI_Bcast(B, K * N, MPI_INT, 0, MPI_COMM_WORLD);

    // Scatter rows of matrix A to all processes
    MPI_Scatterv(A, sendcounts, displs, MPI_INT, local_A, sendcounts[rank], MPI_INT, 0, MPI_COMM_WORLD);

    // Start timing the computation
    double start_time = MPI_Wtime();

    // Perform local matrix multiplication
    for (int i = 0; i < local_rows; i++) {
        for (int j = 0; j < N; j++) {
            local_C[i * N + j] = 0;
            for (int k = 0; k < K; k++) {
                local_C[i * N + j] += local_A[i * K + k] * B[k * N + j];
            }
        }
    }

    // Gather the local result matrices back to the root process
    MPI_Gatherv(local_C, local_rows * N, MPI_INT, C, recvcounts, recvdispls, MPI_INT, 0, MPI_COMM_WORLD);

    // End timing
    double end_time = MPI_Wtime();
    double elapsed_time = end_time - start_time;

    if (rank == 0) {
        //  Prints the result matrix and timing information
        printf("Resultant Matrix C:\n");
        print_matrix(C);
        printf("Time taken for computation: %f seconds\n", elapsed_time);

        
    }

    // Free allocated memory
    if (rank == 0) {
        free(A);
        free(C);
    }
    free(B);
    free(local_A);
    free(local_C);
    free(sendcounts);
    free(displs);
    free(recvcounts);
    free(recvdispls);

    // This is how we finalise the MPI environment
    MPI_Finalize();
    return 0;
}

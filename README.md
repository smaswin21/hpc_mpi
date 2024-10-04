# MPI Matrix Multiplication README

## Introduction

This program performs parallel matrix multiplication using MPI (Message Passing Interface). It distributes the computation across multiple processes, where each process handles a portion of the matrix, performs multiplication, and gathers the results.

### Features:

- Matrix multiplication of two matrices, A (3x3) and B (3x2), using parallel processing.
- Uses OpenMPI for distributing computation.
- Results are gathered and displayed by the root process.

## Prerequisites

To run this program on macOS, you need to have the following installed:
- GCC and g++ compilers
- OpenMPI (via Homebrew)

### Steps to Install Prerequisites:

1. **Install Homebrew** (on MacOS):
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    ```
2. **Install OpenMPI** 
    ```bash
    brew install open-mpi
    ```
3. **Checking the MPI Installation**
    ```bash
    mpicc --version
    ```

### Compilation and Execution

1. Navigate to the directory where your MPI C file is located.

2. Compile the program using mpicc by running the following command:
   ```bash
    mpicc -o mpi_program /path/to/mpi_program.c
    ```
3. Running the program
   
* To run the program using multiple processes, use the 'mpirun' command. Here's an example with 2 processes:
  ```bash
  mpirun -np 4 ./mpi_program
  ```
* This command will distribute the computation across two processes.

### Output

The program will output the resulting matrix and the time taken for the computation into a file named output.txt in the current directory.

### Explanation of Key MPI Functions:

* MPI_Init: Initializes the MPI environment.

* MPI_Comm_rank: Determines the rank of the process (i.e., the ID of the process).

* MPI_Comm_size: Determines the number of processes.

* MPI_Bcast: Broadcasts matrix B from the root process to all other processes.

* MPI_Scatterv: Scatters rows of matrix A among all processes.

* MPI_Gatherv: Gathers the results from all processes to the root process.

* MPI_Finalize: Finalizes the MPI environment.




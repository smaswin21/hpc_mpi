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



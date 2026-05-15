#include <iostream>
#include <cstdlib>
#include <ctime>
#include <mkl.h>
#include <omp.h>
#include <numeric>
#include "rw_matrix.hpp"
#include "bicgstab.hpp"
#include "misc.hpp"
#include "sparseMatrix.hpp"
#include "parPrec.hpp"

int main(int argc, char **argv)
{
    // Check input number
    if (argc != 6)
    {
        std::cout << "\nThis program requires 5 command line arguments:\n";
        std::cout << "(1) Name of the file retaining the Coefficient matrix in Matrix Market format\n";
        std::cout << "(2) Name of the file retaining the right hand side in Matrix Market format\n";
        std::cout << "(3) Levels of fill (lfill)\n\n";
        std::cout << "(4) Drop tolerance (\\epsilon)\n\n";
        std::cout << "(5) Size of local linear systems (warpsize) or -1 if NA\n\n";
        exit(1);
    }

    // Variables to measure execution time
    double start, end;

    // Variables corresponding to the coefficient matrix
    int n;
    sparseMatrix<int, int, double> A;
    std::vector<double> ID;

    // Variables corresponding to the right hand side
    std::vector<double> B;
    int m;

    // Variables corresponding to the AFIIM preconditioner
    sparseMatrix<int, int, double> L, U;
    double ftol = std::max(atof(argv[4]), std::numeric_limits<double>::epsilon());
    int lfill = std::max(atoi(argv[3]), 1), warpsize = atoi(argv[5]);

    // EPBiCGSTAB related variables
    int NMAX = 2000, iter, info;
    double tol = 1e-8, resval;
    std::vector<double> x;
    std::fill(x.begin(), x.end(), 0.0);

    // Read coefficient matrix from file
    read_coeff_matrix(std::string(argv[1]), A);
    n = A.r;

    // Read right hand side from file or dummy
    if (std::string(argv[2]) == std::string("none"))
    {
        B.resize(n);
        form_model_rhs(A, B);
    }
    else
    {
        read_rhs_vector(std::string(argv[2]), B);
    }
    m = B.size();

    // Check if dimensions agree
    if (n != m)
    {
        std::cout << "Coefficient matrix and right hand side have different dimensions";
        exit(1);
    }

    std::cout << "Coefficient Matrix A with n = " << n << " and nnz(A) = " << A.nnz << "\n\n";

    // Allocate and set initial guess
    x.resize(n, 0.0);

    // Form AFIIM preconditioner
    start = omp_get_wtime();

    piilus(A,
           L,
           ID,
           U,
           lfill,
           ftol,
           warpsize,
           'm',
           'B');
    end = omp_get_wtime();

    // Print thread numbers
    std::cout << "A                : " << std::string(argv[1]) << std::endl;
    std::cout << "B                : " << std::string(argv[2]) << std::endl;
    std::cout << "LFill            : " << lfill << std::endl;
    std::cout << "Ftol             : " << ftol << std::endl;
    std::cout << "Warpsize         : " << warpsize << std::endl;
    std::cout << "OpenMP Threads   : " << omp_get_max_threads() << std::endl;
    std::cout << "MKL Threads      : " << mkl_get_max_threads() << std::endl;

    // Print nonzeros of preconditioner
    std::cout << "Elapsed time for computation of PIILUS                   : " << ((double)(end - start)) << " seconds\n";
    std::cout << "Nonzero elements in preconditioner (nnz(L))              : " << L.nnz << "\n";
    std::cout << "Nonzero elements in preconditioner (nnz(U))              : " << U.nnz << "\n";
    std::cout << "Nonzero elements in preconditioner (nnz(L)+nnz(U)+nnz(D)): " << n + L.nnz + U.nnz << "\n";
    std::cout << "Density (\\rho)                                           : " << (double)(n + L.nnz + U.nnz) / A.nnz << "\n\n";

    // Solve the linear system
    start = omp_get_wtime();
    IPBiCGSTAB(A,
               B,
               tol,
               NMAX,
               L,
               ID,
               U,
               x,
               &resval,
               &iter,
               1,
               &info);
    end = omp_get_wtime();

    std::cout << "Elapsed time for computation of PIILUS-IPBiCSTAB         : " << ((double)(end - start)) << " seconds\n";

    // Write solution to file
    write_vector("solution.mtx", x);

    return 0;
}

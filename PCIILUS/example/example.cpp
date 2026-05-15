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
#include "sparseLinearAlgebra.hpp"
#include "parPrec.hpp"
#include "constants.hpp"

int main(int argc, char **argv)
{
        // Check input number
        if (argc != 8)
        {
                std::cout << "\nThis program requires 7 command line arguments:\n";
                std::cout << "(1) Name of the file retaining the Coefficient matrix in Matrix Market format\n";
                std::cout << "(2) Name of the file retaining the right hand side in Matrix Market format\n";
                std::cout << "(3) Levels of fill (lfill)\n";
                std::cout << "(4) Drop tolerance (\\epsilon)\n";
                std::cout << "(5) Drop tolerance (\\eta)\n";
                std::cout << "(6) Size of local linear systems (warpsize) or -1 if NA\n";
                std::cout << "(7) Number of iterations (n_r)\n";
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
        sparseMatrix<int, int, double> L, U, H, G;
        double ftol = std::max(atof(argv[4]), 0.0); // std::numeric_limits<double>::epsilon()
        double ftoli = std::max(atof(argv[5]), 0.0);
        int lfill = std::max(atoi(argv[3]), 1), warpsize = atoi(argv[6]), nr = atoi(argv[7]);

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
                read_rhs_vector(std::string(argv[2]),
                                B);
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

        pciilus(A,
                L,
                H,
                ID,
                U,
                G,
                lfill,
                ftol,
                ftoli,
                warpsize,
                'm',
                'B');
        end = omp_get_wtime();

        // Print thread numbers
        std::cout << "A                : " << std::string(argv[1]) << std::endl;
        std::cout << "B                : " << std::string(argv[2]) << std::endl;
        std::cout << "LFill            : " << lfill << std::endl;
        std::cout << "\\epsilon         : " << ftol << std::endl;
        std::cout << "\\eta             : " << ftoli << std::endl;
        std::cout << "Warpsize         : " << warpsize << std::endl;
        std::cout << "n_r              : " << nr << std::endl;
        std::cout << "OpenMP Threads   : " << omp_get_max_threads() << std::endl;
        std::cout << "MKL Threads      : " << mkl_get_max_threads() << std::endl;

        // Print nonzeros of preconditioner
        std::cout << "Elapsed time for computation of AFIIM: " << ((double)(end - start)) << " seconds\n";
        std::cout << "Nonzero elements in preconditioner (nnz(L))              : " << L.nnz << "\n";
        std::cout << "Nonzero elements in preconditioner (nnz(U))              : " << U.nnz << "\n";
        std::cout << "Nonzero elements in preconditioner (nnz(H))              : " << H.nnz << "\n";
        std::cout << "Nonzero elements in preconditioner (nnz(G))              : " << G.nnz << "\n";

        std::cout << "Nonzero elements in preconditioner (nnz(L)+nnz(U)+nnz(D)): " << n + L.nnz + U.nnz << "\n";
        std::cout << "Density (factorization) (\\rho)                          : " << (double)(n + L.nnz + U.nnz) / A.nnz << "\n";

        std::cout << "Nonzero elements in preconditioner (nnz(G)+nnz(H)+nnz(D)): " << n + G.nnz + H.nnz << "\n";
        std::cout << "Density (inverse)       (\\rho)                          : " << (double)(n + G.nnz + H.nnz) / A.nnz << "\n";

        std::cout << "\n";

        // Solve the linear system
        start = omp_get_wtime();
        PBiCGSTAB(A,
                  B,
                  tol,
                  NMAX,
                  L,
                  H,
                  ID,
                  U,
                  G,
                  nr,
                  x,
                  &resval,
                  &iter,
                  1,
                  &info);
        end = omp_get_wtime();

        std::cout << "Elapsed time for computation of PCIILUS-EPBiCSTAB        : " << ((double)(end - start)) << " seconds\n";

        // Write solution to file
        write_vector("solution.mtx", x);

        return 0;
}

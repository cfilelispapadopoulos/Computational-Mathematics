#include <iostream>
#include <cstdlib>
#include <ctime>
#include <omp.h>
#include <cstring>
#include "rw_matrix.hpp"
#include "prec.hpp"
#include "bicgstab.hpp"
#include "misc.hpp"
#include "qsort.hpp"

int main(int argc, char **argv)
{
    // Check input number
    if (argc != 7)
    {
        std::cout << "\nThis program requires 6 command line arguments:\n";
        std::cout << "(1) Name of the file retaining the Coefficient matrix in Matrix Market format\n";
        std::cout << "(2) Name of the file retaining the right hand side in Matrix Market format\n";
        std::cout << "(3) Drop tolerance for LDU (\\epsilon_1)\n";
        std::cout << "(4) Drop tolerance for GD^{-1}H (\\epsilon_2)\n";
        std::cout << "(5) Number of relaxation steps (n_r)\n";
        std::cout << "(6) Preconditioning mode (I,E,C)\n\n";
        exit(1);
    }

    // Variables to measure execution time
    double start, end;

    // Variables corresponding to the coefficient matrix
    double *Av;
    int *Ai, *Aj;
    int n;

    // Variables corresponding to the right hand side
    double *B;
    int m;

    // Variables corresponding to the AFIIM preconditioner
    double *Gv, *Hv, *IDv, *Lv, *Uv;
    int *Gi, *Gj, *Hi, *Hj, *Li, *Lj, *Ui, *Uj;
    double dtol1 = atof(argv[3]);
    double dtol2 = atof(argv[4]);
    int nr = atoi(argv[5]);
    int elemPerRowCol = 30, growth = 5;
    double eta = 0.0, shift = 0.0, coeff = 2.0;
    int element_limit = 200, filtit = 30;

    // EPBiCGSTAB related variables
    int NMAX = 2000, iter, info, verbose = 0;
    double tol = 1e-8, resval, *x;
    char mode = argv[6][0];

    // Iterator
    int i;

    // Read coefficient matrix from file
    read_coeff_matrix(argv[1],
                      &n,
                      &Av,
                      &Aj,
                      &Ai);

    // Read right hand side from file
    if (!strcmp(argv[2], "none"))
    {
        form_model_rhs(n, Av, Aj, Ai, &B);
        m = n;
    }
    else
    {
        read_rhs_vector(argv[2],
                        &m,
                        &B);
    }

    // Check if dimensions agree
    if (n != m)
    {
        std::cout << "Coefficient matrix and right hand side have different dimensions";
        exit(1);
    }

    std::cout << "Coefficient Matrix A with n = " << n << " and nnz(A) = " << Ai[n] << "\n\n";
    std::cout << "Coefficient Matrix    : " << argv[1] << "\n";
    std::cout << "RHS vector            : " << argv[2] << "\n\n";
    std::cout << "== CIFIM(\\epsilon_1,\\epsilon_2) ==" << "\n";
    std::cout << "\\epsilon_1            : " << dtol1 << "\n";
    std::cout << "\\epsilon_2            : " << dtol2 << "\n";
    std::cout << "Shift                 : " << shift << "\n";
    std::cout << "\\eta                  : " << eta << "\n";
    std::cout << "Elemenets per Row/Col : " << elemPerRowCol << "\n";
    std::cout << "Element Growth        : " << growth << "\n";
    std::cout << "Filtering Multiplier  : " << coeff << "\n";
    std::cout << "Element Limit         : " << element_limit << "\n";
    std::cout << "Filtering Iterations  : " << filtit << "\n\n";
    std::cout << "==           PBiCGSTAB           ==" << "\n";
    std::cout << "Maximum Iters         : " << NMAX << "\n";
    std::cout << "Tolerance             : " << tol << "\n";
    std::cout << "Verbose               : " << verbose << "\n";
    std::cout << "Mode                  : " << mode << "\n";
    std::cout << "Relaxation Iters      : " << nr << "\n\n";

    // Allocate and set initial guess
    x = new double[n];
    for (i = 0; i < n; i++)
        x[i] = 0.0;

    // Form AFIIM preconditioner
    start = omp_get_wtime();
    int cerr = acifim(n,
                      Av,
                      Aj,
                      Ai,
                      &Gv,
                      &Gj,
                      &Gi,
                      &IDv,
                      &Hv,
                      &Hj,
                      &Hi,
                      &Lv,
                      &Lj,
                      &Li,
                      &Uv,
                      &Uj,
                      &Ui,
                      dtol1,
                      dtol2,
                      elemPerRowCol,
                      growth,
                      eta,
                      shift,
                      coeff,
                      element_limit,
                      filtit);
    end = omp_get_wtime();

    // Print nonzeros of preconditioner
    double tcifim = ((double)(end - start));
    if (cerr == 0)
    {
        std::cout << "Elapsed time for computation of AFIIM          : " << tcifim << " seconds\n\n";
        std::cout << "Nonzero elements in LDU preconditioner         : " << -n + Li[n] + Ui[n] << "\n";
        std::cout << "\\rho_{LDU}                                     : " << double(-n + Li[n] + Ui[n]) / Ai[n] << "\n";
        std::cout << "Nonzero elements in GD^{-1}H preconditioner    : " << -n + Gi[n] + Hi[n] << "\n";
        std::cout << "\\rho_{GD^{-1}H}                                : " << double(-n + Gi[n] + Hi[n]) / Ai[n] << "\n";
        std::cout << "\\rho_{Total}                                   : " << double(-2 * n + Gi[n] + Hi[n] + Li[n] + Ui[n]) / Ai[n] << "\n\n";
    }
    else
    {
        std::cout << "Formation of preconditioner failed try changing the choices of parameters" << "\n";
        exit(0);
    }

    // Solve the linear system
    start = omp_get_wtime();
    PBiCGSTAB(n,
              Av,
              Aj,
              Ai,
              B,
              tol,
              NMAX,
              Gv,
              Gj,
              Gi,
              IDv,
              Hv,
              Hj,
              Hi,
              Lv,
              Lj,
              Li,
              Uv,
              Uj,
              Ui,
              nr,
              mode,
              x,
              &resval,
              &iter,
              verbose,
              &info);
    end = omp_get_wtime();
    double tbcg = ((double)(end - start));

    std::cout << "\n"
              << "Elapsed time for computation of CIFIM-PBiCSTAB : " << tbcg << " seconds\n";
    std::cout << "Total time                                     : " << tcifim + tbcg << " seconds\n";
    std::cout << "CIFIM-PBiCGSTAB info                           : " << info << "\n";
    std::cout << "Iterations to convergence                      : " << iter << "\n";

    // Write solution to file
    write_vector("solution.mtx", n, x);

    // Cleanup
    delete[] x;
    delete[] B;
    delete[] Av;
    delete[] Ai;
    delete[] Aj;
    delete[] Hv;
    delete[] Hi;
    delete[] Hj;
    delete[] Gv;
    delete[] Gi;
    delete[] Gj;
    delete[] Lv;
    delete[] Li;
    delete[] Lj;
    delete[] Uv;
    delete[] Ui;
    delete[] Uj;

    return 0;
}

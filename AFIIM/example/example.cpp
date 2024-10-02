#include <iostream>
#include <cstdlib>
#include <ctime>
#include "rw_matrix.hpp"
#include "prec.hpp"
#include "bicgstab.hpp"
#include "misc.hpp"
#include "qsort.hpp"

int main(int argc, char **argv)
{
    // Check input number
    if (argc != 4)
    {
        std::cout<<"\nThis program requires 3 command line arguments:\n";
        std::cout<<"(1) Name of the file retaining the Coefficient matrix in Matrix Market format\n";
        std::cout<<"(2) Name of the file retaining the right hand side in Matrix Market format\n";
        std::cout<<"(3) Drop tolerance (\\epsilon)\n\n";
        exit(1);
    }

    // Variables to measure execution time
    clock_t start, end;

    // Variables corresponding to the coefficient matrix
    double *Av;
    int *Ai,*Aj;
    int n;

    // Variables corresponding to the right hand side
    double *B;
    int m;

    // Variables corresponding to the AFIIM preconditioner
    double *Gv,*Hv,*IDv;
    int *Gi,*Gj,*Hi,*Hj;
    double dtol = atof(argv[3]); 
    int elemPerRowCol = 10, growth = 5;
    double eta = 0.0, shift = 0.0;

    // EPBiCGSTAB related variables
    int NMAX = 500, iter, info;
    double tol = 1e-8, resval,*x;

    // Iterator
    int i;

    // Read coefficient matrix from file
    read_coeff_matrix(argv[1], 
                     &n, 
                     &Av, 
                     &Aj, 
                     &Ai);

    // Read right hand side from file
    read_rhs_vector(argv[2], 
                     &m, 
                     &B);

    // Check if dimensions agree
    if (n != m)
    {
        std::cout<<"Coefficient matrix and right hand side have different dimensions";
        exit(1);
    }

    std::cout<<"Coefficient Matrix A with n = "<<n<<" and nnz(A) = "<<Ai[n]<<"\n\n";


    // Allocate and set initial guess
    x = new double[n];
    for(i=0;i<n;i++) x[i] = 0.0;
    
    // Form AFIIM preconditioner
    start = clock();
    afiim(n, 
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
          dtol, 
          elemPerRowCol, 
          growth, 
          eta,
          shift);
    end = clock();

    // Print nonzeros of preconditioner
    std::cout<<"Elapsed time for computation of AFIIM: "<<((double) (end - start)) / CLOCKS_PER_SEC<<" seconds\n";
    std::cout<<"Nonzero elements in preconditioner (nnz(G)+nnz(H)+nnz(D)): "<<n + Gi[n] + Hi[n]<<"\n\n";
    
    // Solve the linear system
    start = clock();
    EPBiCGSTAB(n, 
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
               x,
               &resval,
               &iter,
               1,
               &info);
    end = clock();
    std::cout<<"Elapsed time for computation of AFIIM-EPBiCSTAB: "<<((double) (end - start)) / CLOCKS_PER_SEC<<" seconds\n";
    

    // Write solution to file
    write_vector("solution.mtx",n,x);

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

    return 0;
}
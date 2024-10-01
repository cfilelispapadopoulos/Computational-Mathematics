#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "rw_matrix.h"
#include "prec.h"
#include "bicgstab.h"
#include "misc.h"

int main(int argc, char **argv)
{
    // Check input number
    if (argc != 4)
    {
        printf("\nThis program requires 3 command line arguments:\n");
        printf("(1) Name of the file retaining the Coefficient matrix in Matrix Market format\n");
        printf("(2) Name of the file retaining the right hand side in Matrix Market format\n");
        printf("(3) Drop tolerance (\\epsilon)\n\n");
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
        printf("Coefficient matrix and right hand side have different dimensions");
        exit(1);
    }

    printf("Coefficient Matrix A with n = %d and nnz(A) = %d\n\n",n,Ai[n]);


    // Allocate and set initial guess
    x = (double *)malloc(n * sizeof(double));
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
    printf("Elapsed time for computation of AFIIM: %lf seconds\n", ((double) (end - start)) / CLOCKS_PER_SEC);
    printf("Nonzero elements in preconditioner (nnz(G)+nnz(H)+nnz(D)): %d\n\n",n + Gi[n] + Hi[n]);
    
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
    printf("Elapsed time for computation of AFIIM-EPBiCSTAB: %lf seconds\n", ((double) (end - start)) / CLOCKS_PER_SEC);
    

    // Write solution to file
    write_vector("solution.mtx",n,x);

    // Cleanup
    free(x);
    free(B);
    free(Av);
    free(Ai);
    free(Aj);
    free(Hv);
    free(Hi);
    free(Hj);
    free(Gv);
    free(Gi);
    free(Gj);

    return 0;
}
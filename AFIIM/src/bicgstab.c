#include "bicgstab.h"
#include "minmax.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <mkl.h>
#include <mkl_spblas.h>

// DISCLAIMER
// The use of the code is regulated by the following copyright agreement.
//
// AFIIM software is freely available for scientific (non-commercial) use.
// 1. The code can be used only for the purpose of internal research, excluding any commercial use of the AFIIM 
//    software as such or as a part of a software product. Users who want to integrate AFIIM sofware or parts of  
//    it into commercial products require a license agreement.
// 2. AFIIM is provided "as is" and for the purpose described at the previous point only. In no circumstances can
//    neither the authors nor their institutions be held liable for any deficiency, fault or other mishappening 
//    with regard to the use or performance of AFIIM.
// 3. All scientific publications, for which AFIIM software has been used, shall mention its usage and refer to 
//    the publication [1] in the References section below.
//
//
// References
// [1] C. K. Filelis - Papadopoulos (2024). Adaptive Factored Incomplete Inverse Matrices. In Review.



// EPBiCGSTAB - Explicit Preconditioned Bi-Conjugate Gradient Stabilized
//
// The Explicit Preconditioned Bi-Conjugate Gradient Stabilized is a smoothly converging Krylov subspace 
// iterative method for nonsymmetric linear systems and was proposed by H.A. van der Vorst [2]. The vesion
// used below is modified for supporting AFIIM preconditioning of the form y = G D^{-1} H, where G and H 
// are sparse factors stored in CSR format and D^{-1} is a dense vector retaining the diagonal elements [2].
//
//
// Author: Christos K. Papadopoulos Filelis
//         Assistant Professor
//         Democritus University of Thrace
//         Department of Electrical and Computer Engineering
//         Xanthi, Greece, GR 67100
//         email: cpapad@ee.duth.gr
//
// ---------------------- Arguments -------------------------------------------------------------------------
// INPUT
// NAME             TYPE                DESCRIPTION
// n                (integer)           Size of square general sparse matrix A.
// Av               (double*)           Vector of size nnz(A) retaining the values of A (CSR).
// Aj               (integer*)          Vector of size nnz(A) retaining the column indices of the 
//                                      elements of matrix A (CSR).
// Ai               (integer*)          Vector of size n+1 retaining the offsets of the rows of A (CSR).
// b                (double*)           Preallocated right hand side (rhs) vector of size n.
// tol              (double)            Prescribed termination tolerance for the relative residual termination
//                                      criterion ||r_i||_2 < tol ||r_0||_2. A good starting value is 1e-8.
// Gv               (double*)           Vector of size nnz(G) retaining the values of G (CSR).
// Gj               (integer*)          Vector of size nnz(G) retaining the column indices of the 
//                                      elements of matrix G (CSR).
// Gi               (integer*)          Vector of size n+1 retaining the offsets of the rows of G (CSR).
// IDv              (double *)          Vector of size n retaining the diagonal elements of factor D^{-1}.
// Hv               (double*)           Vector of size nnz(H) retaining the values of H (CSR).
// Hj               (integer*)          Vector of size nnz(H) retaining the column indices of the 
//                                      elements of matrix H (CSR).
// Hi               (integer*)          Vector of size n+1 retaining the offsets of the rows of H (CSR).
// x                (double*)           Preallocated vector of size n retaining the initial guess. A good initial 
//                                      guess is x = \vec{0}. Upon termination the vector retains the approximation
//                                      to the solution of the sparse linear system.
// verbose          (integer)           Controls verbosity. 0: Zero printing, 1: Print everything.
//
// OUTPUT
// NAME             TYPE                DESCRIPTION
// resval           (double*)           Variable retaining the relative residual ||b-A x_i||_2 / ||b-A x_i||_0 
//                                      upon termination of the method.
// iter             (integer)           Variable retaining the number of iterations performed until termination.
//                                      In case of stagnation or divergence the iter retains the last iteration
//                                      at which the phenomenon was detected.
// info             (integer)           Variable retaining the termination status. 0: method conveged to prescribed
//                                      tolerance, 1: method diverged, 2: method stagnated, 3: method did not
//                                      converge withing the prescribed number of maximum allowed iterations.
//
// 
// References
// [1] H.A. van der Vorst (1992). Bi-cgstab: A fast and smoothly converging variant of bi-cg for the solution of 
//     nonsymmetric linear systems. SIAM Journal on Scientific and Statistical Computing, 13(2):631–644.
//     doi:10.1137/0913035.
// [2] C. K. Filelis - Papadopoulos (2024). Adaptive Factored Incomplete Inverse Matrices. In Review.

void EPBiCGSTAB(int n, 
                double *Av, 
                int *Aj, 
                int *Ai, 
                double *b, 
                double tol, 
                int NMAX, 
                double *Gv, 
                int *Gj, 
                int *Gi, 
                double *IDv, 
                double *Hv, 
                int *Hj, 
                int *Hi,
                double *x,
                double *resval,
                int *iter,
                int verbose,
                int *info)
{
    // Machile precision (double)
	double eps=1.11022302462516e-16;

    // Variables
    // Maximum stagnation steps allowed
    int maxstagsteps=3,stag=0;
    // Max divergence steps allowed
    int maxmsteps=mmin(((n>500)?(n):500)/50,20),msteps=0;
    // Variables retaining norms etc
    double prc;
    double ro,a,om;
    double m1,m2,rc,beta;
    

    // Vectors required by the method
    double *p  = (double *)malloc(n*sizeof(double));
    double *r0 = (double *)malloc(n*sizeof(double));
    double *r  = (double *)malloc(n*sizeof(double));
    double *v  = (double *)malloc(n*sizeof(double));
    double *s  = (double *)malloc(n*sizeof(double));
    double *t  = (double *)malloc(n*sizeof(double));
    double *pc = (double *)malloc(n*sizeof(double));
    double *sc = (double *)malloc(n*sizeof(double));
    double *tm = (double *)malloc(n*sizeof(double));
	
    // Initialize iteration counter
    (*iter)=0;

    // Create the required structures for MKL
    sparse_matrix_t A,G,H;
    struct matrix_descr descr = {SPARSE_MATRIX_TYPE_GENERAL};
    mkl_sparse_d_create_csr (&A, SPARSE_INDEX_BASE_ZERO, n, n, Ai, Ai+1, Aj, Av);
    mkl_sparse_d_create_csr (&G, SPARSE_INDEX_BASE_ZERO, n, n, Gi, Gi+1, Gj, Gv);
    mkl_sparse_d_create_csr (&H, SPARSE_INDEX_BASE_ZERO, n, n, Hi, Hi+1, Hj, Hv);

    // Compute residual r = b - A x
    //mkl_cspblas_dcsrgemv ("N", &n, Av, Ai, Aj, x, r); - Deprecated version
    mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE,1.0,A,descr,x,0.0,r);
    
	cblas_dcopy(n,b,1,p,1);
	cblas_daxpy(n,-1.0,r,1,p,1);

    // Set r0 = r = p using p which retains the residual
    cblas_dcopy(n,p,1,r0,1);
	cblas_dcopy(n,p,1,r,1);

    // Set ro = || r0 ||_2 and m1 = (r,r0)
	ro=cblas_dnrm2(n,r0,1);
	m1=ro*ro;

    // Set current norm of residual equal to the initial
    rc=ro;
    
    // Begin iterations until maximum is reached
    while ((*iter)<NMAX)
    {
        // p_c = G D^{-1} H p
        // mkl_cspblas_dcsrgemv ("N", &n, Hv, Hi, Hj, p, pc); - Deprecated version
        mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE,1.0,H,descr,p,0.0,pc);
        vdMul( n, IDv, pc, tm );
        // mkl_cspblas_dcsrgemv ("N", &n, Gv, Gi, Gj, tm, pc); - Deprecated version
        mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE,1.0,G,descr,tm,0.0,pc);
        // v = A p_c
        // mkl_cspblas_dcsrgemv ("N", &n, Av, Ai, Aj, pc, v); - Deprecated version
        mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE,1.0,A,descr,pc,0.0,v);
        
        // alpha = m1 / (r_0,v)
        a=m1/cblas_ddot(n,r0,1,v,1);

        // s = r - alpha v
		cblas_dcopy(n,r,1,s,1);
		cblas_daxpy(n,-a,v,1,s,1);
        
        // s_c = G D^{-1} H s
        // mkl_cspblas_dcsrgemv ("N", &n, Hv, Hi, Hj, s, sc); - Deprecated version
        mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE,1.0,H,descr,s,0.0,sc);
        vdMul( n, IDv, sc, tm);
        // mkl_cspblas_dcsrgemv ("N", &n, Gv, Gi, Gj, tm, sc); - Deprecated version
        mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE,1.0,G,descr,tm,0.0,sc);
        // t = A S_c
        // mkl_cspblas_dcsrgemv ("N", &n, Av, Ai, Aj, sc, t); - Derpecated version
        mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE,1.0,A,descr,sc,0.0,t);
        
        // omega = (t,s) / (t,t)
        om=cblas_ddot(n,t,1,s,1)/cblas_ddot(n,t,1,t,1);

        // Check for stagnation and increase/reset counter
        if (fabs(a)*cblas_dnrm2(n,pc,1)<eps*cblas_dnrm2(n,x,1))
            stag++;
        else
            stag=0;
        
        // x = x + alpha p_c
        cblas_daxpy(n,a,pc,1,x,1);

        // x = x + omega s_c
		cblas_daxpy(n,om,sc,1,x,1);
        
        // Store previous residual norm
        prc=rc;

        // Compute new residual norm r_c = || r ||_2
        rc=cblas_dnrm2(n,r,1);
   
        // Check for divergence and increase/reset counter
        if (prc<rc)
            msteps++;
        else
            msteps=0;
        
        // Check if maximum allowed steps of divergence have been reached and terminate if so
        if (msteps>=maxmsteps)
        {
            if (verbose)
            {
                printf("BiCGSTAB diverged and iterations stopped with relative residual %e in iteration %d\n",rc/ro,*iter);
                printf("Try to increase the quality of preconditioning\n");
            }
            (*info)=1;
            break;
        }
        
        // Check if maximum allowed steps for stagnation have been reached and terminate id so
        if (stag>=maxstagsteps)
        {
            if (verbose)
            {
                printf("BiCGSTAB stagnated with relative residual %e in iteration %d\n",rc/ro,*iter);
                printf("Try increasing prescribed tolerance\n");
            }
            (*info)=2;
            break;
        }
        
        // Check for convergence to the prescribed tolerance
        if (rc<tol*ro)
        {
            if (verbose)
                printf("Convergence in %d iterations with relative residual %e\n",*iter,rc/ro);
            (*info)=0;
		    break;
        }
		
        // r = s - omega t
        cblas_dcopy(n,s,1,r,1);
		cblas_daxpy(n,-om,t,1,r,1);

        // Retain old value of m1 in m2
        m2=m1;
        // Compute new m1 = (r0,r)
        m1=cblas_ddot(n,r0,1,r,1);

        // beta = m1 alpha / (omega m2)
        beta=m1*a/(om*m2);

        // p = r + beta (p - omega v)
		cblas_daxpby(n,-beta*om,v,1,beta,p,1);
		cblas_daxpy(n,1.0,r,1,p,1);

        // Increase iteration counter
        (*iter)++;
    }
    // Maximum iterations are reached print failure message
    if((*iter)==NMAX)
    {
        if (verbose)
		    printf("Convergence failed after %d iterations with relative residual %e\n",(*iter),rc/ro);
        (*info) = 3;
	}

    // Store final relative residual
    (*resval) = rc / ro;

    free(p);
    free(r0);
    free(r);
    free(v);
    free(s);
    free(t);
    free(pc);
    free(sc);
    free(tm);
    
}
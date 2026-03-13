#include "bicgstab.hpp"
#include "minmax.hpp"
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <mkl.h>
#include <mkl_spblas.h>

// DISCLAIMER
// The use of the code is regulated by the following copyright agreement.
//
// ACIFIM software is freely available for scientific (non-commercial) use.
// 1. The code can be used only for the purpose of internal research, excluding any commercial use of the ACIFIM 
//    software as such or as a part of a software product. Users who want to integrate ACIFIM sofware or parts of  
//    it into commercial products require a license agreement.
// 2. ACIFIM is provided "as is" and for the purpose described at the previous point only. In no circumstances can
//    neither the authors nor their institutions be held liable for any deficiency, fault or other mishappening 
//    with regard to the use or performance of ACIFIM.
// 3. All scientific publications, for which ACIFIM software has been used, shall mention its usage and refer to 
//    the publication [1] in the References section below.
//
//
// References
// [1] C. K. Filelis - Papadopoulos (2026). Adaptive Combined Incomplete Factorizations and Inverse Matrices. To be submitted.



// PBiCGSTAB - Preconditioned Bi-Conjugate Gradient Stabilized
//
// The Preconditioned Bi-Conjugate Gradient Stabilized is a smoothly converging Krylov subspace 
// iterative method for nonsymmetric linear systems and was proposed by H.A. van der Vorst [2]. The vesion
// used below is modified for supporting CIFIM preconditioning of the form y = G D^{-1} H, where G and H 
// are sparse factors stored in CSR format and D^{-1} is a dense vector retaining the diagonal elements 
// coupled with LDU preconditioning through the preconditioned Richardson's iteration [2].
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
// Lv               (double*)           Vector of size nnz(L) retaining the values of L (CSR).
// Lj               (integer*)          Vector of size nnz(L) retaining the column indices of the 
//                                      elements of matrix L (CSR).
// Li               (integer*)          Vector of size n+1 retaining the offsets of the rows of L (CSR).
// Uv               (double*)           Vector of size nnz(U) retaining the values of U (CSR).
// Uj               (integer*)          Vector of size nnz(U) retaining the column indices of the 
//                                      elements of matrix U (CSR).
// Ui               (integer*)          Vector of size n+1 retaining the offsets of the rows of U (CSR).
// nr               (integer)           Number of relaxation steps
// mode             (char)              Mode for preconditioning ('I': implicit, 'E': explicit, 'C': combined).
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
// [2] C. K. Filelis - Papadopoulos (2026). Adaptive Combined Incomplete Factorizations and Inverse Matrices. To be submitted.

void PBiCGSTAB(int n, 
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
               double *Lv, 
               int *Lj, 
               int *Li,
               double *Uv, 
               int *Uj, 
               int *Ui,
               int nr,
               char mode,                            
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
    int maxmsteps=mmin(((n>2000)?(n):2000)/50,40),msteps=0;
    // Variables retaining norms etc
    double prc;
    double ro,a,om;
    double m1,m2,rc,beta;
    

    // Vectors required by the method
    double *p  = new double[n];
    double *r0 = new double[n];
    double *r  = new double[n];
    double *v  = new double[n];
    double *s  = new double[n];
    double *t  = new double[n];
    double *pc = new double[n];
    double *sc = new double[n];
    double *tm = new double[n];

    // Work vector for preconditioning
    double *work = new double[3*n];
	
    // Initialize iteration counter
    (*iter)=0;

    // Create the required structures for MKL
    sparse_matrix_t A,G,H,U,L;
    struct matrix_descr descrA,descrG,descrH,descrU,descrL;
    mkl_sparse_d_create_csr (&A, SPARSE_INDEX_BASE_ZERO, n, n, Ai, Ai+1, Aj, Av);
    descrA.type = SPARSE_MATRIX_TYPE_GENERAL;
    mkl_sparse_d_create_csr (&G, SPARSE_INDEX_BASE_ZERO, n, n, Gi, Gi+1, Gj, Gv);
    descrG.type = SPARSE_MATRIX_TYPE_TRIANGULAR;
    descrG.mode = SPARSE_FILL_MODE_UPPER;
    descrG.diag = SPARSE_DIAG_NON_UNIT;
    mkl_sparse_d_create_csr (&H, SPARSE_INDEX_BASE_ZERO, n, n, Hi, Hi+1, Hj, Hv);
    descrH.type = SPARSE_MATRIX_TYPE_TRIANGULAR;
    descrH.mode = SPARSE_FILL_MODE_LOWER;
    descrH.diag = SPARSE_DIAG_NON_UNIT;
    mkl_sparse_d_create_csr (&U, SPARSE_INDEX_BASE_ZERO, n, n, Ui, Ui+1, Uj, Uv);
    descrU.type = SPARSE_MATRIX_TYPE_TRIANGULAR;
    descrU.mode = SPARSE_FILL_MODE_UPPER;
    descrU.diag = SPARSE_DIAG_NON_UNIT;
    mkl_sparse_d_create_csr (&L, SPARSE_INDEX_BASE_ZERO, n, n, Li, Li+1, Lj, Lv);
    descrL.type = SPARSE_MATRIX_TYPE_TRIANGULAR;
    descrL.mode = SPARSE_FILL_MODE_LOWER;
    descrL.diag = SPARSE_DIAG_NON_UNIT;        

    // Compute residual r = b - A x
    //mkl_cspblas_dcsrgemv ("N", &n, Av, Ai, Aj, x, r); - Deprecated version
    mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE,1.0,A,descrA,x,0.0,r);
    
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
        // p_c = M p
        apply(n,nr,&G,&descrG,IDv,&H,&descrH,&L,&descrL,&U,&descrU,p,pc,work,mode);
        // v = A p_c
        // mkl_cspblas_dcsrgemv ("N", &n, Av, Ai, Aj, pc, v); - Deprecated version
        mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE,1.0,A,descrA,pc,0.0,v);

        // alpha = m1 / (r_0,v)
        a=m1 / cblas_ddot(n,r0,1,v,1);

	if (!std::isfinite(a))
	{
	    	(*iter) = NMAX;
	        (*info) = 3;   
		break; 
	}

	// s = r - alpha v
	cblas_dcopy(n,r,1,s,1);
	cblas_daxpy(n,-a,v,1,s,1);
        
        // s_c = M s
        apply(n,nr,&G,&descrG,IDv,&H,&descrH,&L,&descrL,&U,&descrU,s,sc,work,mode);

        // t = A S_c
        // mkl_cspblas_dcsrgemv ("N", &n, Av, Ai, Aj, sc, t); - Derpecated version
        mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE,1.0,A,descrA,sc,0.0,t);
        
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
                std::cout<<"BiCGSTAB diverged and iterations stopped with relative residual "<<rc/ro<<" in iteration "<<*iter<<"\n";
                std::cout<<"Try to increase the quality of preconditioning\n";
            }
            (*info)=1;
            break;
        }
        
        // Check if maximum allowed steps for stagnation have been reached and terminate id so
        if (stag>=maxstagsteps)
        {
            if (verbose)
            {
                std::cout<<"BiCGSTAB stagnated with relative residual "<<rc/ro<<" in iteration "<<*iter<<"\n";
                std::cout<<"Try increasing prescribed tolerance\n";
            }
            (*info)=2;
            break;
        }
        
        // Check for convergence to the prescribed tolerance
        if (rc<tol*ro)
        {
            if (verbose)
                std::cout<<"Convergence in "<<*iter<<" iterations with relative residual "<<rc/ro<<"\n";
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
		    std::cout<<"Convergence failed after "<<*iter<<" iterations with relative residual "<<rc/ro<<"\n";
        (*info) = 3;
	}

    // Store final relative residual
    (*resval) = rc / ro;

    delete[] p;
    delete[] r0;
    delete[] r;
    delete[] v;
    delete[] s;
    delete[] t;
    delete[] pc;
    delete[] sc;
    delete[] tm;
    delete[] work;
    
}

// apply - Preconditioned Richardson's Iteration used as preconditioner
//
// The Preconditioned Richardson's iterations is used in order to apply the \alpha CIFIM preconditioner. 
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
// n                (integer)           Size of square general sparse matrices.
// nr               (integer)           Number of relaxation steps
// G                (sparse_matrix_t*)  MKL Structure retaining factor G (CSR).
// descrG           (matrix_descr*)     Matrix descriptor for factor G.
// IDv              (double *)          Vector retaining the diagonal elements.
// H                (sparse_matrix_t*)  MKL Structure retaining factor H (CSR).
// descrH           (matrix_descr*)     Matrix descriptor for factor H.
// L                (sparse_matrix_t*)  MKL Structure retaining factor L (CSR).
// descrL           (matrix_descr*)     Matrix descriptor for factor L.
// U                (sparse_matrix_t*)  MKL Structure retaining factor U (CSR).
// descrU           (matrix_descr*)     Matrix descriptor for factor U.
// b                (double*)           Preallocated right hand side (rhs) vector of size n.
// x                (double*)           Preallocated vector of size n retaining the initial guess. A good initial 
//                                      guess is x = \vec{0}. Upon termination the vector retains the approximation
//                                      to the solution of the sparse linear system.
// work             (double*)           Work vector of size n required by the iterative method.
// mode             (char)              Mode for preconditioning ('I': implicit, 'E': explicit, 'C': combined).
//
// OUTPUT
// NAME             TYPE                DESCRIPTION
// x                (double*)           Preallocated vector of size n retaining the initial guess. A good initial 
//                                      guess is x = \vec{0}. Upon termination the vector retains the approximation
//                                      to the solution of the sparse linear system.
//
// 
// References
// [1] C. K. Filelis - Papadopoulos (2026). Adaptive Combined Incomplete Factorizations and Inverse Matrices. To be submitted.

void apply(int n,
           int nr, 
           sparse_matrix_t *G,
           matrix_descr *descrG, 
           double *IDv, 
           sparse_matrix_t *H,
           matrix_descr *descrH,
           sparse_matrix_t *L,
           matrix_descr *descrL,
           sparse_matrix_t *U,
           matrix_descr *descrU,
           double *b,
           double *x,
           double *work,
           char mode)
{
    
    double *t1 = work;
    double *t2 = work+n;
    double *t3 = work+2*n;
    if (mode!='J'){
    if (mode=='E')
        mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE,1.0,*H,*descrH,b,0.0,t1);
    else if (mode=='C')
    {
        mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE,1.0,*H,*descrH,b,0.0,t1);
        for(int i = 1; i < nr; i++)
        {
            mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE,1.0,*L,*descrL,t1,0.0,t2);
            vdSub(n,b,t2,t3);
            mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE,1.0,*H,*descrH,t3,1.0,t1);
        }        
    }
    else
        mkl_sparse_d_trsv (SPARSE_OPERATION_NON_TRANSPOSE, 1.0, *L, *descrL, b, t1);
    
    vdMul(n,t1,IDv,t2);
    if (mode=='E')
        mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE,1.0,*G,*descrG,t2,0.0,x);
    else if (mode=='C')
    {
        mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE,1.0,*G,*descrG,t2,0.0,x);
        for(int i = 1; i < nr; i++)
        {
            mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE,1.0,*U,*descrU,x,0.0,t1);
            vdSub(n,t2,t1,t3);
            mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE,1.0,*G,*descrG,t3,1.0,x);
        }         
    }
    else
        mkl_sparse_d_trsv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, *U, *descrU, t2, x);
    } else
    {
	mkl_sparse_d_trsv (SPARSE_OPERATION_NON_TRANSPOSE, 1.0, *L, *descrL, b, t1);
	mkl_sparse_d_trsv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, *U, *descrU, t1, x);
    }  
}

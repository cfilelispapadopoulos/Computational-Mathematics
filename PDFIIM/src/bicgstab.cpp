#include "bicgstab.hpp"
#include "minmax.hpp"
#include "sparseMatrix.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <mkl.h>
#include <mkl_spblas.h>

// DISCLAIMER
// The use of the code is regulated by the following copyright agreement.
//
// PDFIIM software is freely available for scientific (non-commercial) use.
// 1. The code can be used only for the purpose of internal research, excluding any commercial use of the PDFIIM 
//    software as such or as a part of a software product. Users who want to integrate PDFIIM sofware or parts of  
//    it into commercial products require a license agreement.
// 2. PDFIIM is provided "as is" and for the purpose described at the previous point only. In no circumstances can
//    neither the authors nor their institutions be held liable for any deficiency, fault or other mishappening 
//    with regard to the use or performance of PDFIIM.
// 3. All scientific publications, for which PDFIIM software has been used, shall mention its usage and refer to 
//    the publication [1] in the References section below.
//
//
// References
// [1] C. K. Filelis - Papadopoulos and G. A. Gravvanis (2025). Parallel sparsity patterns for factored incomplete inverse matrices,
//     Journal of Computational Science, Volume 93, 2026, 102736, ISSN 1877-7503, doi: 10.1016/j.jocs.2025.102736.



// EPBiCGSTAB - Explicit Preconditioned Bi-Conjugate Gradient Stabilized
//
// The Explicit Preconditioned Bi-Conjugate Gradient Stabilized is a smoothly converging Krylov subspace 
// iterative method for nonsymmetric linear systems and was proposed by H.A. van der Vorst [2]. The vesion
// used below is modified for supporting PDFIIM preconditioning of the form y = G D^{-1} H, where G and H 
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
// A                (sparseMatrix)      Coeffient matrix of class sparseMatrix (CSR).
// b                (vector)            Preallocated right hand side (rhs) double vector of size n.
// tol              (double)            Prescribed termination tolerance for the relative residual termination
//                                      criterion ||r_i||_2 < tol ||r_0||_2. A good starting value is 1e-8.
// G                (sparseMatrix)      Upper triangular factor of preconditioner of class sparseMatrix (CSR).
// ID               (vector)            Vector of size n retaining the diagonal elements of factor D^{-1}.
// H                (sparseMatrix)      Lower triangular factor of preconditioner of class sparseMatrix (CSR).
// x                (vector)            Preallocated vector of size n retaining the initial guess. A good initial 
//                                      guess is x = \vec{0}. Upon termination the vector retains the approximation
//                                      to the solution of the sparse linear system.
// verbose          (integer)           Controls verbosity. 0: Zero printing, 1: Print everything.
//
// OUTPUT
// NAME             TYPE                DESCRIPTION
// resval           (double*)           Variable retaining the relative residual ||b-A x_i||_2 / ||b-A x_0||_2 
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

void EPBiCGSTAB(sparseMatrix<int,int,double> &A,
                std::vector<double> &b, 
                double tol, 
                int NMAX, 
                sparseMatrix<int,int,double> &G, 
                std::vector<double> &ID, 
                sparseMatrix<int,int,double> &H,
                std::vector<double> &x, 
                double *resval,
                int *iter,
                int verbose,
                int *info)
{
    // Matrix size
    int n = A.r;

    // Variable regarding symmetry
    int issym = int(A.mtype == SYMMETRIC);

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
    
    // Pointers required by MKL
    int *Ai = &(A.i[0]),*Aj = &(A.j[0]);
    double *Av = &(A.v[0]);
    int *Gi = &(G.i[0]),*Gj = &(G.j[0]);
    double *Gv = &(G.v[0]);
    int *Hi,*Hj;
    double *Hv;
    if(!issym)
    {
        Hi = &(H.i[0]);Hj = &(H.j[0]);
        Hv = &(H.v[0]);
    }

    // Vectors required by the method   
    std::vector<double> pv(n);double *p = &pv[0];
    std::vector<double> r0v(n);double *r0 = &r0v[0];
    std::vector<double> rv(n);double *r = &rv[0];
    std::vector<double> vv(n);double *v = &vv[0];
    std::vector<double> sv(n);double *s = &sv[0];
    std::vector<double> tv(n);double *t = &tv[0];
    std::vector<double> pcv(n);double *pc = &pcv[0];
    std::vector<double> scv(n);double *sc = &scv[0];
    std::vector<double> tmv(n);double *tm = &tmv[0];
    double *bb = &b[0];
    double *xx = &x[0];
    double *IDv = &ID[0];
	
    // Initialize iteration counter
    (*iter)=0;

    // Create the required structures for MKL
    sparse_matrix_t AA,GG,HH;
    struct matrix_descr descrA,descrG,descrH;
    mkl_sparse_d_create_csr (&AA, SPARSE_INDEX_BASE_ZERO, n, n, Ai, Ai+1, Aj, Av);
    if (!issym)
        descrA.type = SPARSE_MATRIX_TYPE_GENERAL;
    else
    {
        descrA.type = SPARSE_MATRIX_TYPE_SYMMETRIC;
        descrA.mode = SPARSE_FILL_MODE_UPPER;
        descrA.diag = SPARSE_DIAG_NON_UNIT;    
    }

    mkl_sparse_d_create_csr (&GG, SPARSE_INDEX_BASE_ZERO, n, n, Gi, Gi+1, Gj, Gv);
    descrG.type = SPARSE_MATRIX_TYPE_TRIANGULAR;
    descrG.mode = SPARSE_FILL_MODE_UPPER;
    descrG.diag = SPARSE_DIAG_NON_UNIT;
    if (!issym)
    {
        mkl_sparse_d_create_csr (&HH, SPARSE_INDEX_BASE_ZERO, n, n, Hi, Hi+1, Hj, Hv);
        descrH.type = SPARSE_MATRIX_TYPE_TRIANGULAR;
        descrH.mode = SPARSE_FILL_MODE_LOWER;
        descrH.diag = SPARSE_DIAG_NON_UNIT;
    }
    

    // Compute residual r = b - A x
    mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE,1.0,AA,descrA,xx,0.0,r);
    
	cblas_dcopy(n,bb,1,p,1);
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
        if (issym)
            mkl_sparse_d_mv(SPARSE_OPERATION_TRANSPOSE,1.0,GG,descrG,p,0.0,pc);
        else
            mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE,1.0,HH,descrH,p,0.0,pc);
        vdMul( n, IDv, pc, tm );
        mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE,1.0,GG,descrG,tm,0.0,pc);
        // v = A p_c
        mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE,1.0,AA,descrA,pc,0.0,v);
        
        // alpha = m1 / (r_0,v)
        a=m1/cblas_ddot(n,r0,1,v,1);

        // s = r - alpha v
		cblas_dcopy(n,r,1,s,1);
		cblas_daxpy(n,-a,v,1,s,1);
        
        // s_c = G D^{-1} H s
        if (issym)
            mkl_sparse_d_mv(SPARSE_OPERATION_TRANSPOSE,1.0,GG,descrG,s,0.0,sc);
        else
            mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE,1.0,HH,descrH,s,0.0,sc);
        vdMul( n, IDv, sc, tm);
        mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE,1.0,GG,descrG,tm,0.0,sc);
        // t = A S_c
        mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE,1.0,AA,descrA,sc,0.0,t);
        
        // omega = (t,s) / (t,t)
        om=cblas_ddot(n,t,1,s,1)/cblas_ddot(n,t,1,t,1);

        // Check for stagnation and increase/reset counter
        if (fabs(a)*cblas_dnrm2(n,pc,1)<eps*cblas_dnrm2(n,xx,1))
            stag++;
        else
            stag=0;
        
        // x = x + alpha p_c
        cblas_daxpy(n,a,pc,1,xx,1);

        // x = x + omega s_c
		cblas_daxpy(n,om,sc,1,xx,1);
        
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
                std::cout<<"BiCGSTAB diverged and iterations stopped with relative residual "<<rc/ro<<" in iteration "<<*iter<<"\n";
                std::cout<<"Try to increase the quality of preconditioning\n";
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

    // Cleanup
    mkl_sparse_destroy(AA);
    mkl_sparse_destroy(GG);
    mkl_sparse_destroy(HH);
   
}

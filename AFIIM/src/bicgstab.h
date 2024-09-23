#ifndef BICGSTAB_H
#define BICGSTAB_H

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
                int *info);

#endif
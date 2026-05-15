#ifndef BICGSTAB_HPP
#define BICGSTAB_HPP
#include "sparseMatrix.hpp"
#include <vector>
#include <mkl.h>
#include <mkl_spblas.h>

// DISCLAIMER
// The use of the code is regulated by the following copyright agreement.
//
// PCIILUS software is freely available for scientific (non-commercial) use.
// 1. The code can be used only for the purpose of internal research, excluding any commercial use of the PCIILUS
//    software as such or as a part of a software product. Users who want to integrate PCIILUS sofware or parts of
//    it into commercial products require a license agreement.
// 2. PCIILUS is provided "as is" and for the purpose described at the previous point only. In no circumstances can
//    neither the authors nor their institutions be held liable for any deficiency, fault or other mishappening
//    with regard to the use or performance of PCIILUS.
// 3. All scientific publications, for which PCIILUS software has been used, shall mention its usage and refer to
//    the publication [1] in the References section below.
//
//
// References
// [1] C. K. Filelis - Papadopoulos (2026). Parallel Incomplete LU Factorization, Submitted.

// PBiCGSTAB - Preconditioned Bi-Conjugate Gradient Stabilized
//
// The Preconditioned Bi-Conjugate Gradient Stabilized is a smoothly converging Krylov subspace
// iterative method for nonsymmetric linear systems and was proposed by H.A. van der Vorst [2]. The vesion
// used below is modified for supporting PCIILUS preconditioning.
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
// L                (sparseMatrix)      Lower triangular factor of preconditioner of class sparseMatrix (LDU) (CSR).
// H                (sparseMatrix)      Lower triangular factor of preconditioner of class sparseMatrix (CSR).
// ID               (vector)            Vector of size n retaining the diagonal elements of factor D^{-1}.
// U                (sparseMatrix)      Upper triangular factor of preconditioner of class sparseMatrix (LDU) (CSR).
// G                (sparseMatrix)      Upper triangular factor of preconditioner of class sparseMatrix (CSR).
// nr               (int)               Number of iterations for preconditioning e.g. 2.
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

void PBiCGSTAB(sparseMatrix<int, int, double> &A,
               std::vector<double> &b,
               double tol,
               int NMAX,
               sparseMatrix<int, int, double> &L,
               sparseMatrix<int, int, double> &H,
               std::vector<double> &D,
               sparseMatrix<int, int, double> &U,
               sparseMatrix<int, int, double> &G,
               int nr,
               std::vector<double> &x,
               double *resval,
               int *iter,
               int verbose,
               int *info);
               
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
           double *work);

#endif

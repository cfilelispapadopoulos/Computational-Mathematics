#include "misc.hpp"
#include <iostream>
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



// form_model_rhs - Function that forms a model right hand side corresponding to a sparse coefficient matrix
//                  stored in CSR storage format, such that the solution of the linear system is a vector
//                  with all its components set to 1.
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
//
// OUTPUT
// NAME             TYPE                DESCRIPTION
// y                (double**)          Vector retaining the right hand side. Should be unallocated upon
//                                      call.


void form_model_rhs(int n, double *Av, int *Aj, int *Ai, double **y)
{
    // Iterator
    int i;

    // Allocate vector with all its components set to 1
    double *x = new double[n];
    
    // Allocate output vector
    (*y) = new double[n];

    // Initialize x
    for(i=0;i<n;i++) x[i] = 1.0;

    // Perform sparse matrix by vector multiplication
    sparse_matrix_t A;
    struct matrix_descr descr;
    mkl_sparse_d_create_csr (&A, SPARSE_INDEX_BASE_ZERO, n, n, Ai, Ai+1, Aj, Av);
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;
    mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE,1.0,A,descr,x,0.0,(*y));    

    //Clean up
    delete[] x;
}

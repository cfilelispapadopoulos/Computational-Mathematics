#include "misc.hpp"
#include <iostream>
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
// A                (sparseMatrix)      Sparse matrix stored in a sparseMatrix class (CSR,CSC,COO).
//
// OUTPUT
// NAME             TYPE                DESCRIPTION
// y                (vector)            Vector retaining the right hand side. Should be unallocated upon
//                                      call.


void form_model_rhs(sparseMatrix<int,int,double> &A, std::vector<double> &y)
{
    // Get number of rows and columns
    int n = A.c;
    int m = A.r;

    // Get symmetry variable
    int issym = int(A.mtype == SYMMETRIC);

    // Allocate vector with all its components set to 1
    std::vector<double> x(n,1.0);
    
    // Allocate output vector
    y.resize(m);

    // Perform sparse matrix by vector multiplication
    sparse_matrix_t AA;
    struct matrix_descr descr;
    if (A.stype == CSR)
        mkl_sparse_d_create_csr (&AA, SPARSE_INDEX_BASE_ZERO, n, n, &(A.i[0]), &(A.i[1]), &(A.j[0]), &(A.v[0]));
    else if (A.stype == CSC)
        mkl_sparse_d_create_csr (&AA, SPARSE_INDEX_BASE_ZERO, n, n, &(A.j[0]), &(A.j[1]), &(A.i[0]), &(A.v[0]));
    else if (A.stype == COO)
        mkl_sparse_d_create_coo (&AA, SPARSE_INDEX_BASE_ZERO, n, n, A.nnz, &(A.i[0]), &(A.j[0]),&(A.v[0]));
    if (!issym)
        descr.type = SPARSE_MATRIX_TYPE_GENERAL;
    else
    {
        descr.type = SPARSE_MATRIX_TYPE_SYMMETRIC;
        descr.mode = SPARSE_FILL_MODE_UPPER;
        descr.diag = SPARSE_DIAG_NON_UNIT;    
    }
    mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE,1.0,AA,descr,&x[0],0.0,&y[0]);    

}

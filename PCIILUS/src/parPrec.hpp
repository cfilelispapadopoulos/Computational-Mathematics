#ifndef PARPREC_HPP
#define PARPREC_HPP
#include <vector>
#include <mkl.h>
#include "sparseMatrix.hpp"

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


// PCIILUS - Parallel Combined Improved Incomplete LU for Skyline Storage
//
// Computes the incomplete factorization and inverse matrix based preconditioner in the form A = L D U and M = G D^{i+1} H,
// of a general sparse matrix A stored in Compressed Sparse Row (CSR) storage format (ordered), decoupled computational pattern. 
// The method adaptively computes positions and values of the elements of the factors based on the selection of the lfill, ftol
// ftoli.
//
//
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
// A                (sparseMatrix)      A sparseMatrix, retaining the coefficient matrix of the linear system
//                                      in (CSR).
// Ftol             (double)            Drop tolerance parameter in [0,...,1] which controls the density of
//                                      the incomplete factorization. A good initial value is 0.0001.
// Ftoli            (double)            Drop tolerance parameter in [0,...,1] which controls the density of
//                                      the incomplete inverse matrix. A good initial value is 0.03.
// lfill            (int)               Fill-in for both the LDU and GD^{-1}H e.g. 4.
// warpsize         (int)               Limit in the number of elements per column or row of matrices G and H (>=0). This
//                                      parameter controls the number of elements. If set to -1 no limit is imposed.
// filtype          (char)              Type of filtration used. "1" is mean absolute value. "2" is euclidean norm.
//                                      "m" is inf-norm.
// patt             (char)              Parameter that controls which part of the coefficient matrix will be considered
//                                      when building the dynamic sparsity pattern. "L" is the lower part. "U" is the upper
//                                      part and "B" is both parts.
//
// OUTPUT
// NAME             TYPE                DESCRIPTION
// G                (sparseMatrix)      A sparseMatrix, initally unallocated, which upon exit retains
//                                      the values of the factor G (CSR).
// U                (sparseMatrix)      A sparseMatrix, initally unallocated, which upon exit retains
//                                      the values of the factor G (CSR).
// D                (vector)            A double vector, initially empty, which upon exit points to
//                                      to allocated space of size n retaining the elements of the
//                                      diagonal factor D^{-1}
// H                (sparseMatrix)      A sparseMatrix, initally unallocated, which upon exit retains
//                                      the values of the factor H (CSR).
// L                (sparseMatrix)      A sparseMatrix, initally unallocated, which upon exit retains
//                                      the values of the factor L (CSR).
//
//
// References
// [1] C. K. Filelis - Papadopoulos (2026). Parallel Incomplete LU Factorization, Submitted.

void pciilus(sparseMatrix<int, int, double> &A,
             sparseMatrix<int, int, double> &L,
             sparseMatrix<int, int, double> &H,
             std::vector<double> &D,
             sparseMatrix<int, int, double> &U,
             sparseMatrix<int, int, double> &G,
             int lfill,
             double ftol,
             double ftoli,
             int warpsize,
             char filttype,
             char patt);

#endif

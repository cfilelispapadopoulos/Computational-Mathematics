#ifndef SPARSELINEARALGEBRA_HPP
#define SPARSELINEARALGEBRA_HPP
#include "sparseMatrix.hpp"
#include "sparseAccumulator.hpp"
#include <vector>
#include <unordered_set>

// DISCLAIMER
// The use of the code is regulated by the following copyright agreement.
//
// PIILUS software is freely available for scientific (non-commercial) use.
// 1. The code can be used only for the purpose of internal research, excluding any commercial use of the PIILUS
//    software as such or as a part of a software product. Users who want to integrate PIILUS sofware or parts of
//    it into commercial products require a license agreement.
// 2. PIILUS is provided "as is" and for the purpose described at the previous point only. In no circumstances can
//    neither the authors nor their institutions be held liable for any deficiency, fault or other mishappening
//    with regard to the use or performance of PIILUS.
// 3. All scientific publications, for which PIILUS software has been used, shall mention its usage and refer to
//    the publication [1] in the References section below.
//
//
// References
// [1] C. K. Filelis - Papadopoulos (2026). Parallel Incomplete LU Factorization, Submitted.



// transpose - Transpose a sparse matrix stored in CSR, CSC or COO
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
// B                (sparseMatrix)      Transpose matrix stored in the same format as input.

void transpose(sparseMatrix<int,int,double> &A, sparseMatrix<int,int,double> &B);

#endif

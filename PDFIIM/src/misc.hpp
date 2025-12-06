#ifndef MISC_HPP
#define MISC_HPP
#include "sparseMatrix.hpp"
#include <vector>

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


void form_model_rhs(sparseMatrix<int,int,double> &A, std::vector<double> &y);

#endif

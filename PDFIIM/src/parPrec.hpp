#ifndef PARPREC_HPP
#define PARPREC_HPP
#include <vector>
#include <mkl.h>
#include "sparseMatrix.hpp"

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


// PDFIIM - Parallel Sparsity Patterns for Factored Incomplete Inverse Matrices
//
// Computes the incomplete inverse matrix based preconditioner in factored form M = G D^{i+1} H, of a general sparse 
// matrix A stored in Compressed Sparse Row (CSR) storage format (ordered), following novel dynamic sparsity patterns
// [1] in decoupled computational pattern. The method dynamically computes positions and values of the elements of  
// the factors based on the selection of the $\kappa \ell m$ (pre-fill, lfill, post-fill) triplet dtol parameter 
// [0,...,1].
//
// 			      | A_i b |^{-1}   | G_i -G_i D_i^{-1} H_i b | | D_i^-1     0   | |          H_i           0 |
// A_{i+1}^{-1} = |		  |      = |						 | |  		  	    | |							 |
//			      |  c  d |		   |  0           1          | |    0    s^{-1} | | - c G_i D_i^{-1} H_i   1 |
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
//                                      the preconditioner. A value close to zero leads to the computation of 
//                                      a very dense preconditioner which impacts performance. A value of dtol
//                                      close to one leads to a very sparse preconditioner (diagonal) which may
//                                      be ineffective. A good initial value is 0.1.
// prefill          (int)               Pre fill-in levels (>=1) in order to start the dynamic process with more
//                                      nonzero elements.
// lfill            (int)               Dynamic Fill-in levels (>=1), i.e. the number of steps to dynamically improve
//                                      the sparsity pattern.
// postfill         (int)               Post fill-in levels (>=1) in order to include more levels of fill in per 
//                                      lfill iteration.
// warpsize         (int)               Limit in the number of elements per column or row of matrices G and H (>=0). This
//                                      parameter controls the number of elements. If set to -1 no limit is imposed.
// filtype          (char)              Type of filtration used. "1" is mean absolute value. "2" is euclidean norm.
//                                      "m" is inf-norm.
// part             (char)              Parameter that controls which part of the coefficient matrix will be considered
//                                      when building the dynamic sparsity pattern. "L" is the lower part. "U" is the upper
//                                      part and "B" is both parts.
//
// OUTPUT
// NAME             TYPE                DESCRIPTION
// G                (sparseMatrix)      A sparseMatrix, initally unallocated, which upon exit retains 
//                                      the values of the factor G (CSR).
// D                (vector)            A double vector, initially empty, which upon exit points to
//                                      to allocated space of size n+1 retaining the elements of the 
//                                      diagonal factor D^{-1}
// H                (sparseMatrix)      A sparseMatrix, initally unallocated, which upon exit retains 
//                                      the values of the factor H (CSR).
//
//
// References
// [1] C. K. Filelis - Papadopoulos and G.A. Gravvis (2025). 
//     Parallel Adaptive Factored Incomplete Inverse Matrices. In Review.

void pdfiim(sparseMatrix<int,int,double> &A,
            sparseMatrix<int,int,double> &G,
            std::vector<double> &D,
            sparseMatrix<int,int,double> &H,
            int prefill,
            int lfill,
            int postfill,
            double ftol,
            int warpsize,
            char filttype,
            char patt);        

#endif

#ifndef PREC_HPP
#define PREC_HPP

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

// ACIFIM - Adaptive Combined Incomplete Factorization and Inverse Matrix
//
// Computes the compined incomplete inverse matrix preconditioner in factored form M = G D^{-1} H and M^{-1} = L D U, of a
// general sparse matrix A stored in Compressed Sparse Row (CSR) storage format (ordered) along with an incomplete factorization
//  LDU, following a recursive approach [1]. The method adaptively computes positions and values of the elements of the
// factors based on the dtol parameter [0,...,1].
//
// 			 | A_i b |   |       L_i        0 | | D_i     0 | | U_i   D_i^{-1} H_i b |
// A_{i+1} = |		 | = |				      | |           | |                      |
//			 |  c  d |   | c G_i D_i^{-1}   1 | |  0      s | |  0          1        |
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
// n                (integer)           Size of square general sparse matrix A.
// Av               (double*)           Vector of size nnz(A) retaining the values of A (CSR).
// Aj               (integer*)          Vector of size nnz(A) retaining the column indices of the
//                                      elements of matrix A (CSR).
// Ai               (integer*)          Vector of size n+1 retaining the offsets of the rows of A (CSR).
// elemPerRowCol    (integer)           Initial number of elements (>0) in the rows and columns of the flexible
//                                      storage format. A small value in case of a dense preconditioner
//                                      might lead to large number of reallocations during formation of
//                                      the preconditioner. A good value is 10.
// growth           (integer)           The number of elements (>0) to be added to the already allocated space
//                                      in case it is full. A small value in case of a dense preconditioner
//                                      might lead to large number of reallocations during formation of
//                                      the preconditioner. A good value is 5.
// dtol1            (double)            Drop tolerance parameter in [0,...,1] which controls the density of
//                                      the LDU preconditioner. A value close to zero leads to the computation of
//                                      a very dense preconditioner which impacts performance. A value of dtol
//                                      close to one leads to a very sparse preconditioner (diagonal) which may
// dtol2            (double)            Drop tolerance parameter in [0,...,1] which controls the density of
//                                      the GD^{-1}H preconditioner. A value close to zero leads to the computation of
//                                      a very dense preconditioner which impacts performance. A value of dtol
//                                      close to one leads to a very sparse preconditioner (diagonal) which may
//                                      be ineffective. A good initial value is 0.1.
// eta              (double)            Threshold for the diagonal elements. In case of values lower than the
//                                      threshold the values are substituted with (10^{-4}+dtol).
//                                      A good initial value is approximately 10^{-8}.
// shift			(double)			Diagonal shift such that s_i = s_i + shift | A_{i,i} |.
// coeff			(double)			Multiplier that increases the drop tolerance to meet element limit.
// element_limit    (integer)			Element limit for the rows and columns of the factors.
// filtit			(integer)	    	Number of iterations to meet the element limit.
//
// OUTPUT
// NAME             TYPE                DESCRIPTION
// Gval             (double**)          A double pointer, initally set to NULL, which upon exit points to
//                                      allocated space of size nnz(G) retaining the values of the
//                                      factor G (CSR).
// Gindj            (integer**)         An integer pointer, initally set to NULL, which upon exit points to
//                                      llocated space of size nnz(G) retaining the column indices of the
//                                      elements of the factor G (CSR).
// Gindi            (integer**)         An integer pointer, initially set to NULL, which upon exit points to
//                                      allocated space of size n+1 retaining the row offsets of the
//                                      factor G (CSR).
// IDv              (double**)          A double pointer, initially set to NULL, which upon exit points to
//                                      to allocated space of size n+1 retaining the elements of the
//                                      diagonal factor D^{-1}
// Hval             (double**)          A double pointer, initally set to NULL, which upon exit points to
//                                      allocated space of size nnz(H) retaining the values of the
//                                      factor H (CSR).
// Hindj            (integer**)         An integer pointer, initally set to NULL, which upon exit points to
//                                      allocated space of size nnz(G) retaining the column indices of the
//                                      elements of the factor H (CSR)
// Hindi            (integer**)         An integer pointer, initially set to NULL, which upon exit points to
//                                      allocated space of size n+1 retaining the row offsets of the
//                                      factor H (CSR).
// Lval             (double**)          A double pointer, initally set to NULL, which upon exit points to
//                                      allocated space of size nnz(L) retaining the values of the
//                                      factor L (CSR).
// Lindj            (integer**)         An integer pointer, initally set to NULL, which upon exit points to
//                                      allocated space of size nnz(L) retaining the column indices of the
//                                      elements of the factor L (CSR)
// Lindi            (integer**)         An integer pointer, initially set to NULL, which upon exit points to
//                                      allocated space of size n+1 retaining the row offsets of the
//                                      factor L (CSR).
// Uval             (double**)          A double pointer, initally set to NULL, which upon exit points to
//                                      allocated space of size nnz(U) retaining the values of the
//                                      factor U (CSR).
// Uindj            (integer**)         An integer pointer, initally set to NULL, which upon exit points to
//                                      allocated space of size nnz(U) retaining the column indices of the
//                                      elements of the factor U (CSR)
// Uindi            (integer**)         An integer pointer, initially set to NULL, which upon exit points to
//                                      allocated space of size n+1 retaining the row offsets of the
//                                      factor U (CSR).
// lval             (integer)           Upon successful computation the function returns 0, otherwise 1 when 
//                                      there is non or inf in the computation.
//
//
// References
// [1] C. K. Filelis - Papadopoulos (2026). Adaptive Combined Incomplete Factorizations and Inverse Matrices. To be submitted.

int acifim(int n,
           double *Av,
           int *Aj,
           int *Ai,
           double **Gval,
           int **Gindj,
           int **Gindi,
           double **IDv,
           double **Hval,
           int **Hindj,
           int **Hindi,
           double **Lval,
           int **Lindj,
           int **Lindi,
           double **Uval,
           int **Uindj,
           int **Uindi,
           double dtol1,
           double dtol2,
           int elemPerRowCol,
           int growth,
           double eta,
           double shift,
           double coeff,
           int element_limit,
           int filtit);

#endif

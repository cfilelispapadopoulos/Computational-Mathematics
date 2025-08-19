#ifndef RW_MATRIX_HPP
#define RW_MATRIX_HPP

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
// [1] C. K. Filelis - Papadopoulos (2024). Adaptive Factored Incomplete Inverse Matrices. 
//     Numer Linear Algebra Appl, 32: e2603. https://doi.org/10.1002/nla.2603



// READ_COEFF_MATRIX - Reading a real symmetric or general sparse matrix from an mtx file
//
// The function reads a Matrix Market type file and exports it in Compressed Sparse Row storage format. The
// function supports only general and symmetric real sparse matrices, however it converts the symmetric ones in
// general sparse format in order to be able to be handled by the provided functions. The function relies
// heavily on Matrix Market functions [1].
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
// filename         (char *)            An array of characters retaining the name of the file. It can be a
//                                      path also.
//
// OUTPUT
// NAME             TYPE                DESCRIPTION
// n                (integer)           Size of square general sparse matrix A.
// Av               (double**)          Pointer to a vector of size nnz(A) retaining the values of A (CSR).
// Aj               (integer**)         Pointer to a vector of size nnz(A) retaining the column indices of the 
//                                      elements of matrix A (CSR).
// Ai               (integer**)         Pointer to a vector of size n+1 retaining the offsets of the rows of
//                                      A (CSR).
//
//
// References
// [1] MNIST. Matrix Market. https://math.nist.gov/MatrixMarket/ (last accessed: 20/9/2024).

void read_coeff_matrix(const char* filename, 
                     int *n, 
                     double **Av, 
                     int **Aj, 
                     int **Ai);


// READ_RHS_VECTOR - Reading a real general dense vector from an mtx file
//
// The function reads a Matrix Market type file and exports it in linear memory storage. The function supports
// only general and real dense vectors. The function relies heavily on Matrix Market functions [1].
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
// filename         (char *)            An array of characters retaining the name of the file. It can be a
//                                      path also.
//
// OUTPUT
// NAME             TYPE                DESCRIPTION
// n                (integer*)          Size of square general sparse vector B.
// B                (double**)          Pointer to a vector of size n retaining the values of B.
//
//
// References
// [1] MNIST. Matrix Market. https://math.nist.gov/MatrixMarket/ (last accessed: 20/9/2024).

void read_rhs_vector(const char* filename, 
                     int *n, 
                     double **B);

// WRITE_VECTOR - Writing a real general dense vector to an mtx file
//
// The function writes a Matrix Market type file and exports a vector to it from memory. The function supports
// only general and real dense vectors. The function relies heavily on Matrix Market functions [1].
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
// filename         (char *)            An array of characters retaining the name of the file. It can be a
//                                      path also.
// n                (integer)           Size of square general sparse vector B.
// x                (double*)           Pointer to a vector of size n retaining the values of B.
//
//
// References
// [1] MNIST. Matrix Market. https://math.nist.gov/MatrixMarket/ (last accessed: 20/9/2024).

void write_vector(const char* filename, 
                  int n, 
                  double *x);

#endif

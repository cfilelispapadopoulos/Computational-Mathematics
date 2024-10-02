#include "perm.hpp"
#include "qsort.hpp"
#include <iostream>
#include <algorithm>

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


// MPERM - Function that permutes rows and columns of square sparse matrix based on input integer permultation
//         vector.
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
// Av               (double*)           Vector of size nnz(A) retaining the values of A (CSR). Upon termination
//                                      the permuted matrix is returned.
// Aj               (integer*)          Vector of size nnz(A) retaining the column indices of the 
//                                      elements of matrix A (CSR). Upon termination the permuted matrix is
//                                      returned.
// Ai               (integer*)          Vector of size n+1 retaining the offsets of the rows of A (CSR). Upon 
//                                      termination the permuted matrix is returned.
// Q                (integer*)          Permutation vector.


void mperm(int n, double *Av, int *Aj, int *Ai, int *Q)
{
    // Iterators
    int i,j,k;

    // Variable retaining the number of nonzero elements
    int nnz = Ai[n];
    
    // Work vector that retains the inverse permutation
    int *work     = new int[n+1];
    // Auxiliary work vectors
    int *work2    = new int[nnz];
    double *dwork = new double[nnz];

    // Permute columns first
    for(i=0;i<n;i++) work[Q[i]] = i;

    // Permute
    for(i=0;i<nnz;i++) Aj[i] = work[Aj[i]];

    // Permute rows
    for(i=0;i<n;i++)
    {
        j = Q[i];
        work[i+1] = Ai[j+1] - Ai[j];
    }
    work[0] = 0;

    // Create offsets
    for(i=0;i<n;i++) work[i+1]+=work[i];

    // Permute rows
    for(i=0;i<n;i++)
    {
        j=work[i];
        for(k=Ai[Q[i]];k<Ai[Q[i]+1];k++)
        {
            work2[j]=Aj[k];
            dwork[j]=Av[k];
            j++;
        }
    }

    // Copy result
    std::copy(work,work+n+1,Ai);
    std::copy(work2,work2+nnz,Aj);
    std::copy(dwork,dwork+nnz,Av);

    // Sort indices
    for(i=0;i<n;i++) qSort2(Aj,Av,Ai[i],Ai[i+1]-1,0);

    // Cleanup
    delete[] work;
    delete[] work2;
    delete[] dwork;

}

// VPERM - Function that permutes rows of a dense vector based on input integer permultation vector.
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
// B                (double*)           Vector of size n. Upon termination the permuted matrix is returned.
// Q                (integer*)          Permutation vector.

void vperm(int n, double *B, int *Q)
{
    // Iterators
    int i;

    // Work vector
    double *work = new double[n];

    // Permute vector
    for(i=0;i<n;i++) work[i] = B[Q[i]];

    // Copy to output
    std::copy(work,work+n,B);

    // Cleanup
    delete[] work;

}

// MD - Function that computes a permutation vector such that the rows of a sparse matrix are sorted based
//      on vertex degree in ascending order.
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
// issym            (integer)           Variable denoting if input matrix and preconditioner are symmetric.
//
// OUTPUT
// NAME             TYPE                DESCRIPTION
// Q                (double**)          Vector retaining the permutation. Should be unallocated upon
//                                      call.


void md(int n, double *Av, int *Aj, int *Ai, int issym, int **Q)
{
    // Iterator
    int i,j;

    // Temporary variable to retain column indices
    int jdx;

    // Allocate vector that will retain the degrees
    int *deg = new int[n];

    // Allocate output vector
    (*Q) = new int[n];

    // Initialize Q
    for (i=0;i<n;i++) deg[i] = 0;

    // Form permutation vector
    for(i=0;i<n;i++)
    {
        (*Q)[i] = i;
        deg[i] = Ai[i+1] - Ai[i];
    }

    // In case of symmetric matrix augment the degrees
    if (issym)
        for(i=0;i<n;i++)
            for(j=Ai[i];j<Ai[i+1];j++)
            {
                jdx = Aj[j];
                if (jdx>i)
                    deg[jdx]++;
            }

    // Sort array
    qSort2(deg,*Q,0,n-1,0);

    //Clean up
    delete[] deg;
}

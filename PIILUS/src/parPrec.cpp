#include "parPrec.hpp"
#include "minmax.hpp"
#include <iostream>
#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>
#include <mkl.h>
#include <omp.h>
#include "sparseMatrix.hpp"
#include "sparseAccumulator.hpp"
#include "sparseLinearAlgebra.hpp"
#include <mkl_lapacke.h>
#include "qsort.hpp"

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

// PIILUS - Parallel Improved Incomplete LU with Skyline Storage
//
// Computes the incomplete factorization based preconditioner in the form M = L D U, of a general sparse
// matrix A stored in Compressed Sparse Row (CSR) storage format (ordered), following a decoupled computational
// pattern [1]. The method adaptively computes positions and values of the elements of the factors based on the
// selection of the lfill and ftol.
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
// lfill            (int)               Fill-in levels (>=1).
// ftol             (double)            Drop tolerance parameter in [0,...,1] which controls the density of
//                                      the preconditioner. A value close to zero leads to the computation of
//                                      a very dense preconditioner which impacts performance. A value of dtol
//                                      close to one leads to a very sparse preconditioner (diagonal) which may
//                                      be ineffective. A good initial value is 0.01.
// warpsize         (int)               Limit in the number of elements per column or row of matrices L and U (>=0). This
//                                      parameter controls the number of elements. If set to -1 no limit is imposed.
// filtype          (char)              Type of filtration used. "1" is mean absolute value. "2" is euclidean norm.
//                                      "m" is inf-norm.
// patt             (char)              Parameter that controls which part of the coefficient matrix will be considered
//                                      when building the dynamic sparsity pattern. "L" is the lower part. "U" is the upper
//                                      part and "B" is both parts.
//
// OUTPUT
// NAME             TYPE                DESCRIPTION
// L                (sparseMatrix)      A sparseMatrix, initally unallocated, which upon exit retains
//                                      the values of the factor L (CSR).
// D                (vector)            A double vector, initially empty, which upon exit points to
//                                      to allocated space of size n retaining the elements of the
//                                      diagonal factor D^{-1}
// U                (sparseMatrix)      A sparseMatrix, initally unallocated, which upon exit retains
//                                      the values of the factor U (CSR).
//
//
// References
// [1] C. K. Filelis - Papadopoulos (2026). Parallel Incomplete LU Factorization, Submitted.

void piilus(sparseMatrix<int, int, double> &A,
            sparseMatrix<int, int, double> &L,
            std::vector<double> &D,
            sparseMatrix<int, int, double> &U,
            int lfill,
            double ftol,
            int warpsize,
            char filttype = '1',
            char patt = 'B')
{

    // Temporary variables
    int n = (uint)A.r;

    // Square root of \epsilon_{mach}
    double seps = std::sqrt(std::numeric_limits<double>::epsilon());

    // Upper and lower parts of A
    sparseMatrix<int, int, double> At;
    D.resize(n);

    // Form transpose matrix
    transpose(A, At);

    // Iterate and build approximate inverse
    // G factor
    std::vector<std::vector<double>> Ut(n, std::vector<double>()), Lt(n, std::vector<double>());
    std::vector<std::vector<int>> Utj(n, std::vector<int>()), Ltj(n, std::vector<int>());

    // Store Diagonals
    Ut[0].resize(1);
    Ut[0][0] = 1.0;
    Utj[0].resize(1);
    Utj[0][0] = 0;
    Lt[0].resize(1);
    Lt[0][0] = 1.0;
    Ltj[0].resize(1);
    Ltj[0][0] = 0;

    D[0] = 1. / A.v[0];

#pragma omp parallel
    {
        std::vector<int> inds;
        std::vector<double> B, rhs, rhs2;
        sparseAccumulatorSymbolic<int, int> col(n);
        double DD = 0.0;

#pragma omp for schedule(runtime) nowait
        for (int i = 1; i < n; i++)
        {
            int kmax = 0, kl = 0, ku = 0;
            int c = 0, c2 = 0, len = 0, tlen = 0;
            double norm1g = 0.0, norm1h = 0.0;

            if (patt == 'U' || patt == 'B')
            {
                for (int j = At.i[i]; j < At.i[i + 1]; j++)
                    if (At.j[j] < i)
                        col.push(At.j[j]);
                    else
                        break;
            }
            if (patt == 'L' || patt == 'B')
            {
                for (int j = A.i[i]; j < A.i[i + 1]; j++)
                    if (A.j[j] < i)
                        col.push(A.j[j]);
                    else
                        break;
            }

            // Accumulate Pre-fill level sets
            if (patt == 'B')
                for (int k = 2; k <= lfill; k++)
                {
                    col.rewind();
                    len = col.nnz;
                    for (int j = 0; j < len; j++)
                    {
                        int jind;
                        col.next(jind);
                        for (int l = At.i[jind]; l < At.i[jind + 1]; l++)
                        {
                            if (At.j[l] < i)
                                col.push(At.j[l]);
                            else
                                break;
                        }
                        for (int l = A.i[jind]; l < A.i[jind + 1]; l++)
                        {
                            if (A.j[l] < i)
                                col.push(A.j[l]);
                            else
                                break;
                        }
                    }
                }
            else if (patt == 'U')
                for (int k = 2; k <= lfill; k++)
                {
                    col.rewind();
                    len = col.nnz;
                    for (int j = 0; j < len; j++)
                    {
                        int jind;
                        col.next(jind);
                        for (int l = At.i[jind]; l < At.i[jind + 1]; l++)
                        {
                            if (At.j[l] < i)
                                col.push(At.j[l]);
                            else
                                break;
                        }
                    }
                }
            else
                for (int k = 2; k <= lfill; k++)
                {
                    col.rewind();
                    len = col.nnz;
                    for (int j = 0; j < len; j++)
                    {
                        int jind;
                        col.next(jind);
                        for (int l = A.i[jind]; l < A.i[jind + 1]; l++)
                        {
                            if (A.j[l] < i)
                                col.push(A.j[l]);
                            else
                                break;
                        }
                    }
                }

            // Compute new elements
            // Initilize Counter
            c = 0;
            c2 = 0;
            norm1g = 0.0;

            // Copy to dense vector
            len = col.nnz;

            // Remove entries
            for (int j = len; j >= warpsize && warpsize != -1; j--)
                col.delete_last();
            len = col.nnz;

            // Compress nnz entries to a vector
            col.rewind();
            inds.resize(len);
            for (int j = 0; j < len; j++)
            {
                int jind, kind;
                col.next(jind, kind);
                inds[kind] = jind;
            }

            // Reset matrices
            B.resize(len * len);
            std::fill(B.begin(), B.begin() + len * len, 0.0);
            rhs.resize(len);
            std::fill(rhs.begin(), rhs.begin() + len, 0.0);
            rhs2.resize(len);
            std::fill(rhs2.begin(), rhs2.begin() + len, 0.0);

            // Sort vector
            if (len != 0)
            {
                qSort(&inds[0], 0, len - 1, 0);
                for (int j = 0; j < len; j++)
                    col.o[inds[j]] = j;
            }

            // Copy matrix elements
            for (int j = 0; j < len; j++)
            {
                int kind = inds[j];
                for (int kk = A.i[kind]; kk < A.i[kind + 1] && A.j[kk] < i; kk++)
                {
                    int jind2 = A.j[kk];
                    if (!col.isempty(jind2))
                    {
                        int kind2 = col.o[jind2];
                        B[j * len + kind2] = A.v[kk];
                    }
                }
                // Account for zero diagonals
                if (B[j * len + j] == 0.0)
                    B[j * len + j] = std::max(seps, ftol);
            }

            // Compute LU factorization of local coefficient matrix
            if (len != 0)
            {
                for (int j = 0; j < len; j++)
                {
                    if (inds[j] < A.j[A.i[i]])
                        ku++;
                    if (inds[j] < At.j[At.i[i]])
                        kl++;
                }
                kmax = mmax(kl, ku);

                (void)LAPACKE_mkl_dgetrfnp(LAPACK_ROW_MAJOR, len, len, &B[0], len);
            }

            // Copy rhs elements (u_{i+1})
            for (int j = At.i[i]; j < At.i[i + 1]; j++)
            {
                if (!col.isempty(At.j[j]) && At.j[j] < i)
                {
                    rhs[col.o[At.j[j]]] = At.v[j];
                }
            }

            // Copy rhs2 elements (l_{i+1})
            for (int j = A.i[i]; j < A.i[i + 1]; j++)
            {
                if (!col.isempty(A.j[j]) && A.j[j] < i)
                {
                    rhs2[col.o[A.j[j]]] = A.v[j];
                }
                if (A.j[j] == i)
                    DD = A.v[j];
            }

            double s = 0.0;
            if (len != 0)
            {
                cblas_dtrsv(CblasRowMajor, CblasLower, CblasNoTrans, CblasUnit, len - kl, &B[kl * len + kl], len, &rhs[kl], 1);
                cblas_dtrsv(CblasRowMajor, CblasUpper, CblasTrans, CblasNonUnit, len - ku, &B[ku * len + ku], len, &rhs2[ku], 1);

                // Compute corresponding diagonal element
                for (int j = kmax; j < len; j++)
                    s += (rhs[j] * rhs2[j]);

                // Apply diagonal scaling
                for (int j = kl; j < len; j++)
                    rhs[j] /= B[j * len + j];
            }
            col.empty();

            // Compute the norm and filter elements
            if (filttype == '1')
            {
                norm1g = 1.0;
                for (int j = kl; j < len; j++)
                    norm1g += std::fabs(rhs[j]);
                norm1g /= (len + 1);

                norm1h = 1.0;
                for (int j = ku; j < len; j++)
                    norm1h += std::fabs(rhs2[j]);
                norm1h /= (len + 1);
            }
            else if (filttype == '2')
            {
                norm1g = 1.0;
                for (int j = kl; j < len; j++)
                    norm1g += rhs[j] * rhs[j];
                norm1g = sqrt(norm1g);

                norm1h = 1.0;
                for (int j = ku; j < len; j++)
                    norm1h += rhs2[j] * rhs2[j];
                norm1h = sqrt(norm1h);
            }
            else
            {
                norm1g = 1.0;
                for (int j = kl; j < len; j++)
                    norm1g = (norm1g < std::fabs(rhs[j])) ? (std::fabs(rhs[j])) : norm1g;

                norm1h = 1.0;
                for (int j = ku; j < len; j++)
                    norm1h = (norm1h < std::fabs(rhs2[j])) ? (std::fabs(rhs2[j])) : norm1h;
            }

            c = 0;
            c2 = 0;
            tlen = len;
            for (int j = kl; j < len; j++)
            {
                if (std::fabs(rhs[j]) >= ftol * norm1g)
                    c++;
            }
            for (int j = ku; j < len; j++)
            {
                if (std::fabs(rhs2[j]) >= ftol * norm1h)
                    c2++;
            }

            // Allocate space for elements of G
            Ut[i].resize(c + 1);
            Utj[i].resize(c + 1);

            // Filter and store
            len = 0;
            for (int j = kl; j < tlen; j++)
            {
                if (std::fabs(rhs[j]) >= ftol * norm1g)
                {
                    Ut[i][len] = rhs[j];
                    Utj[i][len] = inds[j];
                    len++;
                }
                rhs[j] = 0.0;
            }

            Ut[i][len] = 1.0;
            Utj[i][len] = i;

            // Allocate space for elements of H
            Lt[i].resize(c2 + 1);
            Ltj[i].resize(c2 + 1);

            // Filter and store
            len = 0;
            for (int j = ku; j < tlen; j++)
            {
                if (std::fabs(rhs2[j]) >= ftol * norm1h)
                {
                    Lt[i][len] = rhs2[j];
                    Ltj[i][len] = inds[j];
                    len++;
                }
                rhs2[j] = 0.0;
            }

            Lt[i][len] = 1.0;
            Ltj[i][len] = i;

            // Store diagonal element
            s = DD - s;

            // Check for zero or near zero diagonal
            if (std::fabs(s) <= seps * std::fabs(DD))
                s = DD;
            D[i] = 1. / (s);
        }
    }

    // Count nonzero elements and form G
    U.r = n;
    U.c = n;
    U.stype = CSR;
    U.mtype = UPPERTRI;

    U.i.resize(n + 1, 0);
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < (int)Ut[i].size(); j++)
            U.i[Utj[i][j] + 1]++;
    }

    for (int i = 0; i < n; i++)
        U.i[i + 1] += U.i[i];

    U.nnz = U.i[n];
    U.j.resize(U.nnz);
    U.v.resize(U.nnz);

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < (int)Ut[i].size(); j++)
        {
            U.j[U.i[Utj[i][j]]] = i;
            U.v[U.i[Utj[i][j]]] = Ut[i][j];
            U.i[Utj[i][j]]++;
        }
    }

    // Deallocate
    Ut = std::vector<std::vector<double>>();
    Utj = std::vector<std::vector<int>>();

    for (int i = n; i >= 1; i--)
        U.i[i] = U.i[i - 1];
    U.i[0] = 0;

    // Count nonzero elements and form H
    int c = 0;
    for (int i = 0; i < n; i++)
        c += (int)Lt[i].size();

    L.resize(CSR, LOWERTRI, n, n, c);
    L.i[0] = 0;

    c = 0;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < (int)Lt[i].size(); j++)
        {
            L.j[c] = Ltj[i][j];
            L.v[c] = Lt[i][j];
            c++;
        }
        L.i[i + 1] = c;
    }
    // Deallocate
    Lt = std::vector<std::vector<double>>();
    Ltj = std::vector<std::vector<int>>();
}

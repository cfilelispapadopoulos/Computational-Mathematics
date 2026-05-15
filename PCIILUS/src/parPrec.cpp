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
             char filttype = '1',
             char patt = 'B')
{

    // Square root of \epsilon_{mach}
    double seps = std::sqrt(std::numeric_limits<double>::epsilon());

    // Temporary variables
    int n = (uint)A.r;

    // Upper and lower parts of A
    sparseMatrix<int, int, double> At;
    D.resize(n);

    // Form transpose matrix
    transpose(A, At);

    // Iterate and build approximate inverse
    // G factor
    std::vector<std::vector<double>> Ut(n, std::vector<double>()), Lt(n, std::vector<double>());
    std::vector<std::vector<int>> Utj(n, std::vector<int>()), Ltj(n, std::vector<int>());
    std::vector<std::vector<double>> Gt, Ht;
    std::vector<std::vector<int>> Gtj, Htj;

    Gt = std::vector<std::vector<double>>(n, std::vector<double>());
    Ht = std::vector<std::vector<double>>(n, std::vector<double>());
    Gtj = std::vector<std::vector<int>>(n, std::vector<int>());
    Htj = std::vector<std::vector<int>>(n, std::vector<int>());

    // Store Diagonals
    Ut[0].resize(1);
    Ut[0][0] = 1.0;
    Utj[0].resize(1);
    Utj[0][0] = 0;
    Lt[0].resize(1);
    Lt[0][0] = 1.0;
    Ltj[0].resize(1);
    Ltj[0][0] = 0;

    Gt[0].resize(1);
    Gt[0][0] = 1.0;
    Gtj[0].resize(1);
    Gtj[0][0] = 0;
    Ht[0].resize(1);
    Ht[0][0] = 1.0;
    Htj[0].resize(1);
    Htj[0][0] = 0;

    D[0] = 1. / A.v[0];

#pragma omp parallel
    {
        // Set number of threads for MKL
        mkl_set_num_threads(1);
        std::vector<int> inds;
        std::vector<double> B, rhs, rhs2;
        sparseAccumulatorSymbolic<int, int> col(n);
        double s = 0.0, DD = 0.0;

#pragma omp for schedule(runtime) nowait
        for (int i = 1; i < n; i++)
        {
            int kl = 0, ku = 0;
            int c = 0, c2 = 0, len = 0, tlen = 0;
            double norm1u = 0.0, norm1l = 0.0;

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

            // Copy to dense vector
            len = col.nnz;

            for (int j = len; j >= warpsize && warpsize != -1; j--)
                col.delete_last();
            len = col.nnz;

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

            // Sort indices
            if (len != 0)
            {
                qSort(&inds[0], 0, len - 1, 0);
                for (int j = 0; j < len; j++)
                    col.o[inds[j]] = j;
            }

            // Copy matrix elements
            for (int j = 0; j < len; j++)
            {
                // Account for zero diagonals
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
                if (B[j * len + j] == 0.0)
                    B[j * len + j] = std::max(std::numeric_limits<double>::epsilon(), ftol);
            }

            // Factor local coefficient matrix
            if (len != 0)
            {
                for (int j = 0; j < len; j++)
                {
                    if (inds[j] < A.j[A.i[i]])
                        ku++;
                    if (inds[j] < At.j[At.i[i]])
                        kl++;
                }
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
                    s = A.v[j];
            }

            // Compute new column and row for L and U factors
            if (len != 0)
            {
                cblas_dtrsv(CblasRowMajor, CblasLower, CblasNoTrans, CblasUnit, len - kl, &B[kl * len + kl], len, &rhs[kl], 1);
                cblas_dtrsv(CblasRowMajor, CblasUpper, CblasTrans, CblasNonUnit, len - ku, &B[ku * len + ku], len, &rhs2[ku], 1);
            }
            col.empty();

            // Compute the norm and filter elements
            if (filttype == '1')
            {
                norm1u = 1.0;
                for (int j = kl; j < len; j++)
                    norm1u += std::fabs(rhs[j] / B[j * len + j]);
                norm1u /= (len + 1);

                norm1l = 1.0;
                for (int j = ku; j < len; j++)
                    norm1l += std::fabs(rhs2[j]);
                norm1l /= (len + 1);
            }
            else if (filttype == '2')
            {
                norm1u = 1.0;
                for (int j = kl; j < len; j++)
                    norm1u += (rhs[j] / B[j * len + j]) * (rhs[j] / B[j * len + j]);
                norm1u = sqrt(norm1u);

                norm1l = 1.0;
                for (int j = ku; j < len; j++)
                    norm1l += rhs2[j] * rhs2[j];
                norm1l = sqrt(norm1l);
            }
            else
            {
                norm1u = 1.0;
                for (int j = kl; j < len; j++)
                    norm1u = (norm1u < std::fabs(rhs[j] / B[j * len + j])) ? (std::fabs(rhs[j] / B[j * len + j])) : norm1u;

                norm1l = 1.0;
                for (int j = ku; j < len; j++)
                    norm1l = (norm1l < std::fabs(rhs2[j])) ? (std::fabs(rhs2[j])) : norm1l;
            }

            c = 0;
            c2 = 0;
            tlen = len;
            for (int j = kl; j < len; j++)
            {
                if (std::fabs(rhs[j] / B[j * tlen + j]) >= ftol * norm1u)
                    c++;
                else
                    rhs[j] = 0.0;
            }
            for (int j = ku; j < len; j++)
            {
                if (std::fabs(rhs2[j]) >= ftol * norm1l)
                    c2++;
                else
                    rhs2[j] = 0.0;
            }

            // Allocate space for elements of U
            Ut[i].resize(c + 1);
            Utj[i].resize(c + 1);

            // Filter and store
            len = 0;
            for (int j = kl; j < tlen; j++)
            {
                if (std::fabs(rhs[j] / B[j * tlen + j]) >= ftol * norm1u)
                {
                    Ut[i][len] = rhs[j] / B[j * tlen + j];
                    Utj[i][len] = inds[j];
                    len++;
                }
            }

            // Set diagonal element
            Ut[i][len] = 1.0;
            Utj[i][len] = i;

            // Allocate space for elements of H
            Lt[i].resize(c2 + 1);
            Ltj[i].resize(c2 + 1);

            // Filter and store
            len = 0;
            for (int j = ku; j < tlen; j++)
            {
                if (std::fabs(rhs2[j]) >= ftol * norm1l)
                {
                    Lt[i][len] = rhs2[j];
                    Ltj[i][len] = inds[j];
                    len++;
                }
            }

            // Set diagonal element
            Lt[i][len] = 1.0;
            Ltj[i][len] = i;

            // Compute the elements of the incomplete inverse factors
            if (tlen != 0)
            {
                cblas_dtrsv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit, tlen, &B[0], tlen, &rhs[0], 1);
                cblas_dtrsv(CblasRowMajor, CblasLower, CblasTrans, CblasUnit, tlen, &B[0], tlen, &rhs2[0], 1);
            }

            // Compute the norm and filter elements
            if (filttype == '1')
            {
                norm1u = 1.0;
                for (int j = 0; j < tlen; j++)
                    norm1u += std::fabs(rhs[j]);
                norm1u /= (tlen + 1);

                norm1l = 1.0;
                for (int j = 0; j < tlen; j++)
                    norm1l += std::fabs(rhs2[j]);
                norm1l /= (tlen + 1);
            }
            else if (filttype == '2')
            {
                norm1u = 1.0;
                for (int j = 0; j < tlen; j++)
                    norm1u += (rhs[j]) * (rhs[j]);
                norm1u = sqrt(norm1u);

                norm1l = 1.0;
                for (int j = 0; j < tlen; j++)
                    norm1l += rhs2[j] * rhs2[j];
                norm1l = sqrt(norm1l);
            }
            else
            {
                norm1u = 1.0;
                for (int j = 0; j < tlen; j++)
                    norm1u = (norm1u < std::fabs(rhs[j])) ? (std::fabs(rhs[j])) : norm1u;

                norm1l = 1.0;
                for (int j = 0; j < tlen; j++)
                    norm1l = (norm1l < std::fabs(rhs2[j])) ? (std::fabs(rhs2[j])) : norm1l;
            }

            // Allocate space
            Gt[i].resize(tlen + 1);
            Gtj[i].resize(tlen + 1);

            // Copy elements
            len = 0;
            for (int j = 0; j < tlen; j++)
            {
                if (std::fabs(rhs[j]) >= ftoli * norm1u)
                {
                    Gt[i][len] = -rhs[j];
                    Gtj[i][len] = inds[j];
                    len++;
                }
                else
                    rhs[j] = 0.0;
            }

            // Set the diagonal element
            Gt[i][len] = 1.0;
            Gtj[i][len] = i;

            Gt[i].resize(len + 1);
            Gtj[i].resize(len + 1);

            Ht[i].resize(tlen + 1);
            Htj[i].resize(tlen + 1);

            len = 0;
            for (int j = 0; j < tlen; j++)
            {
                if (std::fabs(rhs2[j]) >= ftoli * norm1l)
                {
                    Ht[i][len] = -rhs2[j];
                    Htj[i][len] = inds[j];
                    len++;
                }
                else
                    rhs2[j] = 0.0;
            }

            // Set the diagonal element
            Ht[i][len] = 1.0;
            Htj[i][len] = i;

            Ht[i].resize(len + 1);
            Htj[i].resize(len + 1);

            // Compute the diagonal element
            DD = s;
            double s2 = 0.0;
            int kk = 0, cc = At.i[i];
            while (kk < (int)Ht[i].size() - 1 && cc < At.i[i + 1])
            {
                if (Htj[i][kk] == At.j[cc])
                {
                    s2 += Ht[i][kk] * At.v[cc];
                    kk++;
                    cc++;
                }
                else if (Htj[i][kk] < At.j[cc])
                    kk++;
                else
                    cc++;
            }
            kk = 0;
            cc = A.i[i];
            while (kk < (int)Gt[i].size() - 1 && cc < A.i[i + 1])
            {
                if (Gtj[i][kk] == A.j[cc])
                {
                    s2 += Gt[i][kk] * A.v[cc];
                    kk++;
                    cc++;
                }
                else if (Gtj[i][kk] < A.j[cc])
                    kk++;
                else
                    cc++;
            }

            double s4 = 0.0;
            if (tlen != 0)
            {
                cblas_dtrmv(CblasRowMajor, CblasLower, CblasTrans, CblasUnit, tlen, B.data(), tlen, rhs2.data(), 1);
                cblas_dtrmv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit, tlen, B.data(), tlen, rhs.data(), 1);
                s4 = cblas_ddot(tlen, rhs2.data(), 1, rhs.data(), 1);
            }

            // Accumulate all parts of the Schur Complement
            s += s2 + s4;

            // Store diagonal element
            if (fabs(s) <= seps * fabs(DD))
                s = DD;

            // Invert and store diagonal
            D[i] = 1. / (s);
        }
        mkl_set_num_threads(omp_get_max_threads());

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
    Ut = std::vector<std::vector<double>>();
    Utj = std::vector<std::vector<int>>();

    for (int i = n; i >= 1; i--)
        U.i[i] = U.i[i - 1];
    U.i[0] = 0;

    G.r = n;
    G.c = n;
    G.stype = CSR;
    G.mtype = UPPERTRI;

    G.i.resize(n + 1, 0);
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < (int)Gt[i].size(); j++)
            G.i[Gtj[i][j] + 1]++;
    }

    for (int i = 0; i < n; i++)
        G.i[i + 1] += G.i[i];

    G.nnz = G.i[n];
    G.j.resize(G.nnz);
    G.v.resize(G.nnz);

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < (int)Gt[i].size(); j++)
        {
            G.j[G.i[Gtj[i][j]]] = i;
            G.v[G.i[Gtj[i][j]]] = Gt[i][j];
            G.i[Gtj[i][j]]++;
        }
    }
    Gt = std::vector<std::vector<double>>();
    Gtj = std::vector<std::vector<int>>();

    for (int i = n; i >= 1; i--)
        G.i[i] = G.i[i - 1];
    G.i[0] = 0;

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
    Lt = std::vector<std::vector<double>>();
    Ltj = std::vector<std::vector<int>>();

    c = 0;
    for (int i = 0; i < n; i++)
        c += (int)Ht[i].size();

    H.resize(CSR, LOWERTRI, n, n, c);
    H.i[0] = 0;

    c = 0;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < (int)Ht[i].size(); j++)
        {
            H.j[c] = Htj[i][j];
            H.v[c] = Ht[i][j];
            c++;
        }
        H.i[i + 1] = c;
    }
    Ht = std::vector<std::vector<double>>();
    Htj = std::vector<std::vector<int>>();

}

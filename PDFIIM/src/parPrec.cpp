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
// [1] in decoupled computational pattern. The method adaptively computes positions and values of the elements of  
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
// [1] C. K. Filelis - Papadopoulos (2025). Parallel Sparsity Patterns for Factored Incomplete Inverse Matrices. In Review.

void pdfiim(sparseMatrix<int,int,double> &A,
            sparseMatrix<int,int,double> &G,
            std::vector<double> &D,
            sparseMatrix<int,int,double> &H,
            int prefill,
            int lfill,
            int postfill,
            double ftol,
            int warpsize,
            char filttype = '1',
            char patt = 'B')
{

    // Temporary variables
    int n = (uint) A.r;
    int *Ai = &(A.i[0]), *Aj = &(A.j[0]);
    double *Av = &(A.v[0]);

    // Upper and lower parts of A
    sparseMatrix<int,int,double> At;
    D.resize(n);

    // Form transpose matrix
    transpose(A,At);
    
    // Iterate and build approximate inverse
    // G factor

    std::vector<std::vector<double>> Gt(n,std::vector<double>()),Ht(n,std::vector<double>());
    std::vector<std::vector<int>> Gtj(n,std::vector<int>()),Htj(n,std::vector<int>());
    
    // Store Diagonals
    Gt[0].resize(1); Gt[0][0] = 1.0;
    Gtj[0].resize(1); Gtj[0][0] = 0;
    Ht[0].resize(1); Ht[0][0] = 1.0;
    Htj[0].resize(1); Htj[0][0] = 0;    
    
    D[0] = 1. / A.v[0];
    
    #pragma omp parallel
    {
        std::vector<int> ipiv;
        std::vector<int> inds;
        std::vector<double> B,rhs,rhs2;        
        sparseAccumulatorSymbolic<int,int> col(n);
        #pragma omp for schedule(runtime) nowait
        for(int i = 1; i < n; i++)
        {
            int c = 0, c2 = 0,len = 0,tlen = 0;
            double norm1g = 0.0, norm1h = 0.0;

            if (patt == 'U' || patt == 'B')
            {
                for(int j = At.i[i]; j < At.i[i+1]; j++)  if (At.j[j] < i) col.push(At.j[j]); else break;
            }
            if (patt == 'L' || patt == 'B')
            {
                for(int j = A.i[i]; j < A.i[i+1]; j++)  if (A.j[j]< i) col.push(A.j[j]); else break;
            }
            
            // Accumulate Pre-fill level sets
            if (patt == 'B')
                for(int k = 2; k <= prefill; k++)
                {
                    col.rewind();
                    len = col.nnz;
                    for(int j = 0; j < len; j++)
                    {
                        int jind;
                        col.next(jind);
                        for(int l = At.i[jind]; l < At.i[jind+1]; l++)
                        {
                            if (At.j[l] < i)
                                col.push(At.j[l]);
                            else
                                break;
                        }
                        for(int l = A.i[jind]; l < A.i[jind+1]; l++)
                        {
                            if (A.j[l] < i)
                                col.push(A.j[l]);
                            else
                                break;
                        }                    
                    }                         
                }
            else if (patt == 'U')
                for(int k = 2; k <= prefill; k++)
                {
                    col.rewind();
                    len = col.nnz;
                    for(int j = 0; j < len; j++)
                    {
                        int jind;
                        col.next(jind);
                        for(int l = At.i[jind]; l < At.i[jind+1]; l++)
                        {
                            if (At.j[l] < i)
                                col.push(At.j[l]);
                            else
                                break;
                        }
                    }                         
                }
            else
                for(int k = 2; k <= prefill; k++)
                {
                    col.rewind();
                    len = col.nnz;
                    for(int j = 0; j < len; j++)
                    {
                        int jind;
                        col.next(jind);
                        for(int l = A.i[jind]; l < A.i[jind+1]; l++)
                        {
                            if (A.j[l] < i)
                                col.push(A.j[l]);
                            else
                                break;
                        }                    
                    }                         
                } 

            // Compute new elements
            for(int k = 1; k <= lfill; k++)
            {   
                // Initilize Counter
                c = 0;
                c2 = 0;
                norm1g = 0.0;

                // Copy to dense vector
                len = col.nnz;
                tlen = len;  
                if (len == 0) break;

                for(int j = len; j>=warpsize && warpsize != -1; j--) col.delete_last();
                len = col.nnz;
                tlen = len;

                col.rewind();
                inds.resize(len);
                for(int j = 0; j < len; j++) 
                {
                    int jind,kind;
                    col.next(jind,kind);
                    inds[kind] = jind; 
                }
                
                // Reset matrices
                B.resize(len * len);
                std::fill(B.begin(),B.begin() + len*len,0.0);
                rhs.resize(len);
                std::fill(rhs.begin(),rhs.begin() + len,0.0);
                rhs2.resize(len);
                std::fill(rhs2.begin(),rhs2.begin() + len,0.0);    

                // Copy matrix elements
                col.rewind();
                for(int j = 0; j < len; j++ )
                {
                    // Account for zero diagonals
                    int kind = inds[j];
                    B[j * len + j] = std::max(1e-4,ftol);       
                    for(int kk = A.i[kind]; kk < A.i[kind+1] && A.j[kk] < i; kk++)
                    {
                        int jind2 = A.j[kk];
                        if (!col.isempty(jind2))
                        {
                            int kind2 = col.o[jind2];
                            B[j * len + kind2] = A.v[kk];
                        }
                    }
                    if (B[j * len + j] == 0.0)
                            B[j * len + j] = std::max(1e-4,ftol);
                }

                // Copy rhs elements (g_{i+1})
                for(int j = At.i[i]; j < At.i[i+1]; j++)
                {
                    if (!col.isempty(At.j[j]) && At.j[j] < i)
                        rhs[col.o[At.j[j]]] = -At.v[j];
                }

                // Copy rhs2 elements (h_{i+1})
                for(int j = A.i[i]; j < A.i[i+1]; j++)
                {
                    if (!col.isempty(A.j[j]) && A.j[j] < i)
                        rhs2[col.o[A.j[j]]] = -A.v[j];
                }                

                // Solve linear system with pivoting
                ipiv.resize(len);
                (void)LAPACKE_dgetrf(LAPACK_ROW_MAJOR,len,len,&B[0],len,&ipiv[0]);
                (void)LAPACKE_dgetrs(LAPACK_ROW_MAJOR,'N',len,1,&B[0],len,&ipiv[0],&rhs[0],1);
                (void)LAPACKE_dgetrs(LAPACK_ROW_MAJOR,'T',len,1,&B[0],len,&ipiv[0],&rhs2[0],1);
                col.empty();

                // Compute the norm and filter elements
                if (filttype == '1')
                {
                    norm1g = 1.0;
                    for(int j = 0; j < len; j++) norm1g += std::fabs(rhs[j]);
                    norm1g /= (len + 1);

                    norm1h = 1.0;
                    for(int j = 0; j < len; j++) norm1h += std::fabs(rhs2[j]);
                    norm1h /= (len + 1);

                }
                else if (filttype == '2')
                {
                    norm1g = 1.0;
                    for(int j = 0; j < len; j++) norm1g += rhs[j] * rhs[j];
                    norm1g = sqrt(norm1g);

                    norm1h = 1.0;
                    for(int j = 0; j < len; j++) norm1h += rhs2[j] * rhs2[j];
                    norm1h = sqrt(norm1g);
                }            
                else
                {
                    norm1g = 1.0;
                    for(int j = 0; j < len; j++) norm1g = (norm1g < std::fabs(rhs[j]))?(std::fabs(rhs[j])):norm1g;

                    norm1h = 1.0;
                    for(int j = 0; j < len; j++) norm1h = (norm1h < std::fabs(rhs2[j]))?(std::fabs(rhs2[j])):norm1h;
                }

                c = 0; c2 = 0; tlen = len;
                for(int j = 0; j < len; j++) 
                {
                    if (std::fabs(rhs[j]) >= ftol * norm1g)
                    {   
                        col.push(inds[j]);
                        c++;
                    }
                }
                for(int j = 0; j < len; j++) 
                {
                    if (std::fabs(rhs2[j]) >= ftol * norm1h)
                    {   
                        col.push(inds[j]);
                        c2++;
                    }
                }
   
		        bool flag = (c2 == warpsize);
                if ((c==0 && c2==0) || flag)  break;
                
                // Accumulate next level set
                len = col.nnz;
                if (patt == 'B')
                {
                    for(int j = 0; j < len; j++)
                    {
                        int jind = inds[j];
                        for(int l = At.i[jind]; l < At.i[jind+1]; l++)
                        {
                            if (At.j[l] < i)
                                col.push(At.j[l]);
                            else
                                break;
                        }
                        for(int l = A.i[jind]; l < A.i[jind+1]; l++)
                        {
                            if (A.j[l] < i)
                                col.push(A.j[l]);
                            else
                                break;
                        }                     
                    }
                    
                    for(int m = 2; m <= postfill; m++)
                    {
                        col.rewind();
                        int len2 = col.nnz;
                        for(int j = 0; j < len2; j++)
                        {
                            int jind;
                            col.next(jind);
                            for(int l = At.i[jind]; l < At.i[jind+1]; l++)
                            {
                                if (At.j[l] < i)
                                    col.push(At.j[l]);
                                else
                                    break;
                            }
                            for(int l = A.i[jind]; l < A.i[jind+1]; l++)
                            {
                                if (A.j[l] < i)
                                    col.push(A.j[l]);
                                else
                                    break;
                            }                    
                        }                         
                    }
                }
                else if (patt == 'U')
                {
                    for(int j = 0; j < len; j++)
                    {
                        int jind;
                        col.next(jind);
                        for(int l = At.i[jind]; l < At.i[jind+1]; l++)
                        {
                            if (At.j[l] < i)
                                col.push(At.j[l]);
                            else
                                break;
                        }                   
                    }
                    for(int m = 2; m <= postfill; m++)
                    {
                        col.rewind();
                        int len2 = col.nnz;
                        for(int j = 0; j < len2; j++)
                        {
                            int jind;
                            col.next(jind);
                            for(int l = At.i[jind]; l < At.i[jind+1]; l++)
                            {
                                if (At.j[l] < i)
                                    col.push(At.j[l]);
                                else
                                    break;
                            }                   
                        }                         
                    }                    
                }
                else
                {
                    for(int j = 0; j < len; j++)
                    {
                        int jind;
                        col.next(jind);
                        for(int l = A.i[jind]; l < A.i[jind+1]; l++)
                        {
                            if (A.j[l] < i)
                                col.push(A.j[l]);
                            else
                                break;
                        }                     
                    }
                    for(int m = 2; m <= postfill; m++)
                    {
                        col.rewind();
                        int len2 = col.nnz;
                        for(int j = 0; j < len2; j++)
                        {
                            int jind;
                            col.next(jind);
                            for(int l = A.i[jind]; l < A.i[jind+1]; l++)
                            {
                                if (A.j[l] < i)
                                    col.push(A.j[l]);
                                else
                                    break;
                            }                    
                        }                         
                    }                    
                }
                if (col.nnz == len) break;   

            }
            
            // Allocate space for elements of G
            Gt[i].resize(c+1);
            Gtj[i].resize(c+1);

            // Filter and store
            len = 0;
            for(int j = 0; j < tlen; j++)
            {
                if (std::fabs(rhs[j]) >= ftol * norm1g)
                {
                    Gt[i][len] = rhs[j];
                    Gtj[i][len] = inds[j];
                    len++;
                }
            }
            if(len > 0) qSort2(&Gtj[i][0],&Gt[i][0],0,len-1,0);

            Gt[i][len] = 1.0;
            Gtj[i][len] = i;

            // Allocate space for elements of H
            Ht[i].resize(c2+1);
            Htj[i].resize(c2+1);

            // Filter and store
            len = 0;
            for(int j = 0; j < tlen; j++)
            {
                if (std::fabs(rhs2[j]) >= ftol * norm1h)
                {
                    Ht[i][len] = rhs2[j];
                    Htj[i][len] = inds[j];
                    len++;
                }
            }
            if(len > 0) qSort2(&Htj[i][0],&Ht[i][0],0,len-1,0);  
            
            Ht[i][len] = 1.0;
            Htj[i][len] = i;
            col.empty();
        
            // Compute corresponding diagonal element
            double s = 0;
            for(int j = 0; j < (int) Ht[i].size(); j++)
            {
                
                int Hj = Htj[i][j];
                double Hv = Ht[i][j];
                int kk  = Ai[Hj];
                int cc = 0;
                
                while (kk < Ai[Hj+1] && cc < (int) Gt[i].size())
                {   
                    if(Aj[kk] == Gtj[i][cc])
                    {
                        s += Hv * Av[kk] * Gt[i][cc];
                        kk++;
                        cc++;
                    }
                    else if (Aj[kk] < Gtj[i][cc])
                    {
                        kk++;
                    }
                    else if (Aj[kk] > Gtj[i][cc])
                    {
                        cc++;
                    }
                }
            }
 
            // Store diagonal element
            D[i] = 1. / s;
        }
    }

    // Count nonzero elements and form G
    G.r = n; G.c = n;
    G.stype = CSR; G.mtype = UPPERTRI;

    G.i.resize(n+1,0);
    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < (int) Gt[i].size(); j++)
            G.i[Gtj[i][j]+1]++;
    }

    for(int i = 0; i < n; i++) G.i[i+1] += G.i[i];

    G.nnz = G.i[n];
    G.j.resize(G.nnz);
    G.v.resize(G.nnz);

    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < (int) Gt[i].size(); j++)
        {
            G.j[G.i[Gtj[i][j]]] = i;
            G.v[G.i[Gtj[i][j]]] = Gt[i][j];
            G.i[Gtj[i][j]]++;
        }
    }
    Gt = std::vector<std::vector<double>>();
    Gtj = std::vector<std::vector<int>>();

    for(int i = n; i >= 1; i--) G.i[i] = G.i[i-1];
    G.i[0] = 0;

    // Count nonzero elements and form H
    int c = 0;
    for(int i = 0; i < n; i++) c += (int) Ht[i].size();

    H.resize(CSR,LOWERTRI,n,n,c);
    H.i[0] = 0;

    c = 0;
    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < (int) Ht[i].size(); j++)
        {
            H.j[c] = Htj[i][j];
            H.v[c] = Ht[i][j];
            c++;
        }
        H.i[i+1] = c;
    }
    Ht = std::vector<std::vector<double>>();
    Htj = std::vector<std::vector<int>>();  
}

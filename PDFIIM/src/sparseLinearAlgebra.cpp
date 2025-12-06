#include "sparseLinearAlgebra.hpp"
#include "sparseAccumulator.hpp"
#include "sparseMatrix.hpp"
#include "constants.hpp"
#include "qsort.hpp"
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <mkl_spblas.h>

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


void transpose(sparseMatrix<int,int,double> &A,sparseMatrix<int,int,double> &B)
{
    if (A.stype == COO)
    {
        B = A;
        B.i.swap(B.j);
        std::swap(B.r,B.c);
    }
    else
    {
        B.resize(A.stype,A.mtype,A.c,A.r,A.nnz);
     
        std::vector<int> *Aoffs,*Boffs,*Ainds,*Binds;
        std::vector<double> *Avals,*Bvals;
        Aoffs = &(A.i);
        Ainds = &(A.j);
        Boffs = &(B.i);
        Binds = &(B.j);
        if (A.stype == CSC)
        {
            std::swap(Aoffs,Ainds);
            std::swap(Boffs,Binds);
        }
        Avals = &(A.v);
        Bvals = &(B.v);

        // Form transpose offsets
        for(int i=0;i<A.r;i++)
        {
            for(int j=(*Aoffs)[i];j<(*Aoffs)[i+1];j++)
            {
                int idx = (*Ainds)[j]+1;
                (*Boffs)[idx]++;
            }
        }

        // Cumulative sum
        for(int i=0;i<B.r;i++) (*Boffs)[i+1]+=(*Boffs)[i];
        
        // Put elements in the correct place
        for(int i=0;i<A.r;i++)
        {
            for(int j=(*Aoffs)[i];j<(*Aoffs)[i+1];j++)
            {
                int idx = (*Ainds)[j];
                int jdx = (*Boffs)[idx]; 
                (*Binds)[jdx]=i;
                (*Bvals)[jdx]=(*Avals)[j];
                (*Boffs)[idx]++;
            }
        }

        // Fix offsets
        //std::copy_backward(Boffs->begin(),Boffs->end()-1,Boffs->end());
        for(int i=B.r;i>0;i--) (*Boffs)[i]=(*Boffs)[i-1];
        (*Boffs)[0] = 0;       

    }
}


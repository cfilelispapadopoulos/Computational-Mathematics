#ifndef sparseMatrix_HPP
#define sparseMatrix_HPP
#include <string>
#include <algorithm>
#include <vector>
#include <iostream>
#include "constants.hpp"

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



// sparseMatrix - Class that can be used to store sparse matrices in CSR, CSC or COO format
//
// Author: Christos K. Papadopoulos Filelis
//         Assistant Professor
//         Democritus University of Thrace
//         Department of Electrical and Computer Engineering
//         Xanthi, Greece, GR 67100
//         email: cpapad@ee.duth.gr
//
// ---------------------- Arguments -------------------------------------------------------------------------
// MEMBERS
// NAME             TYPE                DESCRIPTION
// stype            (int)               An integer retaining the sparse matrix storage format CSR, CSC or COO
// mtype            (int)               An integer retaining the sparse matrix type (GENERAL, SYMMETRIC, 
//                                      UPPERTRI, LOWERTRI)
// v                (vector)            Vector of floating point numbers (float,double,long double)
// i                (vector)            Integer vector retaining info about rows (int8,int16,int32,...)
// j                (vector)            Integer vector retaining info about columns (int8,int16,int32,...)
// r                (P)                 Number of rows
// c                (P)                 Number of columns
// nnz              (P)                 Number of nonzero elements
//
// METHODS                              
// sparseMatrix()                                   Default constructor
// sparseMatrix(string stype_,                      Second contructor that allocates the internal structure
//              P r_, 
//              P c_,
//              P nnz_)
// sparseMatrix(sparseMatrix &A)                    Copy constructor
// copy(sparseMatrix &A, bool vals)                 Copy of A (only structure or also values)
// resize(int stype_,                               Resize matrix (does not retain elements)
//        int mtype_, 
//        P r_,
//        P c_,
//        P nnz_)
// clear()                                          Deallocates underlying space                                                                                   
// ~sparseMatrix()                                  Default deconstructor
// print()                                          Prints matrix info

template <typename P, typename R, typename S>
class sparseMatrix {
    public:
        int stype;
        int mtype;
        std::vector<S> v;
        std::vector<R> i,j;
        P r,c,nnz;


        // sparseMatrix - Default constructor
        //
        // ---------------------- Arguments -------------------------------------------------------------------------
        // INPUTS
        // NAME             TYPE                DESCRIPTION

        sparseMatrix()
        {
            stype = -1;
            mtype = -1;
            r = P(0);
            c = P(0);
            nnz = P(0);
        };

        // sparseMatrix - Secondary constructor
        //
        // ---------------------- Arguments -------------------------------------------------------------------------
        // INPUTS
        // NAME             TYPE                DESCRIPTION
        // stype_           (int)               Input matrix storage type (CSR, CSC, COO)
        // mtype_           (int)               Input matrix type (GENERAL, SYMMETRIC, UPPERTRI, LOWERTRI)
        // r_               (P)                 Number of rows
        // c_               (P)                 Number of columns
        // nnz_             (P)                 Number of nonzero elements                    

        sparseMatrix(int stype_, int mtype_, P r_, P c_, P nnz_)
        {
            resize(stype_, mtype_, r_, c_, nnz_);
        };

        // sparseMatrix - Copy constructor
        //
        // ---------------------- Arguments -------------------------------------------------------------------------
        // INPUTS
        // NAME             TYPE                DESCRIPTION
        // A                (sparseMatrix)      Sparse matrix
 

        sparseMatrix(sparseMatrix &A)
        {
            stype = A.stype;
            mtype = A.mtype;
            r = A.r;
            c = A.c;
            nnz = A.nnz;
            v = A.v;
            i = A.i;
            j = A.j;
        };

        // copy - Copy a sparse matrix with or without values
        //
        // ---------------------- Arguments -------------------------------------------------------------------------
        // INPUTS
        // NAME             TYPE                DESCRIPTION
        // A                (sparseMatrix)      Sparse matrix
        // vals             (bool)              If true copies also the values if not only the structure
 
        void copy(sparseMatrix &A, bool vals = true)
        {
            stype = A.stype;
            mtype = A.mtype;
            r = A.r;
            c = A.c;
            nnz = A.nnz;
            if (vals) v = A.v;
            i = A.i;
            j = A.j;
        };        

        // resize - Function that allocates a matrix 
        //
        // ---------------------- Arguments -------------------------------------------------------------------------
        // INPUTS
        // NAME             TYPE                DESCRIPTION
        // stype_           (int)               Input matrix storage type (CSR, CSC, COO)
        // mtype_           (int)               Input matrix type (GENERAL, SYMMETRIC, UPPERTRI, LOWERTRI)
        // r_               (P)                 Number of rows
        // c_               (P)                 Number of columns
        // nnz_             (P)                 Number of nonzero elements  

        void resize(int stype_, int mtype_, P r_, P c_, P nnz_)
        {
            stype = stype_;
            mtype = mtype_;
            r = r_;
            c = c_;
            nnz = nnz_;
            v.resize(nnz);
            if (stype == CSR)
            {
                j.resize(nnz);
                i.resize(r+1);
            }
            else if (stype == CSC)
            {
                j.resize(c+1);
                i.resize(nnz);
            }
            else if (stype == COO)
            {
                j.resize(nnz);
                i.resize(nnz);
            }            
        };

        // sparseMatrix - Function that clears class
        //
        // ---------------------- Arguments -------------------------------------------------------------------------
        // INPUTS
        // NAME             TYPE                DESCRIPTION

        void clear()
        {
            mtype = -1;
            stype = -1;
            r = 0;
            c = 0;
            nnz = 0;
            i = std::vector<R>();
            j = std::vector<R>();
            v = std::vector<S>();
        };

        // sparseMatrix - Deconstructor
        //
        // ---------------------- Arguments -------------------------------------------------------------------------
        // INPUTS
        // NAME             TYPE                DESCRIPTION

        ~sparseMatrix()
        {

        };

        // Print - Print info
        //
        // ---------------------- Arguments -------------------------------------------------------------------------
        // INPUTS
        // NAME             TYPE                DESCRIPTION

        void print()
        {
            std::cout<<std::endl;
            std::cout<<"Storage type : "<<stype<<std::endl;
            std::cout<<"Matrix type  : "<<mtype<<std::endl;
            std::cout<<"Size         : "<<r<<" x "<<c<<std::endl;
            std::cout<<"Nonzeros     : "<<nnz<<std::endl;
            std::cout<<std::endl;
        };    

};


//



#endif

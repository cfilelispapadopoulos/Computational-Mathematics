#ifndef sparseAccumulator_HPP
#define sparseAccumulator_HPP
#include <string>
#include <algorithm>
#include <vector>
#include <iostream>

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



// sparseAccumulator - Class that implements a sparse accumulator used for sparse matrix operations
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
// v                (S*)                Vector retaining the nonzero values (float, double, long double,...)
// i                (R*)                Vector retaining the nonzero indices (int8, int16,...)
// last             (P)                 Index to the last position
// tlast            (P)                 Temporary Index to the last position
// n                (P)                 Size of sparse row or column
// nnz              (P)                 Number of nonzero elements
//
// METHODS                              
// sparseAccumulator()                              Default constructor
// sparseAccumulator(P n)                           Second contructor that allocates the internal structure
// sparseAccumulator(sparseAccumulator &A)          Copy constructor
// resize(P n)                                      Resize sparse accumulator
// clear()                                          Deallocates space
// rewind()                                         Rewinds temporary pointer to its original position
// push(R& i_, S& v_)                               Push element into accumulator
// delete_last()                                    Delete last element of the accumulator
// S& top(R& i_)                                    Return last element of the accumulator
// S& next(R& i_)                                   Next element in the accumulator
// ~sparseAccumulator()                             Default deconstructor

template <typename P, typename R, typename S>
class sparseAccumulator {
    public:
        std::vector<S> v;
        std::vector<R> i;
        P n, last ,tlast, nnz;

        // sparseAccumulator - Default constructor
        //
        // ---------------------- Arguments -------------------------------------------------------------------------
        // INPUTS
        // NAME             TYPE                DESCRIPTION

        sparseAccumulator()
        {
            last = P(-2);
            tlast = P(-2);
            n = P(0);
            nnz = P(0);
        };

        // sparseAccumulator - Secondary constructor
        //
        // ---------------------- Arguments -------------------------------------------------------------------------
        // INPUTS
        // NAME             TYPE                DESCRIPTION
        // n_               (P)                 Size of the sparse line or column
                 
        sparseAccumulator(P n_)
        {
            resize(n_);
        };

        // sparseAccumulator - Copy constructor
        //
        // ---------------------- Arguments -------------------------------------------------------------------------
        // INPUTS
        // NAME             TYPE                DESCRIPTION
        // A                (sparseAccumulator) Sparse accumulator
 
        sparseAccumulator(sparseAccumulator &A)
        {
            n = A.n;
            last = A.last;
            tlast = A.tlast;
            nnz = A.last;
            v = A.v;
            i = A.i;
        };

        // clear - Function that clears class
        //
        // ---------------------- Arguments -------------------------------------------------------------------------
        // INPUTS
        // NAME             TYPE                DESCRIPTION

        void clear()
        {
            last = -2;
            tlast = -2;
            n = 0;
            nnz = 0;
            i = std::vector<R>();
            v = std::vector<S>();
        }

        // push - Push an element into the accumulator
        //
        // ---------------------- Arguments -------------------------------------------------------------------------
        // INPUTS
        // NAME             TYPE                DESCRIPTION
        // i_               (R)                 Position of element to be pushed
        // v_               (S)                 Value of element to be pushed   

        inline void push(R &i_, S &v_)
        {
            if (isempty(i_))
            {
                v[i_] = v_;
                i[i_] = last;
                last = i_;
                nnz++;
            }
            else
                v[i_] += v_;
        }

        // delete_last - Pop an element out of the accumulator
        //
        // ---------------------- Arguments -------------------------------------------------------------------------
        // INPUTS
        // NAME             TYPE                DESCRIPTION 

        void delete_last()
        {
            R i_ = last;
            last = i[i_];
            v[i_] = 0.0;
            i[i_] = -1;
            nnz--;
        }

        // top - Return last element of the accumulator
        //
        // ---------------------- Arguments -------------------------------------------------------------------------
        // OUTPUTS
        // NAME             TYPE                DESCRIPTION
        // i_               (R)                 Index of last element
        // (rval)           (S)                 Value of last element

        S& top(R& i_)
        {
            i_ = last;
            return v[last];    
        }

        // isempty - Check if position is empty
        //
        // ---------------------- Arguments -------------------------------------------------------------------------
        // INPUTS
        // NAME             TYPE                DESCRIPTION
        // i_               (R)                 Check position of element
        // (rval)           (bool)              True if empty

        inline bool isempty(R& i_)
        {
            return (i[i_]==-1);    
        }        

        // empty - Empty the sparse accumulator
        //
        // ---------------------- Arguments -------------------------------------------------------------------------
        // INPUTS
        // NAME             TYPE                DESCRIPTION

        void empty()
        {
            for(int j=0;j<nnz;j++)
            {
                R i_ = last;
                last = i[i_];
                v[i_] = 0.0;
                i[i_] = -1;                
            }
            nnz = 0;
        }

        // resize - Secondary constructor
        //
        // ---------------------- Arguments -------------------------------------------------------------------------
        // INPUTS
        // NAME             TYPE                DESCRIPTION
        // n_               (P)                 Size of the sparse line or column
                 
        void resize(P n_)
        {
            n = n_;
            last = P(-2);
            tlast = P(-2);
            nnz = P(0);
            v.resize(n,S(0.0));
            i.resize(n,R(-1));
        };

        // rewind - Rewind the temporary pointer to access the list
        //
        // ---------------------- Arguments -------------------------------------------------------------------------
        // INPUTS
        // NAME             TYPE                DESCRIPTION

        void rewind()
        {
            tlast = last;
        };

        // next - Return next element on the list
        //
        // ---------------------- Arguments -------------------------------------------------------------------------
        // INPUTS
        // NAME             TYPE                DESCRIPTION
        // i_               (R)                 Index of the nonzero value
        //
        // OUTPUTS
        // NAME             TYPE                DESCRIPTION
        // (rval)           (S)                 Value of the next nonzero

        inline S& next(R& i_)
        {
            // Reverse iterator
            i_ = tlast;
            tlast = i[i_];
            return v[i_];       
        };  
         

        // sparseAccumulator - Deconstructor
        //
        // ---------------------- Arguments -------------------------------------------------------------------------
        // INPUTS
        // NAME             TYPE                DESCRIPTION

        ~sparseAccumulator()
        {

        };
};


template <typename P, typename R>
class sparseAccumulatorSymbolic {
    public:
        std::vector<R> i,o;
        P n, last ,tlast, nnz;

        // sparseAccumulator - Default constructor
        //
        // ---------------------- Arguments -------------------------------------------------------------------------
        // INPUTS
        // NAME             TYPE                DESCRIPTION

        sparseAccumulatorSymbolic()
        {
            last = P(-2);
            tlast = P(-2);
            n = P(0);
            nnz = P(0);
        };

        // sparseAccumulator - Secondary constructor
        //
        // ---------------------- Arguments -------------------------------------------------------------------------
        // INPUTS
        // NAME             TYPE                DESCRIPTION
        // n_               (P)                 Size of the sparse line or column
                 
        sparseAccumulatorSymbolic(P n_)
        {
            resize(n_);
        };

        // sparseAccumulator - Copy constructor
        //
        // ---------------------- Arguments -------------------------------------------------------------------------
        // INPUTS
        // NAME             TYPE                DESCRIPTION
        // A                (sparseAccumulator) Sparse accumulator
 
        sparseAccumulatorSymbolic(sparseAccumulatorSymbolic &A)
        {
            n = A.n;
            last = A.last;
            tlast = A.tlast;
            nnz = A.last;
            i = A.i;
            o = A.o;
        };

        // clear - Function that clears class
        //
        // ---------------------- Arguments -------------------------------------------------------------------------
        // INPUTS
        // NAME             TYPE                DESCRIPTION

        void clear()
        {
            last = -2;
            tlast = -2;
            n = 0;
            nnz = 0;
            i = std::vector<R>();
            o = std::vector<R>();
        }

        // push - Push an element into the accumulator
        //
        // ---------------------- Arguments -------------------------------------------------------------------------
        // INPUTS
        // NAME             TYPE                DESCRIPTION
        // i_               (R)                 Position of element to be pushed

        inline void push(R &i_)
        {
            if (isempty(i_))
            {
                i[i_] = last;
                o[i_] = nnz;
                last = i_;
                nnz++;
            }
        }

        // delete_last - Pop an element out of the accumulator
        //
        // ---------------------- Arguments -------------------------------------------------------------------------
        // INPUTS
        // NAME             TYPE                DESCRIPTION 

        void delete_last()
        {
            R i_ = last;
            last = i[i_];
            i[i_] = -1;
            o[i_] = -1;
            nnz--;
        }

        // top - Return last element of the accumulator
        //
        // ---------------------- Arguments -------------------------------------------------------------------------
        // OUTPUTS
        // NAME             TYPE                DESCRIPTION
        // (rval)           (R)                 Index of the element at the top

        R& top()
        {
            return last;    
        }

        R& top(int &idx)
        {
            idx = o[last];
            return last;    
        }        

        // isempty - Check if position is empty
        //
        // ---------------------- Arguments -------------------------------------------------------------------------
        // INPUTS
        // NAME             TYPE                DESCRIPTION
        // i_               (R)                 Position of element
        //
        // OUTPUTS
        // NAME             TYPE                DESCRIPTION
        // (rval)           (bool)              True if empty

        inline bool isempty(R& i_)
        {
            return (i[i_]==-1);    
        }        

        // empty - Empty the sparse accumulator
        //
        // ---------------------- Arguments -------------------------------------------------------------------------
        // INPUTS
        // NAME             TYPE                DESCRIPTION

        void empty()
        {
            for(int j=0;j<nnz;j++)
            {
                R i_ = last;
                last = i[i_];
                i[i_] = -1;
                o[i_] = -1;              
            }
            nnz = 0;
        }

        // resize - Secondary constructor
        //
        // ---------------------- Arguments -------------------------------------------------------------------------
        // INPUTS
        // NAME             TYPE                DESCRIPTION
        // n_               (P)                 Size of the sparse line or column
                 
        void resize(P n_)
        {
            n = n_;
            last = P(-2);
            tlast = P(-2);
            nnz = P(0);
            i.resize(n,R(-1));
            o.resize(n,R(-1));
        };

        // rewind - Rewind the temporary pointer to access the list
        //
        // ---------------------- Arguments -------------------------------------------------------------------------
        // INPUTS
        // NAME             TYPE                DESCRIPTION

        void rewind()
        {
            tlast = last;
        };

        // next - Return next element on the list
        //
        // ---------------------- Arguments -------------------------------------------------------------------------
        // INPUTS
        // NAME             TYPE                DESCRIPTION
        //
        // OUTPUTS
        // NAME             TYPE                DESCRIPTION
        // (rval)           (R)                 Index of the next nonzero value

        inline void next(R& i_)
        {
            // Reverse iterator
            i_ = tlast;
            tlast = i[i_];
        }; 

        inline void next(R& i_, R& o_)
        {
            // Reverse iterator
            i_ = tlast;
            o_=o[i_];
            tlast = i[i_];
        };  
         

        // sparseAccumulator - Deconstructor
        //
        // ---------------------- Arguments -------------------------------------------------------------------------
        // INPUTS
        // NAME             TYPE                DESCRIPTION

        ~sparseAccumulatorSymbolic()
        {

        };
};



#endif

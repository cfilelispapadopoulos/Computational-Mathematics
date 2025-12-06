#include "qsort.hpp"
#include <iostream>
#include <vector>

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



// qSort2 - Function that sorts two arrays (integer and double) with respect to first of them using quicksort. 
//          Sorting is performed in place
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
// arr              (int*)              Pointer to a vector of integers.
// arr2             (double*)           Pointer to a vector of doubles.
// left             (int)               Integer index of the beginning of subarray.
// right            (int)               Integer index of the ending of subvector such that elements of the
//                                      subarray to be sorted reside in [left,right]. To sort a vector on n
//                                      elements left = 0 and right = n-1.
// ascdes           (int)               Integer either 0 or 1 denoting either ascending or descending type of
//                                      sorting.
//
// OUTPUT
// NAME             TYPE                DESCRIPTION
// arr              (int*)              Pointer to a vector of integers.
// arr2             (double*)           Pointer to a vector of doubles.


template<class R, class S,class T> void qSort2(S *arr, T *arr2, R left, R right, int ascdes)
{
	R i = left, j = right;
	R pivot = arr[(left + right) / 2];
    S tmp;
    T tmp2;
	while (i <= j) 
	{
        if (ascdes==0)
        {
		    while (arr[i] < pivot) i++;
            while (arr[j] > pivot) j--;
        }
        else
        {
           	while (arr[i] > pivot) i++;
            while (arr[j] < pivot) j--;
        }
        if (i <= j) 
        {
		    tmp    = arr[i];
            arr[i] = arr[j];
            arr[j] = tmp;
            tmp2    = arr2[i];
            arr2[i] = arr2[j];
            arr2[j] = tmp2;
            i++;
            j--;
        }
	}
	if (left < j)
		qSort2(arr,arr2, left, j, ascdes);
	if (i < right)
        qSort2(arr,arr2, i, right, ascdes);
}

// Declarations in order to avoid linker errors
template void qSort2<int,int,int>(int *arr, int *arr2, int left, int right, int ascdes);
template void qSort2<int,int,double>(int *arr, double *arr2, int left, int right, int ascdes);


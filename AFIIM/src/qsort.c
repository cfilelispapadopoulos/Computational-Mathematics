#include "qsort.h"

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
// left             (int*)              Integer index of the beginning of subarray.
// right            (int*)              Integer index of the ending of subvector such that elements of the
//                                      subarray to be sorted reside in [left,right]. To sort a vector on n
//                                      elements left = 0 and right = n-1.
// ascdes           (int*)              Integer either 0 or 1 denoting either ascending or descending type of
//                                      sorting.
//
// OUTPUT
// NAME             TYPE                DESCRIPTION
// arr              (int*)              Pointer to a vector of integers.
// arr2             (double*)           Pointer to a vector of doubles.


void qSort2(int *arr, double *arr2, int left, int right, int ascdes)
{
	int i = left, j = right;
	int pivot = arr[(left + right) / 2];
    int tmp;
    double tmp2;
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


// qSort2i - Function that sorts two arrays (integer and integer) with respect to first of them using quicksort. 
//           Sorting is performed in place
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
// left             (int*)              Integer index of the beginning of subarray.
// right            (int*)              Integer index of the ending of subvector such that elements of the
//                                      subarray to be sorted reside in [left,right]. To sort a vector on n
//                                      elements left = 0 and right = n-1.
// ascdes           (int*)              Integer either 0 or 1 denoting either ascending or descending type of
//                                      sorting.
//
// OUTPUT
// NAME             TYPE                DESCRIPTION
// arr              (int*)              Pointer to a vector of integers.
// arr2             (int*)              Pointer to a vector of integers.


void qSort2i(int *arr, int *arr2, int left, int right, int ascdes)
{
	int i = left, j = right;
	int pivot = arr[(left + right) / 2];
    int tmp, tmp2;
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
		qSort2i(arr,arr2, left, j, ascdes);
	if (i < right)
        qSort2i(arr,arr2, i, right, ascdes);
}

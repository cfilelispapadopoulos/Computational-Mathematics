#include "rw_matrix.h"
#include "mmio.h"
#include "qsort.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
                     int **Ai)
{
    // Matrix code variable
    MM_typecode matcode; 
    // Return code
    int ret_code;
    // Pointer to file
    FILE *f;
    // Variables for dimensions and number of nonzeroes
    int M,N,nnz;
    // Nonzero counter
    int tnnz = 0;
    // Pointers to retain coordinates and values
    int *I,*J,*tI,*tJ;
    double *val,*tval;
    // Iterators
    int i,idx;

    // Check existence of file and open for reading
    if ((f = fopen(filename, "r")) == NULL)
    {
        printf("File does not exist\n");
        exit(1);
    }

    // Read matrix banner
    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("Could not process Matrix Market banner.\n");
        exit(1);
    }    
    
    // Check the type of the matrix since only real general is supported (and symmetric is turned into general)
    if (mm_is_complex(matcode) || mm_is_array(matcode) || mm_is_integer(matcode) || mm_is_pattern(matcode) ||
        mm_is_hermitian(matcode) || mm_is_skew(matcode))
    {
        printf("Sorry, this application does not support ");
        printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
        exit(1);
    }

    // Get dimensions and number of nonzero elements
    if ((ret_code = mm_read_mtx_crd_size(f, &M, &N, &nnz)) !=0)
    {
        printf("Problem at reading dimensions and nonzero elements\n");
        exit(1);
    }

    // Check if matrix is square
    if (M != N)
    {
        printf("Matrix is not square\n");
        exit(1);
    }
    
    // Allocate more space in memory
    I = (int *)malloc(nnz * sizeof(int));
    J = (int *)malloc(nnz * sizeof(int));
    val = (double *)malloc(nnz * sizeof(double));
    
    // Iterate and read nonzero elements
    for (i=0; i<nnz; i++)
    {
        fscanf(f, "%d %d %lg\n", &I[i], &J[i], &val[i]);
        I[i]--;  /* adjust from 1-based to 0-based */
        J[i]--;
    }
    
    // Close file
    fclose(f);

    // Check if symmetric
    if (mm_is_symmetric(matcode))
    {
        // Count off-diagonal nonzeros
        for(i=0; i<nnz; i++)
            if(I[i] != J[i])
                tnnz++;

        // Allocate temporary arrays
        tI   = (int *)malloc((nnz + tnnz) * sizeof(int));
        tJ   = (int *)malloc((nnz + tnnz) * sizeof(int));
        tval = (double *)malloc((nnz + tnnz) * sizeof(double));

        // Copy elements
        memcpy(tI,I,nnz * sizeof(int));
        memcpy(tJ,J,nnz * sizeof(int));
        memcpy(tval,val,nnz * sizeof(double));

        // Free old storage
        free(I);
        free(J);
        free(val);

        // Form symmetric part
        tnnz = nnz;
        for(i=0;i<nnz;i++)
        {
            if (tI[i] != tJ[i])
            {
                tJ[tnnz] = tI[i];
                tI[tnnz] = tJ[i];
                tval[tnnz] = tval[i];
                tnnz++;
            }
        }

        // Set updated values
        nnz = tnnz;
        I = tI;
        J = tJ;
        val = tval;
        tI = NULL;
        tJ = NULL;
        tval = NULL;
    }

    // Prepare CSR arrays
    *Av = (double *)malloc(nnz * sizeof(double));
    *Aj = (int *)malloc(nnz * sizeof(int));
    *Ai = (int *)malloc((N+1) * sizeof(int));

    // Count offsets
    for(i=0;i<N+1;i++) (*Ai)[i] = 0;
    for(i=0;i<nnz;i++) (*Ai)[I[i]+1]++;
    for(i=0;i<N;i++) (*Ai)[i+1] += (*Ai)[i];

    // Populate column index array
    for(i=0;i<nnz;i++)
    {
        idx = (*Ai)[I[i]];
        (*Aj)[idx] = J[i];
        (*Av)[idx] = val[i];
        (*Ai)[I[i]]++;
    }

    // Rectify offsets
    for(i=N;i>0;i--) (*Ai)[i] = (*Ai)[i-1];
    (*Ai)[0] = 0;

    // Store size
    (*n) = N;

    // Sort the arrays
    for(i=0;i<N;i++) qSort2(*Aj,*Av,(*Ai)[i],(*Ai)[i+1]-1,0);

    // Free the coordinate structure
    free(I);
    free(J);
    free(val);
}

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
                     double **B)
{
    // Matrix code variable
    MM_typecode matcode; 
    // Return code
    int ret_code;
    // Pointer to file
    FILE *f;
    // Variables for dimensions and number of nonzeroes
    int M,N;
    // Iterators
    int i;

    // Check existence of file and open for reading
    if ((f = fopen(filename, "r")) == NULL)
    {
        printf("File does not exist\n");
        exit(1);
    }

    // Read matrix banner
    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("Could not process Matrix Market banner.\n");
        exit(1);
    }    
    
    // Check the type of the matrix since only real general is supported (and symmetric is turned into general)
    if (mm_is_complex(matcode) || mm_is_sparse(matcode) || mm_is_integer(matcode) || mm_is_pattern(matcode) ||
        mm_is_hermitian(matcode) || mm_is_skew(matcode))
    {
        printf("Sorry, this application does not support ");
        printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
        exit(1);
    }

    // Get dimensions and number of nonzero elements
    if ((ret_code = mm_read_mtx_array_size(f, &M, &N)) !=0)
    {
        printf("Problem at reading dimensions\n");
        exit(1);
    }

    // Check if it is a vector
    if (N != 1)
    {
        printf("The input file does not contain a vector\n");
        exit(1);
    }

    // Allocate more space in memory
    *B = (double *) malloc(M * sizeof(double));

    // Iterate and read nonzero elements
    for (i=0; i<M; i++) fscanf(f, "%lg\n", &(*B)[i]);

    // Close file
    fclose(f);

    // Store size
    (*n) = M;
}

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
                  double *x)
{
    // Matrix code variable
    MM_typecode matcode; 
    // Pointer to file
    FILE *f;
    // Iterators
    int i;

    // Check existence of file and open for reading
    if ((f = fopen(filename, "w")) == NULL)
    {
        printf("File could not be opened\n");
        exit(1);
    }

    // Set properties
    mm_set_matrix(&matcode);
    mm_set_dense(&matcode);
    mm_set_real(&matcode);
    mm_set_general(&matcode);

    // Write the banner to the file
    mm_write_banner(f,matcode);

    // Write dimensions of the array to be stored
    mm_write_mtx_array_size(f,n,1);

    // Write matrix
    for (i=0; i<n; i++)
        fprintf(f, "%20.19g\n", x[i]);   

    // Close the file
    fclose(f);

}
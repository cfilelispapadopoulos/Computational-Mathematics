#include "rw_matrix.hpp"
#include "mmio.hpp"
#include "qsort.hpp"
#include <iostream>
#include <cstdlib>
#include <algorithm>
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
// [1] C. K. Filelis - Papadopoulos (2025). Parallel Sparsity Patterns for Factored Incomplete Inverse Matrices. In Review.


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
// filename         (string)            A string retaining the name of the file. It can be a path also.
// mtype            (integer)           Storage type of the returned matrix (COO,CSR,CSC)
//
// OUTPUT
// NAME             TYPE                DESCRIPTION
// A                (sparseMatrix)      Sparse matrix class
//
//
// References
// [1] MNIST. Matrix Market. https://math.nist.gov/MatrixMarket/ (last accessed: 20/9/2024).

void read_coeff_matrix(std::string filename, sparseMatrix<int,int,double> &A, int mtype)
{
    // Matrix code variable
    MM_typecode matcode; 
    // Return code
    int ret_code;
    // Pointer to file
    FILE *f;
    // Variables for dimensions and number of nonzeroes
    int M,N,nnz;
    // Iterators
    int i,idx;

    // Check existence of file and open for reading
    if ((f = fopen(filename.c_str(), "r")) == NULL)
    {
        std::cout<<"File "<<filename<<" does not exist\n";
        exit(1);
    }

    // Read matrix banner
    if (mm_read_banner(f, &matcode) != 0)
    {
        std::cout<<"Could not process Matrix Market banner.\n";
        exit(1);
    }    
    
    // Check the type of the matrix since only real general is supported (and symmetric is turned into general)
    if (mm_is_complex(matcode) || mm_is_array(matcode) || mm_is_integer(matcode) || mm_is_pattern(matcode) ||
        mm_is_hermitian(matcode) || mm_is_skew(matcode))
    {
        std::cout<<"Sorry, this application does not support ";
        std::cout<<"Market Market type: ["<<mm_typecode_to_str(matcode)<<"]\n";
        exit(1);
    }

    // Get dimensions and number of nonzero elements
    if ((ret_code = mm_read_mtx_crd_size(f, &M, &N, &nnz)) !=0)
    {
        std::cout<<"Problem at reading dimensions and nonzero elements\n";
        exit(1);
    }
    
    // Allocate more space in memory
    std::vector<int> I(nnz),J(nnz);
    std::vector<double> val(nnz);
    
    // Iterate and read nonzero elements
    // Check symmetric and flip to get upper
    // triangular part
    if (mm_is_symmetric(matcode))
        for (i=0; i<nnz; i++)
        {
            fscanf(f, "%d %d %lg\n", &J[i], &I[i], &val[i]);
            I[i]--;  /* adjust from 1-based to 0-based */
            J[i]--;
        }    
    else
        for (i=0; i<nnz; i++)
        {
            fscanf(f, "%d %d %lg\n", &I[i], &J[i], &val[i]);
            I[i]--;  /* adjust from 1-based to 0-based */
            J[i]--;
        }
    
    // Close file
    fclose(f);

    // Prepare CSR arrays
    A.resize(mtype,mm_is_symmetric(matcode)?(SYMMETRIC):(GENERAL),M,N,nnz);

    if (mtype == CSR)
    {
        // Count offsets
        for(i=0;i<M+1;i++) A.i[i] = 0;
        for(i=0;i<nnz;i++) A.i[I[i]+1]++;
        for(i=0;i<M;i++) A.i[i+1] += A.i[i];

        // Populate column index array
        for(i=0;i<nnz;i++)
        {
            idx = A.i[I[i]];
            A.j[idx] = J[i];
            A.v[idx] = val[i];
            A.i[I[i]]++;
        }

        // Rectify offsets
        for(i=M;i>0;i--) A.i[i] = A.i[i-1];
        A.i[0] = 0;

        // Sort the arrays
        for(i=0;i<M;i++) qSort2(&(A.j[0]),&(A.v[0]),A.i[i],A.i[i+1]-1,0);
    }
    else if (mtype == CSC)
    {
        // Count offsets
        for(i=0;i<N+1;i++) A.j[i] = 0;
        for(i=0;i<nnz;i++) A.j[J[i]+1]++;
        for(i=0;i<N;i++) A.j[i+1] += A.j[i];

        // Populate column index array
        for(i=0;i<nnz;i++)
        {
            idx = A.j[J[i]];
            A.i[idx] = I[i];
            A.v[idx] = val[i];
            A.j[J[i]]++;
        }

        // Rectify offsets
        for(i=N;i>0;i--) A.j[i] = A.j[i-1];
        A.j[0] = 0;

        // Sort the arrays
        for(i=0;i<N;i++) qSort2(&(A.i[0]),&(A.v[0]),A.j[i],A.j[i+1]-1,0);
    }
    else if (mtype == COO)
    {
        A.i.swap(I);
        A.j.swap(J);
        A.v.swap(val);
    }
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
// filename         (string)            A string retaining the name of the file. It can be a
//                                      path also.
//
// OUTPUT
// NAME             TYPE                DESCRIPTION
// B                (vector)            A double vector of size n retaining the values of B.
//
//
// References
// [1] MNIST. Matrix Market. https://math.nist.gov/MatrixMarket/ (last accessed: 20/9/2024).

void read_rhs_vector(std::string filename, 
                     std::vector<double> &B)
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
    if ((f = fopen(filename.c_str(), "r")) == NULL)
    {
        std::cout<<"File "<<filename<<" does not exist\n";
        exit(1);
    }

    // Read matrix banner
    if (mm_read_banner(f, &matcode) != 0)
    {
        std::cout<<"Could not process Matrix Market banner.\n";
        exit(1);
    }    
    
    // Check the type of the matrix since only real general is supported (and symmetric is turned into general)
    if (mm_is_complex(matcode) || mm_is_sparse(matcode) || mm_is_integer(matcode) || mm_is_pattern(matcode) ||
        mm_is_hermitian(matcode) || mm_is_skew(matcode))
    {
        std::cout<<"Sorry, this application does not support ";
        std::cout<<"Market Market type: ["<<mm_typecode_to_str(matcode)<<"]\n";
        exit(1);
    }

    // Get dimensions and number of nonzero elements
    if ((ret_code = mm_read_mtx_array_size(f, &M, &N)) !=0)
    {
        std::cout<<"Problem at reading dimensions\n";
        exit(1);
    }

    // Check if it is a vector
    if (N != 1)
    {
        std::cout<<"The input file does not contain a vector\n";
        exit(1);
    }

    // Allocate more space in memory
    B.resize(M);

    // Iterate and read nonzero elements
    for (i=0; i<M; i++) fscanf(f, "%lg\n", &B[i]);

    // Close file
    fclose(f);
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
// filename         (string)            A string retaining the name of the file. It can be a
//                                      path also.
// x                (vector)            A double vector of size n retaining the values of B.
//
//
// References
// [1] MNIST. Matrix Market. https://math.nist.gov/MatrixMarket/ (last accessed: 20/9/2024).

void write_vector(std::string filename, 
                  std::vector<double> &x)
{
    // Matrix code variable
    MM_typecode matcode; 
    // Pointer to file
    FILE *f;
    // Iterators
    int i;
    // Size of vector
    int n = x.size();

    // Check existence of file and open for reading
    if ((f = fopen(filename.c_str(), "w")) == NULL)
    {
        std::cout<<"File could not be opened\n";
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

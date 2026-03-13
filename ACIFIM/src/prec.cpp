#include "prec.hpp"
#include "minmax.hpp"
#include <iostream>
#include <cmath>
#include <cfloat>

// DISCLAIMER
// The use of the code is regulated by the following copyright agreement.
//
// ACIFIM software is freely available for scientific (non-commercial) use.
// 1. The code can be used only for the purpose of internal research, excluding any commercial use of the ACIFIM
//    software as such or as a part of a software product. Users who want to integrate ACIFIM sofware or parts of
//    it into commercial products require a license agreement.
// 2. ACIFIM is provided "as is" and for the purpose described at the previous point only. In no circumstances can
//    neither the authors nor their institutions be held liable for any deficiency, fault or other mishappening
//    with regard to the use or performance of ACIFIM.
// 3. All scientific publications, for which ACIFIM software has been used, shall mention its usage and refer to
//    the publication [1] in the References section below.
//
//
// References
// [1] C. K. Filelis - Papadopoulos (2026). Adaptive Combined Incomplete Factorizations and Inverse Matrices. To be submitted.

// ACIFIM - Adaptive Combined Incomplete Factorization and Inverse Matrix
//
// Computes the incomplete inverse matrix based preconditioner in factored form M = G D^{i+1} H, of a general sparse
// matrix A stored in Compressed Sparse Row (CSR) storage format (ordered) along with an incomplete factorization LDU,
// following a recursive approach [1]. The method adaptively computes positions and values of the elements of the
// factors based on the dtol parameter [0,...,1].
//
// 			 | A_i b |   |       L_i        0 | | D_i     0 | | U_i   D_i^{-1} H_i b |
// A_{i+1} = |		 | = |				      | |  		    | |	        		     | 
//			 |  c  d |   | c G_i D_i^{-1}   1 | |  0      s | |  0          1        |
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
// n                (integer)           Size of square general sparse matrix A.
// Av               (double*)           Vector of size nnz(A) retaining the values of A (CSR).
// Aj               (integer*)          Vector of size nnz(A) retaining the column indices of the
//                                      elements of matrix A (CSR).
// Ai               (integer*)          Vector of size n+1 retaining the offsets of the rows of A (CSR).
// elemPerRowCol    (integer)           Initial number of elements (>0) in the rows and columns of the flexible
//                                      storage format. A small value in case of a dense preconditioner
//                                      might lead to large number of reallocations during formation of
//                                      the preconditioner. A good value is 10.
// growth           (integer)           The number of elements (>0) to be added to the already allocated space
//                                      in case it is full. A small value in case of a dense preconditioner
//                                      might lead to large number of reallocations during formation of
//                                      the preconditioner. A good value is 5.
// dtol1            (double)            Drop tolerance parameter in [0,...,1] which controls the density of
//                                      the LDU preconditioner. A value close to zero leads to the computation of
//                                      a very dense preconditioner which impacts performance. A value of dtol
//                                      close to one leads to a very sparse preconditioner (diagonal) which may
// dtol2            (double)            Drop tolerance parameter in [0,...,1] which controls the density of
//                                      the GD^{-1}H preconditioner. A value close to zero leads to the computation of
//                                      a very dense preconditioner which impacts performance. A value of dtol
//                                      close to one leads to a very sparse preconditioner (diagonal) which may
//                                      be ineffective. A good initial value is 0.1.
// eta              (double)            Threshold for the diagonal elements. In case of values lower than the
//                                      threshold the values are substituted with (10^{-4}+dtol).
//                                      A good initial value is approximately 10^{-8}.
// shift			(double)			Diagonal shift such that s_i = s_i + shift | A_{i,i} |.
// coeff			(double)			Multiplier that increases the drop tolerance to meet element limit.
// element_limit    (integer)			Element limit for the rows and columns of the factors.
// filtit			(integer)			Number of iterations to meet the element limit.
//
// OUTPUT
// NAME             TYPE                DESCRIPTION
// Gval             (double**)          A double pointer, initally set to NULL, which upon exit points to
//                                      allocated space of size nnz(G) retaining the values of the
//                                      factor G (CSR).
// Gindj            (integer**)         An integer pointer, initally set to NULL, which upon exit points to
//                                      llocated space of size nnz(G) retaining the column indices of the
//                                      elements of the factor G (CSR).
// Gindi            (integer**)         An integer pointer, initially set to NULL, which upon exit points to
//                                      allocated space of size n+1 retaining the row offsets of the
//                                      factor G (CSR).
// IDv              (double**)          A double pointer, initially set to NULL, which upon exit points to
//                                      to allocated space of size n+1 retaining the elements of the
//                                      diagonal factor D^{-1}
// Hval             (double**)          A double pointer, initally set to NULL, which upon exit points to
//                                      allocated space of size nnz(H) retaining the values of the
//                                      factor H (CSR).
// Hindj            (integer**)         An integer pointer, initally set to NULL, which upon exit points to
//                                      allocated space of size nnz(G) retaining the column indices of the
//                                      elements of the factor H (CSR)
// Hindi            (integer**)         An integer pointer, initially set to NULL, which upon exit points to
//                                      allocated space of size n+1 retaining the row offsets of the
//                                      factor H (CSR).
// Lval             (double**)          A double pointer, initally set to NULL, which upon exit points to
//                                      allocated space of size nnz(L) retaining the values of the
//                                      factor L (CSR).
// Lindj            (integer**)         An integer pointer, initally set to NULL, which upon exit points to
//                                      allocated space of size nnz(L) retaining the column indices of the
//                                      elements of the factor L (CSR)
// Lindi            (integer**)         An integer pointer, initially set to NULL, which upon exit points to
//                                      allocated space of size n+1 retaining the row offsets of the
//                                      factor L (CSR).
// Uval             (double**)          A double pointer, initally set to NULL, which upon exit points to
//                                      allocated space of size nnz(U) retaining the values of the
//                                      factor U (CSR).
// Uindj            (integer**)         An integer pointer, initally set to NULL, which upon exit points to
//                                      allocated space of size nnz(U) retaining the column indices of the
//                                      elements of the factor U (CSR)
// Uindi            (integer**)         An integer pointer, initially set to NULL, which upon exit points to
//                                      allocated space of size n+1 retaining the row offsets of the
//                                      factor U (CSR).
//
//
// References
// [1] C. K. Filelis - Papadopoulos (2026). Adaptive Combined Incomplete Factorizations and Inverse Matrices. To be submitted.

int acifim(int n,
			double *Av,
			int *Aj,
			int *Ai,
			double **Gval,
			int **Gindj,
			int **Gindi,
			double **IDv,
			double **Hval,
			int **Hindj,
			int **Hindi,
			double **Lval,
			int **Lindj,
			int **Lindi,
			double **Uval,
			int **Uindj,
			int **Uindi,
			double dtol1,
			double dtol2,
			int elemPerRowCol,
			int growth,
			double eta,
			double shift,
			double coeff,
			int element_limit,
			int filtit)
{
	// Mutliplier to enforce filtering
	double multiplier;

	// Iteration variables
	int i, j, k, jdx, kdx;

	// Lower, Upper and Diagonal parts of coefficient matrix
	double *Uv = NULL, *Lv = NULL;
	int *Ui = NULL, *Uj = NULL, *Li = NULL, *Lj = NULL;
	double *Dv = NULL;

	// Temp sparse vectors and other variables
	double *tc = NULL, *ttc = NULL;
	int *itc = NULL, *ittc = NULL;
	int ltc = 0, lttc = 0, lastc = -2, lasttc = -2;
	double *tb = NULL, *ttb = NULL;
	int *itb = NULL, *ittb = NULL;
	int ltb = 0, lttb = 0, lastb = -2, lasttb = -2;
	int tlast, cnt;

	double sum;

	// Norm - retaining vectors
	double *gnrms = NULL, *hnrms = NULL, *gnrmsc = NULL, *hnrmsr = NULL;

	// Initialize Temp vectors
	tc = new double[n];
	ttc = new double[n];
	itc = new int[n];
	ittc = new int[n];
	tb = new double[n];
	ttb = new double[n];
	itb = new int[n];
	ittb = new int[n];
	for (i = 0; i < n; i++)
	{
		itc[i] = -1;
		ittc[i] = -1;
		itb[i] = -1;
		ittb[i] = -1;
		tb[i] = 0.0;
		tc[i] = 0.0;
		ttb[i] = 0.0;
		ttc[i] = 0.0;
	}
	// Norms of rows and columns of factors G and H
	gnrms = new double[n];
	gnrmsc = new double[n];
	hnrms = new double[n];
	hnrmsr = new double[n];

	// Initialize to 1 to avoid consideration of the diagonal on the fly
	for (i = 0; i < n; i++)
		gnrms[i] = 1.0;
	for (i = 0; i < n; i++)
		hnrms[i] = 1.0;
	for (i = 0; i < n; i++)
		gnrmsc[i] = 1.0;
	for (i = 0; i < n; i++)
		hnrmsr[i] = 1.0;

	// Pointers to factors G,H,L,U in hybrid CSR - CSC format and inverse diagonal factor
	double **Gv = NULL, **GTv = NULL, **UTv = NULL;
	int **Gi = NULL, **GTi = NULL, **UTi = NULL;
	int *cG = NULL, *cGT = NULL, *mGT = NULL, *cUT = NULL, *mUT = NULL;
	double **Hv = NULL, **HTv = NULL, **LTv = NULL;
	int **Hi = NULL, **HTi = NULL, **LTi = NULL;
	int *cH = NULL, *cHT = NULL, *mHT = NULL, *cLT = NULL, *mLT = NULL;
	(*IDv) = new double[n];
	Gv = new double *[n];
	Gi = new int *[n];
	cG = new int[n];
	Hv = new double *[n];
	Hi = new int *[n];
	cH = new int[n];
	GTv = new double *[n];
	GTi = new int *[n];
	cGT = new int[n];
	mGT = new int[n];
	HTv = new double *[n];
	HTi = new int *[n];
	cHT = new int[n];
	mHT = new int[n];
	UTv = new double *[n];
	UTi = new int *[n];
	cUT = new int[n];
	mUT = new int[n];
	LTv = new double *[n];
	LTi = new int *[n];
	cLT = new int[n];
	mLT = new int[n];

	for (i = 0; i < n; i++)
	{
		HTv[i] = new double[elemPerRowCol];
		HTi[i] = new int[elemPerRowCol];
		GTv[i] = new double[elemPerRowCol];
		GTi[i] = new int[elemPerRowCol];
		UTv[i] = new double[elemPerRowCol];
		UTi[i] = new int[elemPerRowCol];
		LTv[i] = new double[elemPerRowCol];
		LTi[i] = new int[elemPerRowCol];
	}

	for (i = 0; i < n; i++)
	{
		cHT[i] = 0;
		mHT[i] = elemPerRowCol;
		cGT[i] = 0;
		mGT[i] = elemPerRowCol;
		cLT[i] = 0;
		mLT[i] = elemPerRowCol;
		cUT[i] = 0;
		mUT[i] = elemPerRowCol;
	}

	// Lines, Columns and Diagonal
	double d;
	double s, s1, s2;

	// Preparation to store in CSC
	Ui = new int[n + 1];
	for (i = 0; i < n + 1; i++)
		Ui[i] = 0;

	// Nonzero elements (counters)
	int Unnz = 0;
	int Lnnz = 0;
	int Gnnz = 0;
	int Hnnz = 0;

	// Count nonzeros of upper and lower part of coefficient matrix A
	for (i = 0; i < n; i++)
	{
		for (j = Ai[i]; j < Ai[i + 1]; j++)
		{
			jdx = Aj[j];
			if (jdx > i)
			{
				Ui[jdx + 1]++;
				Unnz++;
			}
			else if (jdx < i)
				Lnnz++;
		}
	}

	// Cummulative sum of offsets
	for (i = 0; i < n; i++)
		Ui[i + 1] += Ui[i];

	// Allocate matrices of Lower, Diagonal and Upper part of matrix A
	Dv = new double[n];
	if (Unnz > 0)
	{
		Uv = new double[Unnz];
		Uj = new int[Unnz];
	}
	if (Lnnz > 0)
	{
		Lv = new double[Lnnz];
		Lj = new int[Lnnz];
	}
	Li = new int[n + 1];

	// Split coefficient matrix A into Lower, Diagonal and Upper part
	Li[0] = 0;
	Lnnz = 0;
	for (i = 0; i < n; i++)
	{
		Dv[i] = 0.0;
		for (j = Ai[i]; j < Ai[i + 1]; j++)
		{
			jdx = Aj[j];
			if (jdx > i)
			{
				Uv[Ui[jdx]] = Av[j];
				Uj[Ui[jdx]] = i;
				Ui[jdx]++;
			}
			else if (jdx < i)
			{
				Lv[Lnnz] = Av[j];
				Lj[Lnnz] = Aj[j];
				Lnnz++;
			}
			else
			{
				Dv[i] = Av[j];
			}
		}
		Li[i + 1] = Lnnz;
		if (Dv[i] == 0.0)
			Dv[i] = dtol2;
	}

	// Correction of offsets
	for (i = n - 1; i >= 0; i--)
		Ui[i + 1] = Ui[i];
	Ui[0] = 0;

	// First elements of the factors of the approximate inverse
	s = Dv[0] + shift * fabs(Dv[0]);
	if (fabs(s) < eta)
		(*IDv)[0] = 1.0 / (1e-4 + dtol2);
	else
		(*IDv)[0] = 1.0 / s;

	Gv[0] = new double[1];
	Gv[0][0] = 1.0;
	GTv[0][0] = 1.0;
	Gi[0] = new int[1];
	Gi[0][0] = 0;
	GTi[0][0] = 0;
	cG[0] = 1;
	cGT[0] = 1;

	Hv[0] = new double[1];
	Hv[0][0] = 1.0;
	HTv[0][0] = 1.0;
	Hi[0] = new int[1];
	Hi[0][0] = 0;
	HTi[0][0] = 0;
	cH[0] = 1;
	cHT[0] = 1;

	// Diagonal elements for L,U factors
	for (int i = 0; i < n; i++)
	{
		UTv[i][0] = 1.0;
		UTi[i][0] = i;
		cUT[i] = 1;

		LTv[i][0] = 1.0;
		LTi[i][0] = i;
		cLT[i] = 1;
	}

	// Initiate computation of remaining (n-1) columns, rows and elements of factors G, H and D
	for (i = 0; i < n - 1; i++)
	{
		// g_{i+1} = - G D^{-1} H b
		// t_b = H b
		for (j = Ui[i + 1]; j < Ui[i + 2]; j++)
		{
			jdx = Uj[j];
			for (k = 0; k < cHT[jdx]; k++)
			{
				kdx = HTi[jdx][k];
				if (itb[kdx] == -1)
				{
					tb[kdx] = Uv[j] * HTv[jdx][k];
					itb[kdx] = lastb;
					lastb = kdx;
					ltb++;
				}
				else
				{
					tb[kdx] += Uv[j] * HTv[jdx][k];
				}
			}
		}

		// Filtering u_{i+1} and store
		// Filter
		cnt = 0;
		tlast = lastb;
		for (j = 0; j < ltb; j++)
		{
			jdx = tlast;
			if (fabs((*IDv)[jdx] * tb[jdx]) <= dtol1 / (gnrmsc[jdx]))
			{
				tb[jdx] = 0.0;
				cnt++;
			}
			tlast = itb[jdx];
		}

		int l = 1;
		multiplier = 1.0;
		while ((ltb - cnt) > element_limit && l <= filtit)
		{
			cnt = 0;
			tlast = lastb;
			for (j = 0; j < ltb; j++)
			{
				jdx = tlast;
				if (fabs((*IDv)[jdx] * tb[jdx]) <= multiplier * dtol1 / (gnrmsc[jdx]))
				{
					tb[jdx] = 0.0;
					cnt++;
				}
				tlast = itb[jdx];
			}
			multiplier *= coeff;
			l++;
		}

		// Store
		tlast = lastb;
		for (j = 0; j < ltb; j++)
		{
			jdx = tlast;
			double tval = tb[jdx];
			if (fabs(tval) != 0.0)
			{
				if (cUT[jdx] + 1 >= mUT[jdx])
				{
					mUT[jdx] += growth;
					double *temp = new double[mUT[jdx]];
					int *tempi = new int[mUT[jdx]];

					for (k = 0; k < cUT[jdx]; k++)
						temp[k] = UTv[jdx][k];
					for (k = 0; k < cUT[jdx]; k++)
						tempi[k] = UTi[jdx][k];
					delete[] UTv[jdx];
					delete[] UTi[jdx];
					UTv[jdx] = temp;
					UTi[jdx] = tempi;
					temp = NULL;
					tempi = NULL;
				}
				kdx = cUT[jdx];
				UTv[jdx][kdx] = (*IDv)[jdx] * tval;
				UTi[jdx][kdx] = i + 1;
				cUT[jdx]++;
			}
			tlast = itb[jdx];
		}

		// tt_b = G D^{-1} t_b
		tlast = lastb;
		for (j = 0; j < ltb; j++)
		{

			jdx = tlast;
			double tval = tb[jdx];
			if (fabs(tval) != 0.0)
			{
				for (k = 0; k < cG[jdx]; k++)
				{
					kdx = Gi[jdx][k];
					if (ittb[kdx] == -1)
					{
						ttb[kdx] = tval * (*IDv)[jdx] * Gv[jdx][k];
						ittb[kdx] = lasttb;
						lasttb = kdx;
						lttb++;
					}
					else
					{
						ttb[kdx] += tval * (*IDv)[jdx] * Gv[jdx][k];
					}
				}
			}
			tlast = itb[jdx];
		}

		// h_{i+1} = - c G D^{-1} H
		// t_c = c G
		for (j = Li[i + 1]; j < Li[i + 2]; j++)
		{
			jdx = Lj[j];
			for (k = 0; k < cGT[jdx]; k++)
			{
				kdx = GTi[jdx][k];
				if (itc[kdx] == -1)
				{
					tc[kdx] = Lv[j] * GTv[jdx][k];
					itc[kdx] = lastc;
					lastc = kdx;
					ltc++;
				}
				else
				{
					tc[kdx] += Lv[j] * GTv[jdx][k];
				}
			}
		}

		// Filtering \ell_{i+1} and store
		// Filter
		cnt = 0;
		tlast = lastc;
		for (j = 0; j < ltc; j++)
		{
			jdx = tlast;
			if (fabs((*IDv)[jdx] * tc[jdx]) <= dtol1 / (hnrmsr[jdx]))
			{
				tc[jdx] = 0.0;
				cnt++;
			}
			tlast = itc[jdx];
		}

		l = 1;
		multiplier = 1.0;
		while ((ltc - cnt) > element_limit && l <= filtit)
		{
			cnt = 0;
			tlast = lastc;
			for (j = 0; j < ltc; j++)
			{
				jdx = tlast;
				if (fabs((*IDv)[jdx] * tc[jdx]) <= multiplier * dtol1 / (hnrmsr[jdx]))
				{
					tc[jdx] = 0.0;
					cnt++;
				}
				tlast = itc[jdx];
			}
			multiplier *= coeff;
			l++;
		}

		// Store
		tlast = lastc;
		for (j = 0; j < ltc; j++)
		{
			jdx = tlast;
			double tval = tc[jdx];
			if (fabs(tval) != 0.0)
			{
				if (cLT[jdx] + 1 >= mLT[jdx])
				{
					mLT[jdx] += growth;
					double *temp = new double[mLT[jdx]];
					int *tempi = new int[mLT[jdx]];

					for (k = 0; k < cLT[jdx]; k++)
						temp[k] = LTv[jdx][k];
					for (k = 0; k < cLT[jdx]; k++)
						tempi[k] = LTi[jdx][k];
					delete[] LTv[jdx];
					delete[] LTi[jdx];
					LTv[jdx] = temp;
					LTi[jdx] = tempi;
					temp = NULL;
					tempi = NULL;
				}
				kdx = cLT[jdx];
				LTv[jdx][kdx] = (*IDv)[jdx] * tval;
				LTi[jdx][kdx] = i + 1;
				cLT[jdx]++;
			}
			tlast = itc[jdx];
		}

		// tt_c = t_c D^{-1} H
		tlast = lastc;
		for (j = 0; j < ltc; j++)
		{
			jdx = tlast;
			double tval = tc[jdx];
			if (fabs(tval) != 0.0)
			{
				for (k = 0; k < cH[jdx]; k++)
				{
					kdx = Hi[jdx][k];
					if (ittc[kdx] == -1)
					{
						ttc[kdx] = (*IDv)[jdx] * tval * Hv[jdx][k];
						ittc[kdx] = lasttc;
						lasttc = kdx;
						lttc++;
					}
					else
					{
						ttc[kdx] += (*IDv)[jdx] * tval * Hv[jdx][k];
					}
				}
			}
			tlast = itc[jdx];
		}

		// Compute Schur Complement
		// First term s = d + c G D^{-1} H b
		d = Dv[i + 1];
		sum = 0.0;
		for (j = 0; j < ltb; j++)
		{
			jdx = lastb;
			sum += tb[jdx] * (*IDv)[jdx] * tc[jdx];
			lastb = itb[jdx];
			itb[jdx] = -1;
			tb[jdx] = 0.0;
		}
		ltb = 0;

		for (j = 0; j < ltc; j++)
		{
			jdx = lastc;
			lastc = itc[jdx];
			itc[jdx] = -1;
			tc[jdx] = 0.0;
		}
		ltc = 0;

		// Accumulate part of the Schur complement
		s = d + sum;

		// Compute correction of Schur complement
		// Second term s = s + c g_{i+1}
		s1 = 0.0;
		for (j = Li[i + 1]; j < Li[i + 2]; j++)
		{
			jdx = Lj[j];
			s1 -= Lv[j] * ttb[jdx];
			gnrmsc[i + 1] = mmax(gnrmsc[i + 1], fabs(ttb[jdx]));
		}

		// Filter and store row and column to lower and upper triangular factor G
		// Filter g_{i+1}
		cnt = 0;
		tlast = lasttb;
		for (j = 0; j < lttb; j++)
		{
			jdx = tlast;
			// Update norms for the G factor
			gnrms[jdx] = gnrms[jdx] + fabs(ttb[jdx]);
			if (fabs(ttb[jdx]) <= dtol2 * gnrms[jdx])
			{
				ttb[jdx] = 0.0;
				cnt++;
			}
			tlast = ittb[jdx];
		}

		l = 1;
		multiplier = 1.0;
		while ((lttb - cnt) > element_limit && l <= filtit)
		{
			cnt = 0;
			tlast = lasttb;
			for (j = 0; j < lttb; j++)
			{
				jdx = tlast;
				if (fabs(ttb[jdx]) <= multiplier * dtol2 * gnrms[jdx])
				{
					ttb[jdx] = 0.0;
					cnt++;
				}
				tlast = ittb[jdx];
			}
			multiplier *= coeff;
			l++;
		}

		// Store newly formed column g_{i+1} to CSC structure
		// Allocate adequate space
		Gv[i + 1] = new double[lttb - cnt + 1];
		Gi[i + 1] = new int[lttb - cnt + 1];
		cG[i + 1] = lttb - cnt + 1;
		cnt = 0;
		for (j = 0; j < lttb; j++)
		{
			jdx = lasttb;
			if (ttb[jdx] != 0.0)
			{
				Gv[i + 1][cnt] = -ttb[jdx];
				Gi[i + 1][cnt] = jdx;
				cnt++;
			}
			lasttb = ittb[jdx];
			ittb[jdx] = -1;
			ttb[jdx] = 0.0;
		}
		lttb = 0;
		// Diagonal Element
		Gv[i + 1][cnt] = 1.0;
		Gi[i + 1][cnt] = i + 1;

		// Compute correction of the Schur Complement
		// Third part s = s + h_{i+1} b
		s2 = 0.0;
		for (j = Ui[i + 1]; j < Ui[i + 2]; j++)
		{
			jdx = Uj[j];
			s2 -= Uv[j] * ttc[jdx];
			hnrmsr[i + 1] = mmax(hnrmsr[i + 1], fabs(ttc[jdx]));
		}

		// Filter h_{i+1}
		cnt = 0;
		tlast = lasttc;
		for (j = 0; j < lttc; j++)
		{
			jdx = tlast;
			// Update norms for the H factor
			hnrms[jdx] = hnrms[jdx] + fabs(ttc[jdx]);
			if (fabs(ttc[jdx]) <= dtol2 * hnrms[jdx])
			{
				ttc[jdx] = 0.0;
				cnt++;
			}
			tlast = ittc[jdx];
		}

		l = 1;
		multiplier = 1.0;
		while ((lttc - cnt) > element_limit && l <= filtit)
		{
			cnt = 0;
			tlast = lasttc;
			for (j = 0; j < lttc; j++)
			{
				jdx = tlast;
				if (fabs(ttc[jdx]) <= multiplier * dtol2 * hnrms[jdx])
				{
					ttc[jdx] = 0.0;
					cnt++;
				}
				tlast = ittc[jdx];
			}
			multiplier *= coeff;
			l++;
		}

		// Store newly formed row h_{i+1} to CSR structure
		// Allocate adequate space
		Hv[i + 1] = new double[lttc - cnt + 1];
		Hi[i + 1] = new int[lttc - cnt + 1];
		cH[i + 1] = lttc - cnt + 1;
		cnt = 0;
		for (j = 0; j < lttc; j++)
		{
			jdx = lasttc;
			if (ttc[jdx] != 0.0)
			{

				Hv[i + 1][cnt] = -ttc[jdx];
				Hi[i + 1][cnt] = jdx;
				cnt++;
			}
			lasttc = ittc[jdx];
			ittc[jdx] = -1;
			ttc[jdx] = 0.0;
		}
		lttc = 0;
		// Diagonal Element
		Hv[i + 1][cnt] = 1.0;
		Hi[i + 1][cnt] = i + 1;

		// Populate transpose G and H
		// G^T
		for (j = 0; j < cG[i + 1]; j++)
		{
			jdx = Gi[i + 1][j];
			// Extend allocation by growth elements if maximum is reached
			if (cGT[jdx] + 1 >= mGT[jdx])
			{
				mGT[jdx] += growth;
				double *t_b = new double[mGT[jdx]];
				int *t_bi = new int[mGT[jdx]];

				for (k = 0; k < cGT[jdx]; k++)
					t_b[k] = GTv[jdx][k];
				for (k = 0; k < cGT[jdx]; k++)
					t_bi[k] = GTi[jdx][k];
				delete[] GTv[jdx];
				delete[] GTi[jdx];
				GTv[jdx] = t_b;
				GTi[jdx] = t_bi;
				t_b = NULL;
				t_bi = NULL;
			}
			kdx = cGT[jdx];
			GTv[jdx][kdx] = Gv[i + 1][j];
			GTi[jdx][kdx] = i + 1;
			cGT[jdx]++;
		}

		// H^T
		for (j = 0; j < cH[i + 1]; j++)
		{
			jdx = Hi[i + 1][j];
			// Extend allocation by growth elements if maximum is reached
			if (cHT[jdx] + 1 >= mHT[jdx])
			{
				mHT[jdx] += growth;
				double *t_c = new double[mHT[jdx]];
				int *t_ci = new int[mHT[jdx]];

				for (k = 0; k < cHT[jdx]; k++)
					t_c[k] = HTv[jdx][k];
				for (k = 0; k < cHT[jdx]; k++)
					t_ci[k] = HTi[jdx][k];
				delete[] HTv[jdx];
				delete[] HTi[jdx];
				HTv[jdx] = t_c;
				HTi[jdx] = t_ci;
				t_c = NULL;
				t_ci = NULL;
			}
			kdx = cHT[jdx];
			HTv[jdx][kdx] = Hv[i + 1][j];
			HTi[jdx][kdx] = i + 1;
			cHT[jdx]++;
		}

		// Check and populate matrix IDv
		s += (s1 + s2) + shift * fabs(d);
		if (std::isnan(s))
			return 1;
		if (fabs(s) <= eta)
			s = (1e-4 + dtol2);
		(*IDv)[i + 1] = 1. / s;
	}

	// Clean unnecessary matrices
	if (Unnz > 0)
	{
		delete[] Uv;
		delete[] Uj;
	}
	delete[] Ui;

	if (Lnnz > 0)
	{
		delete[] Lv;
		delete[] Lj;
	}
	delete[] Li;

	delete[] Dv;

	delete[] cG;
	delete[] cH;
	for (i = 0; i < n; i++)
	{
		delete[] Gv[i];
		delete[] Gi[i];
		delete[] Hv[i];
		delete[] Hi[i];
	}
	delete[] Gv;
	delete[] Gi;
	delete[] Hv;
	delete[] Hi;

	delete[] gnrms;
	delete[] hnrms;
	delete[] gnrmsc;
	delete[] hnrmsr;

	// Extract to CSR arrays
	*Gindi = new int[n + 1];
	*Hindi = new int[n + 1];
	*Uindi = new int[n + 1];
	*Lindi = new int[n + 1];

	// Compute offsets for CSR retaining G and U
	(*Gindi)[0] = 0;
	for (i = 0; i < n; i++)
		(*Gindi)[i + 1] = (*Gindi)[i] + cGT[i];
	Gnnz = (*Gindi)[n];

	(*Uindi)[0] = 0;
	for (i = 0; i < n; i++)
		(*Uindi)[i + 1] = (*Uindi)[i] + cUT[i];
	Unnz = (*Uindi)[n];

	// Compute offsets for CSR retaining H and L
	for (i = 0; i < n + 1; i++)
		(*Hindi)[i] = 0;
	for (i = 0; i < n; i++)
	{
		for (j = 0; j < cHT[i]; j++)
		{
			jdx = HTi[i][j];
			(*Hindi)[jdx + 1]++;
		}
	}
	for (i = 0; i < n; i++)
		(*Hindi)[i + 1] += (*Hindi)[i];
	Hnnz = (*Hindi)[n];

	for (i = 0; i < n + 1; i++)
		(*Lindi)[i] = 0;
	for (i = 0; i < n; i++)
	{
		for (j = 0; j < cLT[i]; j++)
		{
			jdx = LTi[i][j];
			(*Lindi)[jdx + 1]++;
		}
	}
	for (i = 0; i < n; i++)
		(*Lindi)[i + 1] += (*Lindi)[i];
	Lnnz = (*Lindi)[n];

	// Allocate vectors retaining values and column indices
	(*Gval) = new double[Gnnz];
	(*Gindj) = new int[Gnnz];
	(*Hval) = new double[Hnnz];
	(*Hindj) = new int[Hnnz];
	(*Uval) = new double[Unnz];
	(*Uindj) = new int[Unnz];
	(*Lval) = new double[Lnnz];
	(*Lindj) = new int[Lnnz];

	// Populate G
	Gnnz = 0;
	for (i = 0; i < n; i++)
	{
		for (j = 0; j < cGT[i]; j++)
		{
			(*Gval)[Gnnz] = GTv[i][j];
			(*Gindj)[Gnnz] = GTi[i][j];
			Gnnz++;
		}
	}

	// Populate U
	Unnz = 0;
	for (i = 0; i < n; i++)
	{
		for (j = 0; j < cUT[i]; j++)
		{
			(*Uval)[Unnz] = UTv[i][j];
			(*Uindj)[Unnz] = UTi[i][j];
			Unnz++;
		}
	}

	// Populate H
	for (i = 0; i < n; i++)
	{
		for (j = 0; j < cHT[i]; j++)
		{
			jdx = HTi[i][j];
			(*Hindj)[(*Hindi)[jdx]] = i;
			(*Hval)[(*Hindi)[jdx]] = HTv[i][j];
			(*Hindi)[jdx]++;
		}
	}

	// Populate L
	for (i = 0; i < n; i++)
	{
		for (j = 0; j < cLT[i]; j++)
		{
			jdx = LTi[i][j];
			(*Lindj)[(*Lindi)[jdx]] = i;
			(*Lval)[(*Lindi)[jdx]] = LTv[i][j];
			(*Lindi)[jdx]++;
		}
	}

	// Reset offsets
	for (i = n; i >= 1; i--)
		(*Hindi)[i] = (*Hindi)[i - 1];
	(*Hindi)[0] = 0;

	for (i = n; i >= 1; i--)
		(*Lindi)[i] = (*Lindi)[i - 1];
	(*Lindi)[0] = 0;

	// Clean up
	delete[] cGT;
	delete[] cHT;
	delete[] cUT;
	delete[] cLT;
	delete[] mHT;
	delete[] mGT;
	delete[] mLT;
	delete[] mUT;
	for (i = 0; i < n; i++)
	{
		delete[] GTv[i];
		delete[] GTi[i];
		delete[] HTv[i];
		delete[] HTi[i];
		delete[] UTv[i];
		delete[] UTi[i];
		delete[] LTv[i];
		delete[] LTi[i];
	}
	delete[] GTv;
	delete[] GTi;
	delete[] HTv;
	delete[] HTi;
	delete[] UTv;
	delete[] UTi;
	delete[] LTv;
	delete[] LTi;

	delete[] tc;
	delete[] ttc;
	delete[] tb;
	delete[] ttb;
	delete[] itc;
	delete[] ittc;
	delete[] itb;
	delete[] ittb;

	return 0;
}

#include "prec.h"
#include "minmax.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

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



// AFIIM - Adaptive Factored Incomplete Inverse Matrix with robust filtering
//
// Computes the incomplete inverse matrix based preconditioner in factored form M = G D^{i+1} H, of a general sparse 
// matrix A stored in Compressed Sparse Row (CSR) storage format (ordered), following a recursive approach [1].
// The method adaptively computes positions and values of the elements of the factors based on the
// dtol parameter [0,...,1].
//
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
// dtol             (double)            Drop tolerance parameter in [0,...,1] which controls the density of
//                                      the preconditioner. A value close to zero leads to the computation of 
//                                      a very dense preconditioner which impacts performance. A value of dtol
//                                      close to one leads to a very sparse preconditioner (diagonal) which may
//                                      be ineffective. A good initial value is 0.1.
// eta              (double)            Threshold for the diagonal elements. In case of values lower than the 
//                                      threshold the values are substituted with (10^{-4}+dtol).
//                                      A good initial value is approximately 10^{-8}.
// shift			(double)			Diagonal shift such that s_i = s_i + shift | A_{i,i} |.
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
//
//
// References
// [1] C. K. Filelis - Papadopoulos (2024). Adaptive Factored Incomplete Inverse Matrices. In Review.

void afiim(int n, 
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
           double dtol, 
           int elemPerRowCol, 
           int growth, 
           double eta,
		   double shift)
{
    // Iteration variables
	int i,j,k,jdx,kdx;

	// Lower, Upper and Diagonal parts of coefficient matrix
	double *Uv,*Lv;
	int *Ui,*Uj,*Li,*Lj;
	double *Dv;

	// Temp sparse vectors and other variables
	double *tc,*ttc;
	int *itc,*ittc;
	int ltc=0,lttc=0,lastc=-2,lasttc=-2;
	double *tb,*ttb;
	int *itb,*ittb;
	int ltb=0,lttb=0,lastb=-2,lasttb=-2;
	int tlast,cnt;
	double sum;

	// Norm - retaining vectors
	double *gnrms,*hnrms,*unrms,*lnrms,*alrnrms,*aucnrms;

    // Initialize Temp vectors
	tc=(double *)malloc(n * sizeof(double));
	ttc=(double *)malloc(n * sizeof(double));
	itc=(int *)malloc(n * sizeof(int));
	ittc=(int *)malloc(n * sizeof(int));
	tb=(double *)malloc(n * sizeof(double));
	ttb=(double *)malloc(n * sizeof(double));
	itb=(int *)malloc(n * sizeof(int));
	ittb=(int *)malloc(n * sizeof(int));
	for(i=0;i<n;i++)
	{
		itc[i]=-1;
		ittc[i]=-1;
		itb[i]=-1;
		ittb[i]=-1;
		tb[i]=0.0;
		tc[i]=0.0;
		ttb[i]=0.0;
		ttc[i]=0.0;
	}
	// Norms of rows and columns of factors G and H
	gnrms=(double *)malloc(n * sizeof(double));
	hnrms=(double *)malloc(n * sizeof(double));

	// Matrices to retain row and column inf-norms of L and U
	unrms=(double *)malloc(n * sizeof(double));
	lnrms=(double *)malloc(n * sizeof(double));

	// Matrices to retain row and column inf-norms of tril(A) and triu(A)
	alrnrms=(double *)malloc(n * sizeof(double));
	aucnrms=(double *)malloc(n * sizeof(double));
	
	// Initialize to 1 to avoid consideration of the diagonal on the fly
	for(i=0;i<n;i++) gnrms[i]=1.0;
	for(i=0;i<n;i++) hnrms[i]=1.0;
	
	// Initialize to 1 to avoid consideration of the diagonal on the fly
	for(i=0;i<n;i++) unrms[i]=1.0;
	for(i=0;i<n;i++) lnrms[i]=1.0;

	// Initialize the vectors retaining the 1-norms of the rows and columns of the scaled
	// lower and upper part of the coefficient matrix, respectively

	for(i=0;i<n;i++) alrnrms[i]=1.0;
	for(i=0;i<n;i++) aucnrms[i]=1.0;
	
	// Pointers to factors G,H in hybrid CSR - CSC format and inverse diagonal factor
	double **Gv,**GTv;
	int **Gi,**GTi;
	int *cG,*cGT,*mGT;
	double **Hv,**HTv;
	int **Hi,**HTi;
	int *cH,*cHT,*mHT;
	(*IDv)=(double *)malloc(n * sizeof(double));
	Gv =(double **)malloc(n * sizeof(double*));
	Gi =(int **)malloc(n * sizeof(int*));
	cG =(int *)malloc(n * sizeof(int));
	Hv =(double **)malloc(n * sizeof(double*));
	Hi =(int **)malloc(n * sizeof(int*));
	cH =(int *)malloc(n * sizeof(int));
	GTv=(double **)malloc(n * sizeof(double*));
	GTi=(int **)malloc(n * sizeof(int*));
	cGT=(int *)malloc(n * sizeof(int));
	mGT=(int *)malloc(n * sizeof(int));
	HTv=(double **)malloc(n * sizeof(double*));
	HTi=(int **)malloc(n * sizeof(int*));
	cHT=(int *)malloc(n * sizeof(int));
	mHT=(int *)malloc(n * sizeof(int));
	
	for(i=0;i<n;i++)
	{
		HTv[i]=(double *)malloc(elemPerRowCol * sizeof(double));
		HTi[i]=(int *)malloc(elemPerRowCol * sizeof(int));
		GTv[i]=(double *)malloc(elemPerRowCol * sizeof(double));
		GTi[i]=(int *)malloc(elemPerRowCol * sizeof(int));
	}
	
	for(i=0;i<n;i++)
	{
		cHT[i]=0;
		mHT[i]=elemPerRowCol;
		cGT[i]=0;
		mGT[i]=elemPerRowCol;
	}

	// Lines, Columns and Diagonal
	double d;
	double s,s1,s2;

	// Preparation to store in CSC
	Ui=(int *)malloc((n+1) * sizeof(int));
	for(i=0;i<n+1;i++)
		Ui[i]=0;

	// Nonzero elements (counters)
	int Unnz=0;
	int Lnnz=0;
	int Gnnz=0;
	int Hnnz=0;
	
	// Count nonzeros of upper and lower part of coefficient matrix A
	for(i=0;i<n;i++)
	{
		for(j=Ai[i];j<Ai[i+1];j++)
		{
			jdx=Aj[j];
			if(jdx>i)
			{
				Ui[jdx+1]++;
				Unnz++;
			}
			else if(jdx<i)
				Lnnz++;
		}
	}
	
	// Cummulative sum of offsets
	for(i=0;i<n;i++)
		Ui[i+1]+=Ui[i];

	// Allocate matrices of Lower, Diagonal and Upper part of matrix A
	Dv=(double *)malloc(n * sizeof(double));
	if (Unnz>0)
	{
		Uv=(double *)malloc(Unnz * sizeof(double));
		Uj=(int *)malloc(Unnz * sizeof(int));
	}
	if (Lnnz>0)
	{
		Lv=(double *)malloc(Lnnz * sizeof(double));
		Lj=(int *)malloc(Lnnz * sizeof(int));
	}
	Li=(int *)malloc((n+1) * sizeof(int));

	// Split coefficient matrix A into Lower, Diagonal and Upper part
	Li[0]=0;
	Lnnz=0;
	for(i=0;i<n;i++)
	{
		Dv[i]=0.0;
		for(j=Ai[i];j<Ai[i+1];j++)
		{
			jdx=Aj[j];
			if(jdx>i)
			{
				Uv[Ui[jdx]]=Av[j];
				Uj[Ui[jdx]]=i;
				Ui[jdx]++;
			}
			else if(jdx<i)
			{	
				Lv[Lnnz]=Av[j];
				Lj[Lnnz]=Aj[j];
				Lnnz++;
			}
			else
			{
				Dv[i]=Av[j];
			}
		}
		Li[i+1]=Lnnz;
		if(Dv[i] == 0.0)
			Dv[i] = dtol;
	}

	// Correction of offsets
	for(i=n-1;i>=0;i--) Ui[i+1]=Ui[i];
	Ui[0]=0;

	// Computate the inf - norms of rows and columns of the lower and upper part of A
    // after scaling with the diagonal
    // Upper part
	for(i=0;i<n;i++)
	{
		for(j=Ui[i];j<Ui[i+1];j++)
		{
			int jdx=Uj[j];
			aucnrms[i]=mmax(aucnrms[i],fabs(Uv[j])/(sqrt(fabs(Dv[i]))*sqrt(fabs(Dv[jdx]))));
		}
	}
    // Lower part
	for(i=0;i<n;i++)
	{
		for(j=Li[i];j<Li[i+1];j++)
		{
			int jdx=Lj[j];
			alrnrms[i]=mmax(alrnrms[i],fabs(Lv[j])/(sqrt(fabs(Dv[i]))*sqrt(fabs(Dv[jdx]))));
		}
	}

    // First elements of the factors of the approximate inverse
	s = Dv[0]; //+ shift * fabs(Dv[0]);
    if (fabs(s) < eta)
        (*IDv)[0]=1.0/(1e-4+dtol);
    else
	    (*IDv)[0]=1.0/s;

	Gv[0]=(double *)malloc(sizeof(double));
	Gv[0][0]=1.0;
	GTv[0][0]=1.0;
	Gi[0]=(int *)malloc(sizeof(int));
	Gi[0][0]=0;
	GTi[0][0]=0;
	cG[0]=1;
	cGT[0]=1;
	
	Hv[0]=(double *)malloc(sizeof(double));
	Hv[0][0]=1.0;
	HTv[0][0]=1.0;
	Hi[0]=(int *)malloc(sizeof(int));
	Hi[0][0]=0;
	HTi[0][0]=0;
	cH[0]=1;
	cHT[0]=1;

	// Initiate computation of remaining (n-1) columns, rows and elements of factors G, H and D
	for(i=0;i<n-1;i++)
	{	
		// g_{i+1} = - G D^{-1} H b
        // t_b = H b
		for(j=Ui[i+1];j<Ui[i+2];j++)
		{
			jdx=Uj[j];
			for(k=0;k<cHT[jdx];k++)
			{
				kdx=HTi[jdx][k];
				if(itb[kdx]==-1)
				{
					tb[kdx]=Uv[j]*HTv[jdx][k];
					itb[kdx]=lastb;
					lastb=kdx;
					ltb++;
				}
				else
				{
					tb[kdx]+=Uv[j]*HTv[jdx][k];
				}
			}
		}

        // tt_b = G D^{-1} t_b
		tlast=lastb;
		for(j=0;j<ltb;j++)
		{
			
			jdx=tlast;
			for(k=0;k<cG[jdx];k++)
			{
				kdx=Gi[jdx][k];
				if(ittb[kdx]==-1)
				{
					ttb[kdx]=tb[jdx]*(*IDv)[jdx]*Gv[jdx][k];
					ittb[kdx]=lasttb;
					lasttb=kdx;
					lttb++;
				}
				else
				{
					ttb[kdx]+=tb[jdx]*(*IDv)[jdx]*Gv[jdx][k];
				}
			}
            // Update matrix retaining inf - norms of the columns of the U factor (A \approx LU)
			unrms[i+1]=mmax(unrms[i+1],fabs((*IDv)[jdx]*tb[jdx]));
			tlast=itb[jdx];
		}
		
		// h_{i+1} = - c G D^{-1} H
        // t_c = c G
		for(j=Li[i+1];j<Li[i+2];j++)
		{
			jdx=Lj[j];
			for(k=0;k<cGT[jdx];k++)
			{
				kdx=GTi[jdx][k];
				if(itc[kdx]==-1)
				{
					tc[kdx]=Lv[j]*GTv[jdx][k];
					itc[kdx]=lastc;
					lastc=kdx;
					ltc++;
				}
				else
				{
					tc[kdx]+=Lv[j]*GTv[jdx][k];
				}
			}
		}
		
        // tt_c = t_c D^{-1} H
		tlast=lastc;
		for(j=0;j<ltc;j++)
		{
			jdx=tlast;
			for(k=0;k<cH[jdx];k++)
			{
				kdx=Hi[jdx][k];
				if(ittc[kdx]==-1)
				{
					ttc[kdx]=(*IDv)[jdx]*tc[jdx]*Hv[jdx][k];
					ittc[kdx]=lasttc;
					lasttc=kdx;
					lttc++;
				}
				else
				{
					ttc[kdx]+=(*IDv)[jdx]*tc[jdx]*Hv[jdx][k];
				}
			}
            // Update matrix retaining inf - norms of the rows of the L factor (A \approx LU)
			lnrms[i+1]=mmax(lnrms[i+1],fabs(tc[jdx]*(*IDv)[jdx]));
			tlast=itc[jdx];
		}
			
		// Compute Schur Complement
        // First term s = d + c G D^{-1} H b
		d=Dv[i+1];
		sum=0.0;
		for(j=0;j<ltb;j++)
		{
			jdx=lastb;
			sum+=tb[jdx]*(*IDv)[jdx]*tc[jdx];			
			lastb=itb[jdx];
			itb[jdx]=-1;
			tb[jdx]=0.0;
		}
		ltb=0;

		for(j=0;j<ltc;j++)
		{
			jdx=lastc;
			lastc=itc[jdx];
			itc[jdx]=-1;
			tc[jdx]=0.0;
		}
		ltc=0;

		// Accumulate part of the Schur complement
		s=d+sum;

		// Compute correction of Schur complement
        // Second term s = s + c g_{i+1}
		s1=0.0;
		for(j=Li[i+1];j<Li[i+2];j++)
		{
			jdx=Lj[j];
			s1-=Lv[j]*ttb[jdx];
		}

		// Filter and store row and column to lower and upper triangular factor G
        // Filter g_{i+1}
		cnt=0;
		tlast=lasttb;
		for(j=0;j<lttb;j++)
		{
			jdx=tlast;
			// Update norms for the G factor
			gnrms[jdx]=mmax(gnrms[jdx],fabs(ttb[jdx]));
			if(fabs(ttb[jdx])<=dtol*gnrms[jdx]/mmin(unrms[i+1],alrnrms[i+1]))
			{
				ttb[jdx]=0.0;
				cnt++;
			}
			tlast=ittb[jdx];
		}
		
        // Store newly formed column g_{i+1} to CSC structure
        // Allocate adequate space
		Gv[i+1]=(double *)malloc((lttb-cnt+1) * sizeof(double));
		Gi[i+1]=(int *)malloc((lttb-cnt+1) * sizeof(int));
		cG[i+1]=lttb-cnt+1;
		cnt=0;
		for(j=0;j<lttb;j++)
		{
			jdx=lasttb;
			if(ttb[jdx]!=0.0)
			{

				Gv[i+1][cnt]=-ttb[jdx];
				Gi[i+1][cnt]=jdx;
				cnt++;
			}
			lasttb=ittb[jdx];
			ittb[jdx]=-1;
			ttb[jdx]=0.0;			
		}
		lttb=0;
        // Diagonal Element
		Gv[i+1][cnt]=1.0;
		Gi[i+1][cnt]=i+1;

		// Compute correction of the Schur Complement
        // Third part s = s + h_{i+1} b
		s2=0.0;
		for(j=Ui[i+1];j<Ui[i+2];j++)
		{
			jdx=Uj[j];
			s2-=Uv[j]*ttc[jdx];
		}

        // Filter h_{i+1}
		cnt=0;
		tlast=lasttc;		
		for(j=0;j<lttc;j++)
		{
			jdx=tlast;
			// Update norms for the H factor
			hnrms[jdx]=mmax(hnrms[jdx],fabs(ttc[jdx]));	
			if(fabs(ttc[jdx])<=dtol*hnrms[jdx]/mmin(lnrms[i+1],aucnrms[i+1]))
			{
				ttc[jdx]=0.0;
				cnt++;
			}
			tlast=ittc[jdx];
		}

        // Store newly formed row h_{i+1} to CSR structure
        // Allocate adequate space
		Hv[i+1]=(double *)malloc((lttc-cnt+1) * sizeof(double));
		Hi[i+1]=(int *)malloc((lttc-cnt+1) * sizeof(int));
		cH[i+1]=lttc-cnt+1;
		cnt=0;
		for(j=0;j<lttc;j++)
		{
			jdx=lasttc;
			if(ttc[jdx]!=0.0)
			{

				Hv[i+1][cnt]=-ttc[jdx];
				Hi[i+1][cnt]=jdx;
				cnt++;
			}
			lasttc=ittc[jdx];
			ittc[jdx]=-1;
			ttc[jdx]=0.0;			
		}
		lttc=0;
        // Diagonal Element
		Hv[i+1][cnt]=1.0;
		Hi[i+1][cnt]=i+1;
		
		// Populate transpose G and H
        // G^T
		for(j=0;j<cG[i+1];j++)
		{
			jdx=Gi[i+1][j];
            // Extend allocation by growth elements if maximum is reached
			if(cGT[jdx]+1>=mGT[jdx])
			{
				double *tb=(double *)malloc((mGT[jdx]+growth) * sizeof(double));
				int *tbi=(int *)malloc((mGT[jdx]+growth) * sizeof(int));
				mGT[jdx]+=growth;
				for(k=0;k<cGT[jdx];k++)
					tb[k]=GTv[jdx][k];
				for(k=0;k<cGT[jdx];k++)
					tbi[k]=GTi[jdx][k];
				free(GTv[jdx]);
				free(GTi[jdx]);
				GTv[jdx]=tb;
				GTi[jdx]=tbi;
				tb=NULL;
				tbi=NULL;
				
			}
			kdx=cGT[jdx];
			GTv[jdx][kdx]=Gv[i+1][j];
			GTi[jdx][kdx]=i+1;
			cGT[jdx]++;
		}
		
        // H^T
		for(j=0;j<cH[i+1];j++)
		{
			jdx=Hi[i+1][j];
            // Extend allocation by growth elements if maximum is reached
			if(cHT[jdx]+1>=mHT[jdx])
			{
				double *tc=(double *)malloc((mHT[jdx]+growth) * sizeof(double));
				int *tci=(int *)malloc((mHT[jdx]+growth) * sizeof(int));
				mHT[jdx]+=growth;
				for(k=0;k<cHT[jdx];k++)
					tc[k]=HTv[jdx][k];
				for(k=0;k<cHT[jdx];k++)
					tci[k]=HTi[jdx][k];
				free(HTv[jdx]);
				free(HTi[jdx]);
				HTv[jdx]=tc;
				HTi[jdx]=tci;
				tc=NULL;
				tci=NULL;
			}
			kdx=cHT[jdx];
			HTv[jdx][kdx]=Hv[i+1][j];
			HTi[jdx][kdx]=i+1;
			cHT[jdx]++;
		}
		
		//Check and populate matrix IDv
		s+=(s1+s2) + shift * fabs(d);
		if (fabs(s)<eta) s=(1e-4+dtol);
		(*IDv)[i+1]=1./s;
	
	}

	// Clean unnecessary matrices
	if (Unnz>0)
	{
		free(Uv);
		free(Uj);
	}
	free(Ui);

	if (Lnnz>0)
	{
		free(Lv);
		free(Lj);
	}
	free(Li);

	free(Dv);

	free(cG);
	free(cH);
	for(i=0;i<n;i++)
	{
		free(Gv[i]);
		free(Gi[i]);
		free(Hv[i]);
		free(Hi[i]);
	}
	free(Gv);
	free(Gi);
	free(Hv);
	free(Hi);

	free(gnrms);
	free(hnrms);

	free(unrms);
	free(lnrms);

	free(alrnrms);
	free(aucnrms);

	// Extract to CSR arrays
	*Gindi=(int *)malloc((n+1) * sizeof(int));
	*Hindi=(int *)malloc((n+1) * sizeof(int));

    // Compute offsets for CSR retaining G
	(*Gindi)[0]=0;
	for(i=0;i<n;i++)
		(*Gindi)[i+1]=(*Gindi)[i]+cGT[i];	
	Gnnz=(*Gindi)[n];
	
    // Compute offsets for CSR retaining H
	for(i=0;i<n+1;i++)
		(*Hindi)[i]=0;
	for(i=0;i<n;i++)
	{
		for(j=0;j<cHT[i];j++)
		{
			jdx=HTi[i][j];
			(*Hindi)[jdx+1]++;
		}
	}
	for(i=0;i<n;i++)
		(*Hindi)[i+1]+=(*Hindi)[i];
	Hnnz=(*Hindi)[n];

    // Allocate vectors retaining values and column indices
	(*Gval) =(double *)malloc(Gnnz * sizeof(double));
    (*Gindj)=(int *)malloc(Gnnz * sizeof(int));
	(*Hval) =(double *)malloc(Hnnz * sizeof(double));
	(*Hindj)=(int *)malloc(Hnnz * sizeof(int));

	// Populate G
	Gnnz=0;
	for(i=0;i<n;i++)
	{
		for(j=0;j<cGT[i];j++)
		{
			(*Gval)[Gnnz]=GTv[i][j];
			(*Gindj)[Gnnz]=GTi[i][j];
			Gnnz++;
		}
	}

	// Populate H
	for(i=0;i<n;i++)
	{
		for(j=0;j<cHT[i];j++)
		{
			jdx=HTi[i][j];
			(*Hindj)[(*Hindi)[jdx]]=i;
			(*Hval)[(*Hindi)[jdx]]=HTv[i][j];
			(*Hindi)[jdx]++;
		}
	}

	// Reset offsets
	for(i=n;i>=1;i--)
		(*Hindi)[i]=(*Hindi)[i-1];
	(*Hindi)[0]=0;

	// Clean up
	free(cGT);
	free(cHT);
	free(mHT);
	free(mGT);
	for(i=0;i<n;i++)
	{
		free(GTv[i]);
		free(GTi[i]);
		free(HTv[i]);
		free(HTi[i]);
	}
	free(GTv);
	free(GTi);
	free(HTv);
	free(HTi);

	free(tc);
	free(ttc);
	free(tb);
	free(ttb);
	free(itc);
	free(ittc);
	free(itb);
	free(ittb);
}
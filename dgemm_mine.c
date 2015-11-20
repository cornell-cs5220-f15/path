const char* dgemm_desc = "My awesome matmul.";

#include <stdlib.h>

#ifndef L1_BS
#define L1_BS ((int) 16)
#endif

#ifndef L2_BS
#define L2_BS ((int) 8)
#endif

// This code was written by Marc Aurele Gilles for the Matrix multiply hw for cs5220 at Cornell



/*

  A,B,C are M-by-M
  L1_BS is the size of sub matrix that will fit into L1
  L2_BS is the number of submatrix that will fit in L2
  L3*L2 is the number of submatrix that will fit in L3

*/
void row_to_block(const int M, const int padsize,  const int nblock, const double *restrict A, double *restrict newA)
{
	// converts to block indexing and pads the new matrix with zeros so that it is divisble by L1_BS
	int bi,bj,i,j;	
	int inf = M +1;
	for(bi=0; bi < nblock; ++bi){
		for(bj=0; bj < nblock; ++bj){
			for(i=0; i < L1_BS; ++i){
				for(j=0; j < L1_BS; ++j){
					if ((bj*L1_BS+j) >= M || bi*L1_BS+i >= M){ 
						// we can optimize this to delete this "if". this is doing the padding 
						newA[((bj*nblock+bi)*L1_BS*L1_BS+ i*L1_BS+j)]=inf;
					}
					else{
						newA[(bj*nblock+bi)*L1_BS*L1_BS+ i*L1_BS+j]=A[(j+bj*L1_BS)*M+bi*L1_BS+i]; 
					}				
				}
			}
		}
	}
}








void block_to_row(const int M, const int nblock, double *restrict A, const double *restrict newA)
{
	int bi, bj,i,j;	
	for(bi=0; bi < nblock; ++bi){
		for(bj=0; bj < nblock; ++bj){
			for(i=0; i < L1_BS; ++i){
				for(j=0; j < L1_BS; ++j){
					if ((bj*L1_BS+j)>= M || bi*L1_BS+i >= M){
					   	// we can optimize this to delete this "if". this is doing the padding 
					}
					else{
						A[(j+bj*L1_BS)*M+bi*L1_BS+i]=newA[(bj*nblock+bi)*L1_BS*L1_BS+ i*L1_BS+j];
				   	}

				}
			}
		}
	}
}	
void row_to_block_transpose(const int M, const int nblock, const double *restrict A, double *restrict newA)
{
	// converts to block indexing and pads the new matrix with zeros so that it is divisble by L1_BS
	int bi,bj,i,j;	
	for(bi=0; bi < nblock; ++bi){
		for(bj=0; bj < nblock; ++bj){
			for(i=0; i < L1_BS; ++i){
				for(j=0; j < L1_BS; ++j){
					if ((bj*L1_BS+j) >= M || bi*L1_BS+i >= M){ 
						// we can optimize this to delete this "if". this is doing the padding 
						newA[((bj*nblock+bi)*L1_BS*L1_BS+ j*L1_BS+i)]=0;
					}
					else{
						newA[(bj*nblock+bi)*L1_BS*L1_BS+ j*L1_BS+i]=A[(j+bj*L1_BS)*M+bi*L1_BS+i]; 
					}				
				}
			}
		}
	}
}







	
void do_block(const int M, const int nblock,
              const double * restrict A, double * restrict C,
              const int bi, const int bj, const int bk)
{	// A is old matrix, C is new matrix
	int i, j, k, BA_A, BA_B, sub_BA_A, sub_BA_B, BA_C, sub_BA_C;
    __assume_aligned(A, 64);
    __assume_aligned(B, 64);
    __assume_aligned(C, 64);

	// BA stands for block adress 
    BA_Ar=(bk*nblock+bi)*L1_BS*L1_BS; // for the column block of old matrix
    BA_Ac=(bj*nblock+bk)*L1_BS*L1_BS; // for the row block of old matrix
    BA_C=(bj*nblock+bi)*L1_BS*L1_BS;  // for the new matrix
    for (i = 0; i < L1_BS; ++i) {
	// finds sub_BA, tells compiler its aligned
	sub_BA_Ar=BA_Ar+L1_BS*i;
	__assume(sub_BA_A%8==0);	
	//same for C                
    sub_BA_C=BA_C+L1_BS*i;
    __assume(sub_BA_C%8==0);

	for (j = 0; j < L1_BS; ++j){
	    sub_BA_Ac=BA_Ac+L1_BS*j;
        __assume(sub_BA_Ac%8==0); // need to change factor since int are smaller than float

	    int cij = C[sub_BA_C+j];
            for (k = 0; k < L1_BS; ++k) {
		// new kernel using block to rogitw transpose	
                if (A[sub_BA_A+k] + B[sub_BA_B+k] < cij){
					cij = A[sub_BA_A+k] + B[sub_BA_B+k];
				}
       		//without transpose: 
       		//cij += A[((bk*nblock+bi))*L1_BS*L1_BS+L1_BS*i+k] * B[((bj*nblock)+bk)*L1_BS*L1_BS+L1_BS*k+j];
		}
	 C[sub_BA_C+j]= cij;
        }
    }
}


void square_dgemm(const int M, const double *A, const double *B, double *C)
{
	/* pad size : size of matrix after padding
	   block i= row index of block
	   block j = column index of block
	   nblock = number of blocks
	   L2bi, L2bj index of L2
           L2nblock= number of L1block that fit into L2
	   L2rem number of remaining blocks 
           rem : remainder of blocks after l2 blocking
	   Ai, Aj, Ak : "addresses of i j k loops"
	*/

	int pad_size, bi, bj, bk, L2bi, L2bj, L2bk, nblock, L2nblock, rem;

	if (M%L1_BS==0){
		nblock=M/L1_BS;
		pad_size=M;
	}
	else{ 
		pad_size=((M/L1_BS)+1)*L1_BS;
		nblock=M/L1_BS+1;
	}
	
	// number of L2

	if(pad_size%(L2_BS*L1_BS)==0){
	L2nblock=pad_size/(L2_BS*L1_BS);
	// define remainder to be a whole block in this case for consistency
	 rem = L2_BS;}

	else{L2nblock=pad_size/(L2_BS*L1_BS)+1;
	     rem= (pad_size%(L2_BS*L1_BS))/L1_BS;	}

	
	double *restrict bA= (double*) _mm_malloc(pad_size*pad_size*sizeof(double),64);
	double *restrict bB= (double*) _mm_malloc(pad_size*pad_size*sizeof(double),64);
	double *restrict bC= (double*) _mm_malloc(pad_size*pad_size*sizeof(double),64);
	

	// change indexing
	row_to_block(M,nblock, A, bA);
	row_to_block_transpose(M,nblock, B, bB);
	row_to_block(M,nblock, C, bC);





		// MAIN LOOP
	for (L2bk=0; L2bk < L2nblock-1; ++L2bk){  
	for (L2bj=0; L2bj < L2nblock-1; ++L2bj){
	for (L2bi=0; L2bi < L2nblock-1; ++L2bi){
	for (bk = 0; bk < L2_BS; ++bk) {
		int Ak=L2bk*L2_BS +bk;
		for (bj = 0; bj < L2_BS; ++bj) {
			int Aj=L2bj*L2_BS+bj;
			for (bi = 0; bi < L2_BS; ++bi) {
				int Ai=L2bi*L2_BS+bi;	
	
				do_block(M, nblock, bA, bB, bC, Ai, Aj, Ak);}
			}
		}
	}
	}
	}


	

	// ADDITIONAL LOOPS TO AVOID IF STATEMENTS
	// there are 8 cases: (we denote j not at boundary by j0 and j at boundary with j1
	// 1: k0 j0 i0 (main loop)
	// 2: k1 j0 i0
	// 3: k0 j1 i0
	// 4: k0 j0 i1
	// 5: k1 j1 i0
	// 6: k1 j0 i1
	// 7: k0 j0 i1
	// 8: k1 j1 i1
	//printf("we got here\n");
	

	// case 2 k1 j0 i0
	L2bk=L2nblock-1; 
	for (L2bj=0; L2bj < L2nblock-1; ++L2bj){
	
	for (L2bi=0; L2bi < L2nblock-1; ++L2bi){
	for (bk = 0; bk < rem; ++bk) {
		int Ak=L2bk*L2_BS+bk;
		for (bj = 0; bj < L2_BS; ++bj) {
			int Aj=L2bj*L2_BS+bj;
			for (bi = 0; bi < L2_BS; ++bi) {
				int Ai=L2bi*L2_BS+bi;	
				do_block(M, nblock, bA, bB, bC, Ai, Aj, Ak);
           
			
		}
	}
	}
	}
	}
	// case 3: k0 j1 i0
	L2bj=L2nblock-1; 
	for (L2bk=0; L2bk < L2nblock-1; ++L2bk){
	for (L2bi=0; L2bi < L2nblock-1; ++L2bi){
	for (bk = 0; bk < L2_BS; ++bk) {
		int Ak=L2bk*L2_BS+bk;
		for (bj = 0; bj < rem; ++bj) {
			int Aj=L2bj*L2_BS+bj;
			for (bi = 0; bi < L2_BS; ++bi) {
				int Ai=L2bi*L2_BS+bi;	
				do_block(M, nblock, bA, bB, bC, Ai, Aj, Ak);
                                
			
		}
	}
	}
	}
	}
	// case 4: k0 j0 i1
	L2bi=L2nblock-1; 
	for (L2bk=0; L2bk < L2nblock-1; ++L2bk){
	for (L2bj=0; L2bj < L2nblock-1; ++L2bj){
	for (bk = 0; bk < L2_BS; ++bk) {
		int Ak=L2bk*L2_BS+bk;
		for (bj = 0; bj < L2_BS; ++bj) {
			int Aj=L2bj*L2_BS+bj;
			for (bi = 0; bi < rem; ++bi) {
				int Ai=L2bi*L2_BS+bi;	
				do_block(M, nblock, bA, bB, bC, Ai, Aj, Ak);
                               
			
		}
	}
	}
	}
	}



	// case 5: k1 j1 i0
	L2bj=L2nblock-1;
	L2bk=L2nblock-1;
	for (L2bi=0; L2bi < L2nblock-1; ++L2bi){
	for (bk = 0; bk < rem; ++bk) {
		int Ak=L2bk*L2_BS +bk;
		for (bj = 0; bj < rem; ++bj) {
			int Aj=L2bj*L2_BS+bj;
			for (bi = 0; bi < L2_BS; ++bi) {

				int Ai=L2bi*L2_BS+bi;	
                       
				do_block(M, nblock, bA, bB, bC, Ai, Aj, Ak);
			}		
		}
	}
	}
	
	// case 6: k1 j0 i1
	L2bi=L2nblock-1;
	L2bk=L2nblock-1;
	for (L2bj=0; L2bj < L2nblock-1; ++L2bj){
	for (bk = 0; bk < rem; ++bk) {
		int Ak=L2bk*L2_BS +bk;
		for (bj = 0; bj < L2_BS; ++bj) {
			int Aj=L2bj*L2_BS+bj;
			for (bi = 0; bi < rem; ++bi) {

				int Ai=L2bi*L2_BS+bi;	
                       
				do_block(M, nblock, bA, bB, bC, Ai, Aj, Ak);
			}		
		}
	}
	}
	// case 7: k0 j1 i1
	L2bj=L2nblock-1;
	L2bi=L2nblock-1;
	for (L2bk=0; L2bk < L2nblock-1; ++L2bk){
	for (bk = 0; bk < L2_BS; ++bk) {
		int Ak=L2bk*L2_BS +bk;
		for (bj = 0; bj < rem; ++bj) {
			int Aj=L2bj*L2_BS+bj;
			for (bi = 0; bi < rem; ++bi) {

				int Ai=L2bi*L2_BS+bi;	
                       
				do_block(M, nblock, bA, bB, bC, Ai, Aj, Ak);
			}		
		}
	}
	}
	// case 8: k1 j1 i1
	L2bj=L2nblock-1;
	L2bj=L2nblock-1;
 	L2bi=L2nblock-1;
	for (bk = 0; bk < rem; ++bk) {
		int Ak=L2bk*L2_BS +bk;
		for (bj = 0; bj < rem; ++bj) {
			int Aj=L2bj*L2_BS+bj;
			for (bi = 0; bi < rem; ++bi) {
				int Ai=L2bi*L2_BS+bi;	

        
				do_block(M, nblock, bA, bB, bC, Ai, Aj, Ak);
			
		}
	}
	}
	
	
		
// reindex back to column
	block_to_row(M,nblock,C,bC);
}


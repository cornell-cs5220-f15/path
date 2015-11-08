#include "square.h"
#include <math.h>
#include <omp.h>

int square(int n, int* restrict l, int* restrict lnew){
int square(int n,               // Number of nodes
               int* restrict l,     // Partial distance at step s
                          int* restrict lnew)  // Partial distance at step s+1
{
      // Make a copy of transposed matrix to speed up vectorization
      int* restrict l_transpose = (int*) _mm_malloc(n*n*sizeof(int),16);
      for( int i = 0; i < n; i++){
          for( int j = 0; j < n; j++){
              l_transpose[j*n+i] = l[i*n+j];
          }
      }
      
      int done = 1;
      //#pragma omp parallel for shared(l, lnew) reduction(&& : done)
      #pragma omp parallel for shared(l, lnew, l_transpose) reduction(&& : done)
      for (int j = 0; j < n; ++j) {
          for (int i = 0; i < n; ++i) {
              int lij = lnew[j*n+i];
              for (int k = 0; k < n; ++k) {
      // Do better memory access here
      // See if you can make an explicity copy before carrying out the computation
      //int lik = l[k*n+i];
                  int lik = l_transpose[i*n+k];
                  int lkj = l[j*n+k];
                  if (lik + lkj < lij) {
                      lij = lik+lkj;
                      done = 0;
                  }
              }
              lnew[j*n+i] = lij;
          }
      }
      _mm_free(l_transpose);
      return done;
}

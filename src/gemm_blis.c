/* 
   GEMM FLAVOURS

   -----

   GEMM FLAVOURS is a family of algorithms for matrix multiplication based
   on the BLIS approach for this operation: https://github.com/flame/blis

   -----

   This program is free software: you can redistribute it and/or modify it under
   the terms of the GNU General Public License as published by the Free Software
   Foundation, either version 3 of the License, or (at your option) any later
   version.

   This program is distributed in the hope that it will be useful, but WITHOUT
   ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
   FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
   You should have received a copy of the GNU General Public License along with
   this program. If not, see <http://www.gnu.org/licenses/>.

   -----

   author    = "Enrique S. Quintana-Orti"
   contact   = "quintana@disca.upv.es"
   copyright = "Copyright 2021, Universitat Politecnica de Valencia"
   license   = "GPLv3"
   status    = "Production"
   version   = "1.1"
*/

#include "gemm_blis.h"

/*
void gemm_blis_B3A2C0( char orderA, char orderB, char orderC,
                       char transA, char transB, 
                       size_t m, size_t n, size_t k, 
                       float alpha, float *A, size_t ldA, 
		                    float *B, size_t ldB, 
		       float beta,  float *C, size_t ldC, 
		       float *Ac, float *Bc, 
                       size_t MC, size_t NC, size_t KC,
		       ukernel_SIMD ukernel, int MR, int NR) {


  size_t    ic, jc, pc, mc, nc, kc, ir, jr, mr, nr; 
  float  zero = 0.0, one = 1.0, betaI; 
  float  *Aptr, *Bptr, *Cptr;

  #if defined(CHECK)
  #include "check_params.h"
  #endif
   
  // Quick return if possible
  if ( (m==0)||(n==0)||(((alpha==zero)||(k==0))&&(beta==one)) )
    return;
  
  #include "quick_gemm.h"
  
  for ( jc=0; jc<n; jc+=NC ) {
    nc = min(n-jc, NC); 

    for ( pc=0; pc<k; pc+=KC ) {
      kc = min(k-pc, KC); 
      
      if ( (transB=='N')&&(orderB=='C') )
        Bptr = &Bcol(pc,jc);
      else if ( (transB=='N')&&(orderB=='R') )
        Bptr = &Brow(pc,jc);
      else if ( (transB=='T')&&(orderB=='C') )
        Bptr = &Bcol(jc,pc);
      else
        Bptr = &Brow(jc,pc);
      
      pack_CB( orderB, transB, kc, nc, Bptr, ldB, Bc, NR);
      
      if ( pc==0 )
        betaI = beta;
      else
        betaI = one;
      
      for ( ic=0; ic<m; ic+=MC ) {
        mc = min(m-ic, MC); 
	
        if ( (transA=='N')&&(orderA=='C') ){
          Aptr = &Acol(ic, pc);
	}else if ( (transA=='N')&&(orderA=='R') ){
          Aptr = &Arow(ic, pc);
	}else if ( (transA=='T')&&(orderA=='C') ){
          Aptr = &Acol(pc, ic);
	}else{
          Aptr = &Arow(pc, ic);
	}
	
	//Comment or uncomment for packing or not
        pack_RB( orderA, transA, mc, kc, Aptr, ldA, Ac, MR);
	
        for ( jr=0; jr<nc; jr+=NR ) {
          nr = min(nc-jr, NR); 
	  
          for ( ir=0; ir<mc; ir+=MR ) {
            mr = min(mc-ir, MR); 
	    
            if ( orderC=='C' )
              Cptr = &Ccol(ic+ir,jc+jr);
	    else
              Cptr = &Crow(ic+ir,jc+jr);

	    ukernel(mr, nr, kc, &Ac[ir*kc], ldA, &Bc[jr*kc], ldB, Cptr, ldC, orderC, betaI);

          }
	  
        }
      }
    }
  }

}
*/

void sgemm_family(char *, char *, void *_m, void *_n, void *_k, float *_alpha, float *A, void *_ldA, float *B,
		  void *_ldB, float *_beta, float *C, void *_ldC) {

  char orderA = 'C';
  char orderB = 'C';
  char orderC = 'C';
  char transA = 'N';
  char transB = 'N';

  size_t m = *((size_t *)_m);
  size_t n = *((size_t *)_n);
  size_t k = *((size_t *)_k);

  size_t ldA = *((size_t *)_ldA);
  size_t ldB = *((size_t *)_ldB);
  size_t ldC = *((size_t *)_ldC);
 
  float beta  = (float)(*_beta); 
  float alpha = (float)(*_alpha); 

  int MR = 8;
  int NR = 12;
    
  //------------------------------------------------------------------------
  // MODEL VALUES: MC, NC and KC
  //------------------------------------------------------------------------
  int mc_tmp, nc_tmp, kc_tmp;
  get_optim_mc_nc_kc(sizeof(float), m, n, k, MR, NR, &mc_tmp, &nc_tmp, &kc_tmp);
   
  mc_tmp = mc_tmp / MR * MR;
  nc_tmp = nc_tmp / NR * NR;
     
  size_t MC = (size_t)mc_tmp;
  size_t NC = (size_t)nc_tmp;
  size_t KC = (size_t)kc_tmp;
	   
  //-------------------------------------------------------------------------

  float *Ac = (float *)malloc((MR+MC)*KC * sizeof(float));
  float *Bc = (float *)malloc(KC*(NR+NC) * sizeof(float));

  int mr, nr;
  size_t ic, jc, pc,  ir, jr, mc, nc, kc; 
  float zero = 0.0, one = 1.0, betaI; 
  float *Aptr, *Bptr, *Cptr;

  // Quick return if possible
  if ( (m==0)||(n==0)||(((alpha==zero)||(k==0))&&(beta==one)) )
    return;
  
  for ( jc=0; jc<n; jc+=NC ) {
    nc = min(n-jc, NC); 

    for ( pc=0; pc<k; pc+=KC ) {
      kc = min(k-pc, KC); 
      
      if ( (transB=='N')&&(orderB=='C') )
        Bptr = &Bcol(pc,jc);
      else if ( (transB=='N')&&(orderB=='R') )
        Bptr = &Brow(pc,jc);
      else if ( (transB=='T')&&(orderB=='C') )
        Bptr = &Bcol(jc,pc);
      else
        Bptr = &Brow(jc,pc);
      
      pack_CB( orderB, transB, kc, nc, Bptr, ldB, Bc, NR);

      if ( pc==0 )
        betaI = beta;
      else
        betaI = one;
      
      for ( ic=0; ic<m; ic+=MC ) {
        mc = min(m-ic, MC); 
	
        if ( (transA=='N')&&(orderA=='C') ){
          Aptr = &Acol(ic, pc);
	}else if ( (transA=='N')&&(orderA=='R') ){
          Aptr = &Arow(ic, pc);
	}else if ( (transA=='T')&&(orderA=='C') ){
          Aptr = &Acol(pc, ic);
	}else{
          Aptr = &Arow(pc, ic);
	}
	
	//Comment or uncomment for packing or not
        pack_RB( orderA, transA, mc, kc, Aptr, ldA, Ac, MR);
	
        for ( jr=0; jr<nc; jr+=NR ) {
          nr = min(nc-jr, NR); 
	  
          for ( ir=0; ir<mc; ir+=MR ) {
            mr = min(mc-ir, MR); 
	    
            if ( orderC=='C' )
              Cptr = &Ccol(ic+ir,jc+jr);
	    else
              Cptr = &Crow(ic+ir,jc+jr);

	    //ukernel(mr, nr, kc, &Ac[ir*kc], ldA, &Bc[jr*kc], ldB, Cptr, ldC, orderC, betaI);
	    gemm_base_Cresident( orderC, mr, nr, kc, alpha, &Ac[ir*kc], MR, 
                               &Bc[jr*kc], NR, betaI, Cptr, ldC );

          }
	  
        }
      }
    }
  }

  free(Ac);
  free(Bc);

}


void pack_RB( char orderM, char transM, int mc, int nc, float *M, int ldM, float *Mc, int RR ){
/*
  BLIS pack for M-->Mc
*/
  int    i, j, ii, k, rr;

  if ( ((transM=='N')&&( orderM=='C'))||
       ((transM=='T')&&( orderM=='R')) ) {
    for ( i=0; i<mc; i+=RR ) { 
      k = i*nc;
      rr = min( mc-i, RR );
      
      for ( j=0; j<nc; j++ ) {
        for ( ii=0; ii<rr; ii++ ) {
	  Mc[k] = Mcol(i+ii, j);
          k++;
        } 
        k += (RR-rr);
      }
    }
  } else {
    for ( i=0; i<mc; i+=RR ) { 
      k = i*nc;
      rr = min( mc-i, RR );
      for ( j=0; j<nc; j++ ) {
        for ( ii=0; ii<rr; ii++ ) {
           Mc[k] = Mcol(j,i+ii);
          k++;
        }
        k += (RR-rr);
      }
    }
  }
}

void pack_CB( char orderM, char transM, int mc, int nc, float *M, int ldM, float *Mc, int RR ){
/*
  BLIS pack for M-->Mc
*/
  int    i, j, jj, k, nr;

  k = 0;
  if ( ((transM=='N')&&( orderM=='C'))||
       ((transM=='T')&&( orderM=='R')) )
    for ( j=0; j<nc; j+=RR ) { 
      k = j*mc;
      nr = min( nc-j, RR );
      for ( i=0; i<mc; i++ ) {
        for ( jj=0; jj<nr; jj++ ) {
          Mc[k] = Mcol(i,j+jj);
          k++;
        }
        k += (RR-nr);
      }
    }
  else
    for ( j=0; j<nc; j+=RR ) { 
      k = j*mc;
      nr = min( nc-j, RR );
      for ( i=0; i<mc; i++ ) {
        for ( jj=0; jj<nr; jj++ ) {
          Mc[k] = Mcol(j+jj,i);
          k++;
        }
        k += (RR-nr);
      }
    }
}

void gemm_base_Cresident( char orderC, int m, int n, int k, 
                          float alpha, float *A, int ldA, 
                                       float *B, int ldB, 
                          float beta,  float *C, int ldC ){
/*
  Baseline micro-kernel 
  Replace with specialized micro-kernel where C-->m x n is resident in registers
*/
  int    i, j, p;
  float  zero = 0.0, tmp;

  for ( j=0; j<n; j++ )
    for ( i=0; i<m; i++ ) {
      tmp = 0.0; 
      for ( p=0; p<k; p++ ) 
        tmp += Acol(i,p) * Brow(p,j);

      if ( beta==zero ) {
        if ( orderC=='C' )
          Ccol(i,j) = alpha*tmp;
        else
          Crow(i,j) = alpha*tmp;
      }
      else {
        if ( orderC=='C' )
          Ccol(i,j) = alpha*tmp + beta*Ccol(i,j);
        else
          Crow(i,j) = alpha*tmp + beta*Crow(i,j);
      }
    }
}



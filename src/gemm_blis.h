#include <stdio.h>
#include <stdlib.h>

#include "dtypes.h"
#include "modelLevel/model_level.h"

//#include <arm_neon.h>
//#include "ARMv8/microkernel.h"

#ifndef min
  #define min(a,b) (((a)<(b))?(a):(b))
#endif

#ifndef max
  #define max(a,b) (((a)>(b))?(a):(b))
#endif

#define Acol(a1,a2)  A[ (a2)*(ldA)+(a1) ]
#define Bcol(a1,a2)  B[ (a2)*(ldB)+(a1) ]
#define Ccol(a1,a2)  C[ (a2)*(ldC)+(a1) ]
#define Mcol(a1,a2)  M[ (a2)*(ldM)+(a1) ]

#define Arow(a1,a2)  A[ (a1)*(ldA)+(a2) ]
#define Brow(a1,a2)  B[ (a1)*(ldB)+(a2) ]
#define Crow(a1,a2)  C[ (a1)*(ldC)+(a2) ]
#define Mrow(a1,a2)  M[ (a1)*(ldM)+(a2) ]

void sgemm_family(char *, char *, void *, void *, void *, float *, float *, void *, float *,
		  void *, float *, float *, void *);

//void gemm_blis_B3A2C0( char, char, char, char, char, size_t, size_t, size_t, float, float *, size_t, float *, size_t, float, float *, size_t, 
//		       float *, float *, size_t, size_t, size_t, ukernel_SIMD, int, int);

void gemm_base_Cresident( char, int, int, int, float, float *, int, float *, int, float, float *, int );

void pack_RB( char, char, int, int, float *, int, float *, int );
void pack_CB( char, char, int, int, float *, int, float *, int );



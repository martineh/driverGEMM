#include <stdio.h>
#include <stdlib.h>

#include "dtypes.h"

#include <arm_neon.h>
#include "ARMv8/FP32/microkernel.h"

#define min(a,b) (((a)<(b))?(a):(b))
#define max(a,b) (((a)>(b))?(a):(b))

#define Acol(a1,a2)  A[ (a2)*(ldA)+(a1) ]
#define Bcol(a1,a2)  B[ (a2)*(ldB)+(a1) ]
#define Ccol(a1,a2)  C[ (a2)*(ldC)+(a1) ]
#define Mcol(a1,a2)  M[ (a2)*(ldM)+(a1) ]

#define Arow(a1,a2)  A[ (a1)*(ldA)+(a2) ]
#define Brow(a1,a2)  B[ (a1)*(ldB)+(a2) ]
#define Crow(a1,a2)  C[ (a1)*(ldC)+(a2) ]
#define Mrow(a1,a2)  M[ (a1)*(ldM)+(a2) ]

void gemm_blis_B3A2C0( char, char, char, char, char, size_t, size_t, size_t, DTYPE, DTYPE *, size_t, DTYPE *, size_t, DTYPE, DTYPE *, size_t, 
		       DTYPE *, DTYPE *, size_t, size_t, size_t, ukernel_SIMD, int, int);

void gemm_base_Cresident( char, int, int, int, DTYPE, DTYPE *, int, DTYPE *, int, DTYPE, DTYPE *, int );

void pack_RB( char, char, int, int, DTYPE *, int, DTYPE *, int );
void pack_CB( char, char, int, int, DTYPE *, int, DTYPE *, int );



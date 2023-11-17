#include <arm_neon.h>
#include "../../dtypes.h" 

typedef void (*ukernel_SIMD)(int, int, int, DTYPE *, int, DTYPE *, int, DTYPE *, int, char, DTYPE);


void gemm_ukernel_Cresident_SIMD_2x2(int, int, int, DTYPE *, int, DTYPE *, int, DTYPE *, int, char, DTYPE); 
void gemm_ukernel_Cresident_SIMD_2x4(int, int, int, DTYPE *, int, DTYPE *, int, DTYPE *, int, char, DTYPE); 
void gemm_ukernel_Cresident_SIMD_2x6(int, int, int, DTYPE *, int, DTYPE *, int, DTYPE *, int, char, DTYPE); 
void gemm_ukernel_Cresident_SIMD_2x8(int, int, int, DTYPE *, int, DTYPE *, int, DTYPE *, int, char, DTYPE); 
void gemm_ukernel_Cresident_SIMD_2x10(int, int, int, DTYPE *, int, DTYPE *, int, DTYPE *, int, char, DTYPE); 
void gemm_ukernel_Cresident_SIMD_2x12(int, int, int, DTYPE *, int, DTYPE *, int, DTYPE *, int, char, DTYPE); 

void gemm_ukernel_Cresident_SIMD_4x2(int, int, int, DTYPE *, int, DTYPE *, int, DTYPE *, int, char, DTYPE); 
void gemm_ukernel_Cresident_SIMD_4x4(int, int, int, DTYPE *, int, DTYPE *, int, DTYPE *, int, char, DTYPE); 
void gemm_ukernel_Cresident_SIMD_4x6(int, int, int, DTYPE *, int, DTYPE *, int, DTYPE *, int, char, DTYPE); 
void gemm_ukernel_Cresident_SIMD_4x8(int, int, int, DTYPE *, int, DTYPE *, int, DTYPE *, int, char, DTYPE); 
void gemm_ukernel_Cresident_SIMD_4x10(int, int, int, DTYPE *, int, DTYPE *, int, DTYPE *, int, char, DTYPE); 
void gemm_ukernel_Cresident_SIMD_4x12(int, int, int, DTYPE *, int, DTYPE *, int, DTYPE *, int, char, DTYPE); 

void gemm_ukernel_Cresident_SIMD_6x2(int, int, int, DTYPE *, int, DTYPE *, int, DTYPE *, int, char, DTYPE); 
void gemm_ukernel_Cresident_SIMD_6x4(int, int, int, DTYPE *, int, DTYPE *, int, DTYPE *, int, char, DTYPE); 

void gemm_ukernel_Cresident_SIMD_8x2(int, int, int, DTYPE *, int, DTYPE *, int, DTYPE *, int, char, DTYPE); 
void gemm_ukernel_Cresident_SIMD_8x4(int, int, int, DTYPE *, int, DTYPE *, int, DTYPE *, int, char, DTYPE); 

void gemm_ukernel_Cresident_SIMD_10x2(int, int, int, DTYPE *, int, DTYPE *, int, DTYPE *, int, char, DTYPE); 
void gemm_ukernel_Cresident_SIMD_10x4(int, int, int, DTYPE *, int, DTYPE *, int, DTYPE *, int, char, DTYPE); 

void gemm_ukernel_Cresident_SIMD_12x2(int, int, int, DTYPE *, int, DTYPE *, int, DTYPE *, int, char, DTYPE); 
void gemm_ukernel_Cresident_SIMD_12x4(int, int, int, DTYPE *, int, DTYPE *, int, DTYPE *, int, char, DTYPE); 


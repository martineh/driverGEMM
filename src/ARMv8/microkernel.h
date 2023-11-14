#include <arm_neon.h>
#include "../dtypes.h" 

typedef void (*ukernel_SIMD)(int, int, int, DTYPE *, int, DTYPE *, int, DTYPE *, int, char, DTYPE);


void gemm_ukernel_Cresident_SIMD_4x4(int, int, int, DTYPE *, int, DTYPE *, int, DTYPE *, int, char, DTYPE); 
void gemm_ukernel_Cresident_SIMD_4x8(int, int, int, DTYPE *, int, DTYPE *, int, DTYPE *, int, char, DTYPE); 
void gemm_ukernel_Cresident_SIMD_4x12(int, int, int, DTYPE *, int, DTYPE *, int, DTYPE *, int, char, DTYPE); 
void gemm_ukernel_Cresident_SIMD_4x16(int, int, int, DTYPE *, int, DTYPE *, int, DTYPE *, int, char, DTYPE); 
void gemm_ukernel_Cresident_SIMD_4x20(int, int, int, DTYPE *, int, DTYPE *, int, DTYPE *, int, char, DTYPE); 
void gemm_ukernel_Cresident_SIMD_8x4(int, int, int, DTYPE *, int, DTYPE *, int, DTYPE *, int, char, DTYPE); 
void gemm_ukernel_Cresident_SIMD_8x8(int, int, int, DTYPE *, int, DTYPE *, int, DTYPE *, int, char, DTYPE); 
void gemm_ukernel_Cresident_SIMD_8x12(int, int, int, DTYPE *, int, DTYPE *, int, DTYPE *, int, char, DTYPE); 
void gemm_ukernel_Cresident_SIMD_12x4(int, int, int, DTYPE *, int, DTYPE *, int, DTYPE *, int, char, DTYPE); 
void gemm_ukernel_Cresident_SIMD_12x8(int, int, int, DTYPE *, int, DTYPE *, int, DTYPE *, int, char, DTYPE); 
void gemm_ukernel_Cresident_SIMD_16x4(int, int, int, DTYPE *, int, DTYPE *, int, DTYPE *, int, char, DTYPE); 
void gemm_ukernel_Cresident_SIMD_20x4(int, int, int, DTYPE *, int, DTYPE *, int, DTYPE *, int, char, DTYPE); 

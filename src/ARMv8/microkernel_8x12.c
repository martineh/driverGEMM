#include "microkernel.h"
// Macros for ARM NEON 
    #define vl_fp32 4 
    #define vregister float32x4_t 
    #define vload(vreg, mem)                vreg = vld1q_f32(mem) 
    #define vstore(mem, vreg)               vst1q_f32(mem, vreg) 
    #define vupdate(vreg1, vreg2, vreg3, j) vreg1 = vfmaq_laneq_f32(vreg1, vreg2, vreg3, j) 
    #define vinit(vreg)                     vreg = vmovq_n_f32(0) 
    #define vbroadcast(a1,a2)               a1 = vdupq_n_f32(a2) 
    #define Ccol(a1,a2)  C[(a2) * (ldC) + (a1)] 
    #define Crow(a1,a2)  C[(a1) * (ldC) + (a2)] 
    #define Ctcol(a1,a2) Ctmp[(a2) * (ldCt) + (a1)] 
    #define Ctrow(a1,a2) Ctmp[(a1) * (ldCt) + (a2)] 
    #define Ctref(a1,a2) Ctmp[(a2) * (ldCt) + (a1)] 
    #define FUNROLL 1
    #define LOOP_BODY_8x12 \
{\
    vload(ar0, Aptr + baseA + 0);\
    vload(ar1, Aptr + baseA + 4);\
    vload(br0, Bptr + baseB + 0);\
    vload(br1, Bptr + baseB + 4);\
    vload(br2, Bptr + baseB + 8);\
    vupdate(Cr0_0, ar0, br0, 0);\
    vupdate(Cr0_1, ar0, br0, 1);\
    vupdate(Cr0_2, ar0, br0, 2);\
    vupdate(Cr0_3, ar0, br0, 3);\
    vupdate(Cr0_4, ar0, br1, 0);\
    vupdate(Cr0_5, ar0, br1, 1);\
    vupdate(Cr0_6, ar0, br1, 2);\
    vupdate(Cr0_7, ar0, br1, 3);\
    vupdate(Cr0_8, ar0, br2, 0);\
    vupdate(Cr0_9, ar0, br2, 1);\
    vupdate(Cr0_10, ar0, br2, 2);\
    vupdate(Cr0_11, ar0, br2, 3);\
    vupdate(Cr1_0, ar1, br0, 0);\
    vupdate(Cr1_1, ar1, br0, 1);\
    vupdate(Cr1_2, ar1, br0, 2);\
    vupdate(Cr1_3, ar1, br0, 3);\
    vupdate(Cr1_4, ar1, br1, 0);\
    vupdate(Cr1_5, ar1, br1, 1);\
    vupdate(Cr1_6, ar1, br1, 2);\
    vupdate(Cr1_7, ar1, br1, 3);\
    vupdate(Cr1_8, ar1, br2, 0);\
    vupdate(Cr1_9, ar1, br2, 1);\
    vupdate(Cr1_10, ar1, br2, 2);\
    vupdate(Cr1_11, ar1, br2, 3);\
    baseA += Amr; baseB += Bnr;\
} 

inline void gemm_ukernel_Cresident_SIMD_8x12(int mr, int nr, int kc, DTYPE *Ar, int ldA, DTYPE *Br, int ldB, DTYPE *C, int ldC, char orderC, DTYPE beta) {
// mr x nr = 8 x 12 micro-kernel and C resident in regs.
// Inputs:
//   - C stored in column-major order, with leading dimension ldC
//   - Ar packed by columns, with leading dimension mr = 8
//   - Br packed by rows, with leading dimension nr = 12

  if (kc == 0) return;
  int       i, j, pr, baseA = 0, baseB = 0, ldCt = MR, Amr, Bnr;
  vregister Cr0_0, Cr0_1, Cr0_2, Cr0_3, Cr0_4, Cr0_5, Cr0_6, Cr0_7, Cr0_8, Cr0_9, Cr0_10, Cr0_11, 
            Cr1_0, Cr1_1, Cr1_2, Cr1_3, Cr1_4, Cr1_5, Cr1_6, Cr1_7, Cr1_8, Cr1_9, Cr1_10, Cr1_11, 
            A0_0, A0_1, A0_2, A0_3, A0_4, A0_5, A0_6, A0_7, A0_8, A0_9, A0_10, A0_11, 
            A1_0, A1_1, A1_2, A1_3, A1_4, A1_5, A1_6, A1_7, A1_8, A1_9, A1_10, A1_11;
  vregister ar0, ar1, br0, br1, br2; 
  DTYPE zero = 0.0, one = 1.0, Ctmp[MR * NR], *Aptr, *Bptr; 


  vinit(Cr0_0); vinit(Cr0_1); vinit(Cr0_2); vinit(Cr0_3); vinit(Cr0_4); vinit(Cr0_5); vinit(Cr0_6); vinit(Cr0_7); vinit(Cr0_8); vinit(Cr0_9); vinit(Cr0_10); vinit(Cr0_11); 
  vinit(Cr1_0); vinit(Cr1_1); vinit(Cr1_2); vinit(Cr1_3); vinit(Cr1_4); vinit(Cr1_5); vinit(Cr1_6); vinit(Cr1_7); vinit(Cr1_8); vinit(Cr1_9); vinit(Cr1_10); vinit(Cr1_11); 
  vinit(ar0); vinit(ar1); 
  vinit(br0); vinit(br1); vinit(br2); 
  if (orderC == 'C') { 
    Aptr = &Ar[0]; 
    Bptr = &Br[0]; 
    Amr  = MR; 
    Bnr  = NR; 
  } else { 
    Aptr = &Br[0]; 
    Bptr = &Ar[0]; 
    Amr  = NR; 
    Bnr  = MR; 
  } 
  int it = kc % FUNROLL;
    for (pr = 0; pr < it; pr++) { 
      LOOP_BODY_8x12; 
    } 
    for (pr = it; pr < kc; pr += FUNROLL) { 
    LOOP_BODY_8x12;
    } 
  if ((mr < MR) || (nr < NR)) {
  vstore(&Ctref(0, 0), Cr0_0); 
  vstore(&Ctref(0, 1), Cr0_1); 
  vstore(&Ctref(0, 2), Cr0_2); 
  vstore(&Ctref(0, 3), Cr0_3); 
  vstore(&Ctref(0, 4), Cr0_4); 
  vstore(&Ctref(0, 5), Cr0_5); 
  vstore(&Ctref(0, 6), Cr0_6); 
  vstore(&Ctref(0, 7), Cr0_7); 
  vstore(&Ctref(0, 8), Cr0_8); 
  vstore(&Ctref(0, 9), Cr0_9); 
  vstore(&Ctref(0, 10), Cr0_10); 
  vstore(&Ctref(0, 11), Cr0_11); 
  vstore(&Ctref(4, 0), Cr1_0); 
  vstore(&Ctref(4, 1), Cr1_1); 
  vstore(&Ctref(4, 2), Cr1_2); 
  vstore(&Ctref(4, 3), Cr1_3); 
  vstore(&Ctref(4, 4), Cr1_4); 
  vstore(&Ctref(4, 5), Cr1_5); 
  vstore(&Ctref(4, 6), Cr1_6); 
  vstore(&Ctref(4, 7), Cr1_7); 
  vstore(&Ctref(4, 8), Cr1_8); 
  vstore(&Ctref(4, 9), Cr1_9); 
  vstore(&Ctref(4, 10), Cr1_10); 
  vstore(&Ctref(4, 11), Cr1_11); 

    if (beta != zero) { 
      for (j = 0; j < nr; j++) 
        for (i = 0; i < mr; i++) 
          Ccol(i,j) = beta * Ccol(i,j) + Ctcol(i,j); 
    } else { 
      for (j = 0; j < nr; j++) 
        for (i = 0; i < mr; i++) 
          Ccol(i,j) = Ctcol(i,j); 
    } 
  } else if ((mr == MR) && (nr == NR)) { 
  if ( beta != zero ) {
    vload(A0_0, &Ccol(0, 0));
    vload(A0_1, &Ccol(0, 1));
    vload(A0_2, &Ccol(0, 2));
    vload(A0_3, &Ccol(0, 3));
    vload(A0_4, &Ccol(0, 4));
    vload(A0_5, &Ccol(0, 5));
    vload(A0_6, &Ccol(0, 6));
    vload(A0_7, &Ccol(0, 7));
    vload(A0_8, &Ccol(0, 8));
    vload(A0_9, &Ccol(0, 9));
    vload(A0_10, &Ccol(0, 10));
    vload(A0_11, &Ccol(0, 11));
    vload(A1_0, &Ccol(4, 0));
    vload(A1_1, &Ccol(4, 1));
    vload(A1_2, &Ccol(4, 2));
    vload(A1_3, &Ccol(4, 3));
    vload(A1_4, &Ccol(4, 4));
    vload(A1_5, &Ccol(4, 5));
    vload(A1_6, &Ccol(4, 6));
    vload(A1_7, &Ccol(4, 7));
    vload(A1_8, &Ccol(4, 8));
    vload(A1_9, &Ccol(4, 9));
    vload(A1_10, &Ccol(4, 10));
    vload(A1_11, &Ccol(4, 11));
    Cr0_0 += beta * A0_0; 
    Cr0_1 += beta * A0_1; 
    Cr0_2 += beta * A0_2; 
    Cr0_3 += beta * A0_3; 
    Cr0_4 += beta * A0_4; 
    Cr0_5 += beta * A0_5; 
    Cr0_6 += beta * A0_6; 
    Cr0_7 += beta * A0_7; 
    Cr0_8 += beta * A0_8; 
    Cr0_9 += beta * A0_9; 
    Cr0_10 += beta * A0_10; 
    Cr0_11 += beta * A0_11; 

    Cr1_0 += beta * A1_0; 
    Cr1_1 += beta * A1_1; 
    Cr1_2 += beta * A1_2; 
    Cr1_3 += beta * A1_3; 
    Cr1_4 += beta * A1_4; 
    Cr1_5 += beta * A1_5; 
    Cr1_6 += beta * A1_6; 
    Cr1_7 += beta * A1_7; 
    Cr1_8 += beta * A1_8; 
    Cr1_9 += beta * A1_9; 
    Cr1_10 += beta * A1_10; 
    Cr1_11 += beta * A1_11; 

  }
  // Store the micro-tile in memory
  vstore(&Ccol(0,0), Cr0_0);
  vstore(&Ccol(0,1), Cr0_1);
  vstore(&Ccol(0,2), Cr0_2);
  vstore(&Ccol(0,3), Cr0_3);
  vstore(&Ccol(0,4), Cr0_4);
  vstore(&Ccol(0,5), Cr0_5);
  vstore(&Ccol(0,6), Cr0_6);
  vstore(&Ccol(0,7), Cr0_7);
  vstore(&Ccol(0,8), Cr0_8);
  vstore(&Ccol(0,9), Cr0_9);
  vstore(&Ccol(0,10), Cr0_10);
  vstore(&Ccol(0,11), Cr0_11);
  vstore(&Ccol(4,0), Cr1_0);
  vstore(&Ccol(4,1), Cr1_1);
  vstore(&Ccol(4,2), Cr1_2);
  vstore(&Ccol(4,3), Cr1_3);
  vstore(&Ccol(4,4), Cr1_4);
  vstore(&Ccol(4,5), Cr1_5);
  vstore(&Ccol(4,6), Cr1_6);
  vstore(&Ccol(4,7), Cr1_7);
  vstore(&Ccol(4,8), Cr1_8);
  vstore(&Ccol(4,9), Cr1_9);
  vstore(&Ccol(4,10), Cr1_10);
  vstore(&Ccol(4,11), Cr1_11);
    } 
}

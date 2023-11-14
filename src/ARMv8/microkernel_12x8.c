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
    #define LOOP_BODY_12x8 \
{\
    vload(ar0, Aptr + baseA + 0);\
    vload(ar1, Aptr + baseA + 4);\
    vload(ar2, Aptr + baseA + 8);\
    vload(br0, Bptr + baseB + 0);\
    vload(br1, Bptr + baseB + 4);\
    vupdate(Cr0_0, ar0, br0, 0);\
    vupdate(Cr0_1, ar0, br0, 1);\
    vupdate(Cr0_2, ar0, br0, 2);\
    vupdate(Cr0_3, ar0, br0, 3);\
    vupdate(Cr0_4, ar0, br1, 0);\
    vupdate(Cr0_5, ar0, br1, 1);\
    vupdate(Cr0_6, ar0, br1, 2);\
    vupdate(Cr0_7, ar0, br1, 3);\
    vupdate(Cr1_0, ar1, br0, 0);\
    vupdate(Cr1_1, ar1, br0, 1);\
    vupdate(Cr1_2, ar1, br0, 2);\
    vupdate(Cr1_3, ar1, br0, 3);\
    vupdate(Cr1_4, ar1, br1, 0);\
    vupdate(Cr1_5, ar1, br1, 1);\
    vupdate(Cr1_6, ar1, br1, 2);\
    vupdate(Cr1_7, ar1, br1, 3);\
    vupdate(Cr2_0, ar2, br0, 0);\
    vupdate(Cr2_1, ar2, br0, 1);\
    vupdate(Cr2_2, ar2, br0, 2);\
    vupdate(Cr2_3, ar2, br0, 3);\
    vupdate(Cr2_4, ar2, br1, 0);\
    vupdate(Cr2_5, ar2, br1, 1);\
    vupdate(Cr2_6, ar2, br1, 2);\
    vupdate(Cr2_7, ar2, br1, 3);\
    baseA += Amr; baseB += Bnr;\
} 

inline void gemm_ukernel_Cresident_SIMD_12x8(int mr, int nr, int kc, DTYPE *Ar, int ldA, DTYPE *Br, int ldB, DTYPE *C, int ldC, char orderC, DTYPE beta) {
// mr x nr = 12 x 8 micro-kernel and C resident in regs.
// Inputs:
//   - C stored in column-major order, with leading dimension ldC
//   - Ar packed by columns, with leading dimension mr = 12
//   - Br packed by rows, with leading dimension nr = 8

  if (kc == 0) return;
  const int MR = 12; 
  const int NR = 8; 
  int       i, j, pr, baseA = 0, baseB = 0, ldCt = MR, Amr, Bnr;
  vregister Cr0_0, Cr0_1, Cr0_2, Cr0_3, Cr0_4, Cr0_5, Cr0_6, Cr0_7, 
            Cr1_0, Cr1_1, Cr1_2, Cr1_3, Cr1_4, Cr1_5, Cr1_6, Cr1_7, 
            
            Cr2_0, Cr2_1, Cr2_2, Cr2_3, Cr2_4, Cr2_5, Cr2_6, Cr2_7, 
            A0_0, A0_1, A0_2, A0_3, A0_4, A0_5, A0_6, A0_7, 
            A1_0, A1_1, A1_2, A1_3, A1_4, A1_5, A1_6, A1_7, 

            A2_0, A2_1, A2_2, A2_3, A2_4, A2_5, A2_6, A2_7;
  vregister ar0, ar1, ar2, br0, br1; 
  DTYPE zero = 0.0, Ctmp[MR * NR], *Aptr, *Bptr; 


  vinit(Cr0_0); vinit(Cr0_1); vinit(Cr0_2); vinit(Cr0_3); vinit(Cr0_4); vinit(Cr0_5); vinit(Cr0_6); vinit(Cr0_7); 
  vinit(Cr1_0); vinit(Cr1_1); vinit(Cr1_2); vinit(Cr1_3); vinit(Cr1_4); vinit(Cr1_5); vinit(Cr1_6); vinit(Cr1_7); 
  vinit(Cr2_0); vinit(Cr2_1); vinit(Cr2_2); vinit(Cr2_3); vinit(Cr2_4); vinit(Cr2_5); vinit(Cr2_6); vinit(Cr2_7); 
  vinit(ar0); vinit(ar1); vinit(ar2); 
  vinit(br0); vinit(br1); 
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
      LOOP_BODY_12x8; 
    } 
    for (pr = it; pr < kc; pr += FUNROLL) { 
    LOOP_BODY_12x8;
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
  vstore(&Ctref(4, 0), Cr1_0); 
  vstore(&Ctref(4, 1), Cr1_1); 
  vstore(&Ctref(4, 2), Cr1_2); 
  vstore(&Ctref(4, 3), Cr1_3); 
  vstore(&Ctref(4, 4), Cr1_4); 
  vstore(&Ctref(4, 5), Cr1_5); 
  vstore(&Ctref(4, 6), Cr1_6); 
  vstore(&Ctref(4, 7), Cr1_7); 
  vstore(&Ctref(8, 0), Cr2_0); 
  vstore(&Ctref(8, 1), Cr2_1); 
  vstore(&Ctref(8, 2), Cr2_2); 
  vstore(&Ctref(8, 3), Cr2_3); 
  vstore(&Ctref(8, 4), Cr2_4); 
  vstore(&Ctref(8, 5), Cr2_5); 
  vstore(&Ctref(8, 6), Cr2_6); 
  vstore(&Ctref(8, 7), Cr2_7); 

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
    vload(A1_0, &Ccol(4, 0));
    vload(A1_1, &Ccol(4, 1));
    vload(A1_2, &Ccol(4, 2));
    vload(A1_3, &Ccol(4, 3));
    vload(A1_4, &Ccol(4, 4));
    vload(A1_5, &Ccol(4, 5));
    vload(A1_6, &Ccol(4, 6));
    vload(A1_7, &Ccol(4, 7));
    vload(A2_0, &Ccol(8, 0));
    vload(A2_1, &Ccol(8, 1));
    vload(A2_2, &Ccol(8, 2));
    vload(A2_3, &Ccol(8, 3));
    vload(A2_4, &Ccol(8, 4));
    vload(A2_5, &Ccol(8, 5));
    vload(A2_6, &Ccol(8, 6));
    vload(A2_7, &Ccol(8, 7));
    Cr0_0 += beta * A0_0; 
    Cr0_1 += beta * A0_1; 
    Cr0_2 += beta * A0_2; 
    Cr0_3 += beta * A0_3; 
    Cr0_4 += beta * A0_4; 
    Cr0_5 += beta * A0_5; 
    Cr0_6 += beta * A0_6; 
    Cr0_7 += beta * A0_7; 

    Cr1_0 += beta * A1_0; 
    Cr1_1 += beta * A1_1; 
    Cr1_2 += beta * A1_2; 
    Cr1_3 += beta * A1_3; 
    Cr1_4 += beta * A1_4; 
    Cr1_5 += beta * A1_5; 
    Cr1_6 += beta * A1_6; 
    Cr1_7 += beta * A1_7; 

    Cr2_0 += beta * A2_0; 
    Cr2_1 += beta * A2_1; 
    Cr2_2 += beta * A2_2; 
    Cr2_3 += beta * A2_3; 
    Cr2_4 += beta * A2_4; 
    Cr2_5 += beta * A2_5; 
    Cr2_6 += beta * A2_6; 
    Cr2_7 += beta * A2_7; 

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
  vstore(&Ccol(4,0), Cr1_0);
  vstore(&Ccol(4,1), Cr1_1);
  vstore(&Ccol(4,2), Cr1_2);
  vstore(&Ccol(4,3), Cr1_3);
  vstore(&Ccol(4,4), Cr1_4);
  vstore(&Ccol(4,5), Cr1_5);
  vstore(&Ccol(4,6), Cr1_6);
  vstore(&Ccol(4,7), Cr1_7);
  vstore(&Ccol(8,0), Cr2_0);
  vstore(&Ccol(8,1), Cr2_1);
  vstore(&Ccol(8,2), Cr2_2);
  vstore(&Ccol(8,3), Cr2_3);
  vstore(&Ccol(8,4), Cr2_4);
  vstore(&Ccol(8,5), Cr2_5);
  vstore(&Ccol(8,6), Cr2_6);
  vstore(&Ccol(8,7), Cr2_7);
    } 
}
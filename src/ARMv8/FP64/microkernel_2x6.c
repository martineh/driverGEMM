#include "microkernel.h"
// Macros for ARM NEON 
    #define vl_fp64 2 
    #define vregister float64x2_t 
    #define vload(vreg, mem)                vreg = vld1q_f64(mem) 
    #define vstore(mem, vreg)               vst1q_f64(mem, vreg) 
    #define vupdate(vreg1, vreg2, vreg3, j) vreg1 = vfmaq_laneq_f64(vreg1, vreg2, vreg3, j) 
    #define vinit(vreg)                     vreg = vmovq_n_f64(0) 
    #define vbroadcast(a1,a2)               a1 = vdupq_n_f64(a2) 
    #define Ccol(a1,a2)  C[(a2) * (ldC) + (a1)] 
    #define Crow(a1,a2)  C[(a1) * (ldC) + (a2)] 
    #define Ctcol(a1,a2) Ctmp[(a2) * (ldCt) + (a1)] 
    #define Ctrow(a1,a2) Ctmp[(a1) * (ldCt) + (a2)] 
    #define Ctref(a1,a2) Ctmp[(a2) * (ldCt) + (a1)] 
    #define FUNROLL 1
    #define LOOP_BODY_2x6 \
{\
    vload(ar0, Aptr + baseA + 0);\
    vload(br0, Bptr + baseB + 0);\
    vload(br1, Bptr + baseB + 2);\
    vload(br2, Bptr + baseB + 4);\
    vupdate(Cr0_0, ar0, br0, 0);\
    vupdate(Cr0_1, ar0, br0, 1);\
    vupdate(Cr0_2, ar0, br1, 0);\
    vupdate(Cr0_3, ar0, br1, 1);\
    vupdate(Cr0_4, ar0, br2, 0);\
    vupdate(Cr0_5, ar0, br2, 1);\
    baseA += Amr; baseB += Bnr;\
} 

inline void gemm_ukernel_Cresident_SIMD_2x6(int mr, int nr, int kc, DTYPE *Ar, int ldA, DTYPE *Br, int ldB, DTYPE *C, int ldC, char orderC, DTYPE beta) {
// mr x nr = 2 x 6 micro-kernel and C resident in regs.
// Inputs:
//   - C stored in column-major order, with leading dimension ldC
//   - Ar packed by columns, with leading dimension mr = 2
//   - Br packed by rows, with leading dimension nr = 6

  if (kc == 0) return;
  const int MR = 2; 
  const int NR = 6; 
  int       i, j, pr, baseA = 0, baseB = 0, ldCt = MR, Amr, Bnr;
  vregister Cr0_0, Cr0_1, Cr0_2, Cr0_3, Cr0_4, Cr0_5, A0_0, A0_1, A0_2, A0_3, A0_4, A0_5;
  vregister ar0, br0, br1, br2; 
  DTYPE zero = 0.0, Ctmp[MR * NR], *Aptr, *Bptr; 


  vinit(Cr0_0); vinit(Cr0_1); vinit(Cr0_2); vinit(Cr0_3); vinit(Cr0_4); vinit(Cr0_5); 
  vinit(ar0); 
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
      LOOP_BODY_2x6; 
    } 
    for (pr = it; pr < kc; pr += FUNROLL) { 
    LOOP_BODY_2x6;
    } 
  if ((mr < MR) || (nr < NR)) {
  vstore(&Ctref(0, 0), Cr0_0); 
  vstore(&Ctref(0, 1), Cr0_1); 
  vstore(&Ctref(0, 2), Cr0_2); 
  vstore(&Ctref(0, 3), Cr0_3); 
  vstore(&Ctref(0, 4), Cr0_4); 
  vstore(&Ctref(0, 5), Cr0_5); 

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
    Cr0_0 += beta * A0_0; 
    Cr0_1 += beta * A0_1; 
    Cr0_2 += beta * A0_2; 
    Cr0_3 += beta * A0_3; 
    Cr0_4 += beta * A0_4; 
    Cr0_5 += beta * A0_5; 

  }
  // Store the micro-tile in memory
  vstore(&Ccol(0,0), Cr0_0);
  vstore(&Ccol(0,1), Cr0_1);
  vstore(&Ccol(0,2), Cr0_2);
  vstore(&Ccol(0,3), Cr0_3);
  vstore(&Ccol(0,4), Cr0_4);
  vstore(&Ccol(0,5), Cr0_5);
    } 
}

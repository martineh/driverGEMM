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
    #define LOOP_BODY_8x2 \
{\
    vload(ar0, Aptr + baseA + 0);\
    vload(ar1, Aptr + baseA + 2);\
    vload(ar2, Aptr + baseA + 4);\
    vload(ar3, Aptr + baseA + 6);\
    vload(br0, Bptr + baseB + 0);\
    vupdate(Cr0_0, ar0, br0, 0);\
    vupdate(Cr0_1, ar0, br0, 1);\
    vupdate(Cr1_0, ar1, br0, 0);\
    vupdate(Cr1_1, ar1, br0, 1);\
    vupdate(Cr2_0, ar2, br0, 0);\
    vupdate(Cr2_1, ar2, br0, 1);\
    vupdate(Cr3_0, ar3, br0, 0);\
    vupdate(Cr3_1, ar3, br0, 1);\
    baseA += Amr; baseB += Bnr;\
} 

inline void gemm_ukernel_Cresident_SIMD_8x2(int mr, int nr, int kc, DTYPE *Ar, int ldA, DTYPE *Br, int ldB, DTYPE *C, int ldC, char orderC, DTYPE beta) {
// mr x nr = 8 x 2 micro-kernel and C resident in regs.
// Inputs:
//   - C stored in column-major order, with leading dimension ldC
//   - Ar packed by columns, with leading dimension mr = 8
//   - Br packed by rows, with leading dimension nr = 2

  if (kc == 0) return;
  const int MR = 8; 
  const int NR = 2; 
  int       i, j, pr, baseA = 0, baseB = 0, ldCt = MR, Amr, Bnr;
  vregister Cr0_0, Cr0_1, 
            Cr1_0, Cr1_1, 
            
            Cr2_0, Cr2_1, 
            
            Cr3_0, Cr3_1, 
            A0_0, A0_1, 
            A1_0, A1_1, 

            A2_0, A2_1, 

            A3_0, A3_1;
  vregister ar0, ar1, ar2, ar3, br0; 
  DTYPE zero = 0.0, Ctmp[MR * NR], *Aptr, *Bptr; 


  vinit(Cr0_0); vinit(Cr0_1); 
  vinit(Cr1_0); vinit(Cr1_1); 
  vinit(Cr2_0); vinit(Cr2_1); 
  vinit(Cr3_0); vinit(Cr3_1); 
  vinit(ar0); vinit(ar1); vinit(ar2); vinit(ar3); 
  vinit(br0); 
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
      LOOP_BODY_8x2; 
    } 
    for (pr = it; pr < kc; pr += FUNROLL) { 
    LOOP_BODY_8x2;
    } 
  if ((mr < MR) || (nr < NR)) {
  vstore(&Ctref(0, 0), Cr0_0); 
  vstore(&Ctref(0, 1), Cr0_1); 
  vstore(&Ctref(2, 0), Cr1_0); 
  vstore(&Ctref(2, 1), Cr1_1); 
  vstore(&Ctref(4, 0), Cr2_0); 
  vstore(&Ctref(4, 1), Cr2_1); 
  vstore(&Ctref(6, 0), Cr3_0); 
  vstore(&Ctref(6, 1), Cr3_1); 

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
    vload(A1_0, &Ccol(2, 0));
    vload(A1_1, &Ccol(2, 1));
    vload(A2_0, &Ccol(4, 0));
    vload(A2_1, &Ccol(4, 1));
    vload(A3_0, &Ccol(6, 0));
    vload(A3_1, &Ccol(6, 1));
    Cr0_0 += beta * A0_0; 
    Cr0_1 += beta * A0_1; 

    Cr1_0 += beta * A1_0; 
    Cr1_1 += beta * A1_1; 

    Cr2_0 += beta * A2_0; 
    Cr2_1 += beta * A2_1; 

    Cr3_0 += beta * A3_0; 
    Cr3_1 += beta * A3_1; 

  }
  // Store the micro-tile in memory
  vstore(&Ccol(0,0), Cr0_0);
  vstore(&Ccol(0,1), Cr0_1);
  vstore(&Ccol(2,0), Cr1_0);
  vstore(&Ccol(2,1), Cr1_1);
  vstore(&Ccol(4,0), Cr2_0);
  vstore(&Ccol(4,1), Cr2_1);
  vstore(&Ccol(6,0), Cr3_0);
  vstore(&Ccol(6,1), Cr3_1);
    } 
}

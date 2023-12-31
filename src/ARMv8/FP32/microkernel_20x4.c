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
    #define LOOP_BODY_20x4 \
{\
    vload(ar0, Aptr + baseA + 0);\
    vload(ar1, Aptr + baseA + 4);\
    vload(ar2, Aptr + baseA + 8);\
    vload(ar3, Aptr + baseA + 12);\
    vload(ar4, Aptr + baseA + 16);\
    vload(br0, Bptr + baseB + 0);\
    vupdate(Cr0_0, ar0, br0, 0);\
    vupdate(Cr0_1, ar0, br0, 1);\
    vupdate(Cr0_2, ar0, br0, 2);\
    vupdate(Cr0_3, ar0, br0, 3);\
    vupdate(Cr1_0, ar1, br0, 0);\
    vupdate(Cr1_1, ar1, br0, 1);\
    vupdate(Cr1_2, ar1, br0, 2);\
    vupdate(Cr1_3, ar1, br0, 3);\
    vupdate(Cr2_0, ar2, br0, 0);\
    vupdate(Cr2_1, ar2, br0, 1);\
    vupdate(Cr2_2, ar2, br0, 2);\
    vupdate(Cr2_3, ar2, br0, 3);\
    vupdate(Cr3_0, ar3, br0, 0);\
    vupdate(Cr3_1, ar3, br0, 1);\
    vupdate(Cr3_2, ar3, br0, 2);\
    vupdate(Cr3_3, ar3, br0, 3);\
    vupdate(Cr4_0, ar4, br0, 0);\
    vupdate(Cr4_1, ar4, br0, 1);\
    vupdate(Cr4_2, ar4, br0, 2);\
    vupdate(Cr4_3, ar4, br0, 3);\
    baseA += Amr; baseB += Bnr;\
} 

inline void gemm_ukernel_Cresident_SIMD_20x4(int mr, int nr, int kc, DTYPE *Ar, int ldA, DTYPE *Br, int ldB, DTYPE *C, int ldC, char orderC, DTYPE beta) {
// mr x nr = 20 x 4 micro-kernel and C resident in regs.
// Inputs:
//   - C stored in column-major order, with leading dimension ldC
//   - Ar packed by columns, with leading dimension mr = 20
//   - Br packed by rows, with leading dimension nr = 4

  if (kc == 0) return;
  const int MR = 20; 
  const int NR = 4; 
  int       i, j, pr, baseA = 0, baseB = 0, ldCt = MR, Amr, Bnr;
  vregister Cr0_0, Cr0_1, Cr0_2, Cr0_3, 
            Cr1_0, Cr1_1, Cr1_2, Cr1_3, 
            
            Cr2_0, Cr2_1, Cr2_2, Cr2_3, 
            
            Cr3_0, Cr3_1, Cr3_2, Cr3_3, 
            
            Cr4_0, Cr4_1, Cr4_2, Cr4_3, 
            A0_0, A0_1, A0_2, A0_3, 
            A1_0, A1_1, A1_2, A1_3, 

            A2_0, A2_1, A2_2, A2_3, 

            A3_0, A3_1, A3_2, A3_3, 

            A4_0, A4_1, A4_2, A4_3;
  vregister ar0, ar1, ar2, ar3, ar4, br0; 
  DTYPE zero = 0.0, Ctmp[MR * NR], *Aptr, *Bptr; 


  vinit(Cr0_0); vinit(Cr0_1); vinit(Cr0_2); vinit(Cr0_3); 
  vinit(Cr1_0); vinit(Cr1_1); vinit(Cr1_2); vinit(Cr1_3); 
  vinit(Cr2_0); vinit(Cr2_1); vinit(Cr2_2); vinit(Cr2_3); 
  vinit(Cr3_0); vinit(Cr3_1); vinit(Cr3_2); vinit(Cr3_3); 
  vinit(Cr4_0); vinit(Cr4_1); vinit(Cr4_2); vinit(Cr4_3); 
  vinit(ar0); vinit(ar1); vinit(ar2); vinit(ar3); vinit(ar4); 
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
      LOOP_BODY_20x4; 
    } 
    for (pr = it; pr < kc; pr += FUNROLL) { 
    LOOP_BODY_20x4;
    } 
  if ((mr < MR) || (nr < NR)) {
  vstore(&Ctref(0, 0), Cr0_0); 
  vstore(&Ctref(0, 1), Cr0_1); 
  vstore(&Ctref(0, 2), Cr0_2); 
  vstore(&Ctref(0, 3), Cr0_3); 
  vstore(&Ctref(4, 0), Cr1_0); 
  vstore(&Ctref(4, 1), Cr1_1); 
  vstore(&Ctref(4, 2), Cr1_2); 
  vstore(&Ctref(4, 3), Cr1_3); 
  vstore(&Ctref(8, 0), Cr2_0); 
  vstore(&Ctref(8, 1), Cr2_1); 
  vstore(&Ctref(8, 2), Cr2_2); 
  vstore(&Ctref(8, 3), Cr2_3); 
  vstore(&Ctref(12, 0), Cr3_0); 
  vstore(&Ctref(12, 1), Cr3_1); 
  vstore(&Ctref(12, 2), Cr3_2); 
  vstore(&Ctref(12, 3), Cr3_3); 
  vstore(&Ctref(16, 0), Cr4_0); 
  vstore(&Ctref(16, 1), Cr4_1); 
  vstore(&Ctref(16, 2), Cr4_2); 
  vstore(&Ctref(16, 3), Cr4_3); 

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
    vload(A1_0, &Ccol(4, 0));
    vload(A1_1, &Ccol(4, 1));
    vload(A1_2, &Ccol(4, 2));
    vload(A1_3, &Ccol(4, 3));
    vload(A2_0, &Ccol(8, 0));
    vload(A2_1, &Ccol(8, 1));
    vload(A2_2, &Ccol(8, 2));
    vload(A2_3, &Ccol(8, 3));
    vload(A3_0, &Ccol(12, 0));
    vload(A3_1, &Ccol(12, 1));
    vload(A3_2, &Ccol(12, 2));
    vload(A3_3, &Ccol(12, 3));
    vload(A4_0, &Ccol(16, 0));
    vload(A4_1, &Ccol(16, 1));
    vload(A4_2, &Ccol(16, 2));
    vload(A4_3, &Ccol(16, 3));
    Cr0_0 += beta * A0_0; 
    Cr0_1 += beta * A0_1; 
    Cr0_2 += beta * A0_2; 
    Cr0_3 += beta * A0_3; 

    Cr1_0 += beta * A1_0; 
    Cr1_1 += beta * A1_1; 
    Cr1_2 += beta * A1_2; 
    Cr1_3 += beta * A1_3; 

    Cr2_0 += beta * A2_0; 
    Cr2_1 += beta * A2_1; 
    Cr2_2 += beta * A2_2; 
    Cr2_3 += beta * A2_3; 

    Cr3_0 += beta * A3_0; 
    Cr3_1 += beta * A3_1; 
    Cr3_2 += beta * A3_2; 
    Cr3_3 += beta * A3_3; 

    Cr4_0 += beta * A4_0; 
    Cr4_1 += beta * A4_1; 
    Cr4_2 += beta * A4_2; 
    Cr4_3 += beta * A4_3; 

  }
  // Store the micro-tile in memory
  vstore(&Ccol(0,0), Cr0_0);
  vstore(&Ccol(0,1), Cr0_1);
  vstore(&Ccol(0,2), Cr0_2);
  vstore(&Ccol(0,3), Cr0_3);
  vstore(&Ccol(4,0), Cr1_0);
  vstore(&Ccol(4,1), Cr1_1);
  vstore(&Ccol(4,2), Cr1_2);
  vstore(&Ccol(4,3), Cr1_3);
  vstore(&Ccol(8,0), Cr2_0);
  vstore(&Ccol(8,1), Cr2_1);
  vstore(&Ccol(8,2), Cr2_2);
  vstore(&Ccol(8,3), Cr2_3);
  vstore(&Ccol(12,0), Cr3_0);
  vstore(&Ccol(12,1), Cr3_1);
  vstore(&Ccol(12,2), Cr3_2);
  vstore(&Ccol(12,3), Cr3_3);
  vstore(&Ccol(16,0), Cr4_0);
  vstore(&Ccol(16,1), Cr4_1);
  vstore(&Ccol(16,2), Cr4_2);
  vstore(&Ccol(16,3), Cr4_3);
    } 
}

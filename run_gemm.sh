#!/bin/bash

#-------------------------------------------------------------
# Variants : example : B3A2C0(B3-->L3, A-->L2, C--> v-register)
# ORDER    : C = column major, R = row major
# TRANS    : T = transpose   , N = !T
#-------------------------------------------------------------
ORDERA=C  #R
ORDERB=C  #R
ORDERC=C  #R
TRANSA=N  #T
TRANSB=N  #T
#-------------------------------------------------------------


#-------------------------------------------------------------
# ALPHA - BETA Fixed Values
#-------------------------------------------------------------
ALPHA=1.0   
BETA=0.0   
#-------------------------------------------------------------


#-------------------------------------------------------------
# Execution Minimum Time
#-------------------------------------------------------------
TIMIN=1.0 
#-------------------------------------------------------------


#-------------------------------------------------------------
# Enable (T) | Disable (F) Testing Mode
#-------------------------------------------------------------
TEST=T
#-------------------------------------------------------------


#-------------------------------------------------------------
# Enable (0) | Disable (1) Visual Mode
#-------------------------------------------------------------
VISUAL=0
#-------------------------------------------------------------

#-------------------------------------------------------------
# Micro-kernel size. 
#-------------------------------------------------------------
#
#   Available sizes (MR x NR) 
#  ---------------------------
#   FP32: 4x4,  4x8,  4x12, 4x16,  4x20,   8x4, 
#         8x8, 8x12,  12x4, 12x8,  16x4,  20x4
#
#   FP64: 2x2,   2x4,  2x6,  2x8,  2x10,  2x12, 
#         4x2,   4x4,  4x6,  4x8,  4x20,  4x12,
#         6x2,   6x4,  8x2,  8x4,  10x2,  10x4,
#         12x2, 12x4, 
#
#-------------------------------------------------------------
MR=12
NR=2
#-------------------------------------------------------------

if $(echo $1 | grep -q "batch"); then 
	source $1
else
	source batch/null.sh
fi

mkdir -p output

taskset -c 0 ./build/test_gemm.x "" $ORDERA $ORDERB $ORDERC $TRANSA $TRANSB $ALPHA $BETA $MMIN $MMAX $MSTEP $NMIN $NMAX $NSTEP $KMIN $KMAX $KSTEP $VISUAL $TIMIN $TEST $MR $NR $1 $2


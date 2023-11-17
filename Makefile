
#------------------------------------------
#| COMPILERS                              |
#------------------------------------------
CC       =  gcc
CLINKER  =  gcc
#------------------------------------------


#------------------------------------------
#| TYPES                                  |
#------------------------------------------
# [*] FP32                                |
# [*] FP64                                |
#------------------------------------------
DTYPE = FP64
#------------------------------------------


#------------------------------------------
#| COMPILER FLAGS                         |
#------------------------------------------
MODE   = FAMILY
SIMD   = ARMv8

FLAGS  = -O3 -march=armv8-a+simd+fp -Wall 
LIBS   = -lm
#------------------------------------------


#------------------------------------------
#| PATHS CONFIGURE                        |
#------------------------------------------
vpath %.c ./src
vpath %.h ./src

vpath %.c ./modelLevel
vpath %.h ./modelLevel

#------------------------------------------

OBJDIR = build
BIN    = test_gemm.x
_OBJ   = model_level.o gemm_blis.o inutils.o sutils.o test_gemm.o

OBJ = $(patsubst %, $(OBJDIR)/%, $(_OBJ))

ifeq ($(DTYPE), FP32)

  vpath %.c ./src/ARMv8/FP32
  vpath %.h ./src/ARMv8/FP32

  SRC_FILES = $(wildcard ./src/ARMv8/FP32/*.c)
  OBJ_FILES = $(patsubst ./src/ARMv8/FP32/%.c, $(OBJDIR)/%.o, $(SRC_FILES))

else

  vpath %.c ./src/ARMv8/FP64
  vpath %.h ./src/ARMv8/FP64

  SRC_FILES = $(wildcard ./src/ARMv8/FP64/*.c)
  OBJ_FILES = $(patsubst ./src/ARMv8/FP64/%.c, $(OBJDIR)/%.o, $(SRC_FILES))

endif

OBJ += $(OBJ_FILES) 
OPTFLAGS = $(FLAGS) -DCHECK -D$(MODE) -D$(SIMD) -D$(DTYPE)
#------------------------------------------

default: $(OBJDIR)/$(BIN)

$(OBJDIR)/model_level.o: src/modelLevel/model_level.c
	@mkdir -p $(OBJDIR)
	$(CC) $(CFLAGS) $(OPTFLAGS) -c -o $@ $< $(INCLUDE) $(LIBS)

$(OBJDIR)/%.o: ./src/ARMv8/$(DTYPE)/%.c
	$(CC) $(CFLAGS) $(OPTFLAGS) -c -o $@ $< $(INCLUDE) $(LIBS)

$(OBJDIR)/%.o:%.c
	$(CC) $(CFLAGS) $(OPTFLAGS) -c -o $@ $< $(INCLUDE) $(LIBS)

$(OBJDIR)/$(BIN): $(OBJ) 
	$(CLINKER) $(OPTFLAGS) -o $@ $^ $(LIBS)

clean:
	rm $(OBJDIR)/* 


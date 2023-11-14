
#------------------------------------------
#| COMPILERS                              |
#------------------------------------------
CC       =  gcc
CLINKER  =  gcc
#------------------------------------------

OBJDIR = build
BIN    = test_gemm.x
_OBJ   = gemm_blis.o inutils.o sutils.o test_gemm.o model_level.o 

#------------------------------------------
#| PATHS CONFIGURE                        |
#------------------------------------------
vpath %.c ./src
vpath %.h ./src

vpath %.c ./modelLevel
vpath %.h ./modelLevel

vpath %.c ./src/ARMv8
vpath %.h ./src/ARMv8
#------------------------------------------

#------------------------------------------
#| COMPILER FLAGS                         |
#------------------------------------------
MODE=-DFAMILY

#Dependes Arquitecture Mode
SIMD  = -DARMv8
FLAGS = -O3 -march=armv8-a+simd+fp -fopenmp -Wall -Wunused-function

OBJ = $(patsubst %, $(OBJDIR)/%, $(_OBJ))

SRC_FILES = $(wildcard ./src/ARMv8/*.c)
OBJ_FILES = $(patsubst ./src/ARMv8/%.c, $(OBJDIR)/%.o, $(SRC_FILES))

OBJ += $(OBJ_FILES) 

OPTFLAGS = $(FLAGS) -DCHECK $(MODE) $(SIMD) $(DTYPE)
#------------------------------------------

default: $(OBJDIR)/$(BIN)

$(OBJDIR)/%.o:%.c
	@mkdir -p $(OBJDIR)
	$(CC) $(CFLAGS) $(OPTFLAGS) -c -o $@ $< $(INCLUDE) $(LIBS)

$(OBJDIR)/$(BIN): $(OBJ) 
	$(CLINKER) $(OPTFLAGS) -o $@ $^ $(LIBS)

clean:
	rm $(OBJDIR)/* 


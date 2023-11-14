
#------------------------------------------
#| COMPILERS                              |
#------------------------------------------
CC       =  gcc
CLINKER  =  gcc
#------------------------------------------

OBJDIR = build
BIN    = test_gemm.x
_OBJ   = model_level.o gemm_blis.o inutils.o sutils.o test_gemm.o

#------------------------------------------
#| PATHS CONFIGURE                        |
#------------------------------------------
vpath %.c ./src
vpath %.h ./src

vpath %.c ./modelLevel
vpath %.h ./modelLevel

#vpath %.c ./src/ARMv8
#vpath %.h ./src/ARMv8
#------------------------------------------

#------------------------------------------
#| COMPILER FLAGS                         |
#------------------------------------------
#DTYPE = -DFP32
MODE  = -DFAMILY
#SIMD  = -DARMv8
FLAGS = -O3 -Wall #-march=armv8-a+simd+fp -Wall 
LIBS  = -lm

OBJ = $(patsubst %, $(OBJDIR)/%, $(_OBJ))

#SRC_FILES = $(wildcard ./src/ARMv8/*.c)
#OBJ_FILES = $(patsubst ./src/ARMv8/%.c, $(OBJDIR)/%.o, $(SRC_FILES))

#OBJ += $(OBJ_FILES) 

OPTFLAGS = $(FLAGS) -DCHECK $(MODE) $(SIMD) $(DTYPE)
#------------------------------------------

default: $(OBJDIR)/$(BIN)

$(OBJDIR)/model_level.o: src/modelLevel/model_level.c
	@mkdir -p $(OBJDIR)
	$(CC) $(CFLAGS) $(OPTFLAGS) -c -o $@ $< $(INCLUDE) $(LIBS)

$(OBJDIR)/%.o: ./src/ARMv8/%.c
	$(CC) $(CFLAGS) $(OPTFLAGS) -c -o $@ $< $(INCLUDE) $(LIBS)

$(OBJDIR)/%.o:%.c
	$(CC) $(CFLAGS) $(OPTFLAGS) -c -o $@ $< $(INCLUDE) $(LIBS)

$(OBJDIR)/$(BIN): $(OBJ) 
	$(CLINKER) $(OPTFLAGS) -o $@ $^ $(LIBS)

clean:
	rm $(OBJDIR)/* 


#if defined(FP16)
  #define DTYPE _Float16
#elif defined(FP32)
  #define DTYPE float
#elif defined(FP64)
  #define DTYPE double
#endif

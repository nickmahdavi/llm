#ifndef OPS_H
#define OPS_H

#include "struct.h"

#define DEC_BROADCAST_OP(name) \
    void name(Tensor *x, Tensor *y, Tensor *out);

DEC_BROADCAST_OP(madd)
DEC_BROADCAST_OP(msub)
DEC_BROADCAST_OP(mmult)

void mscal(Tensor *x, float s, Tensor *out);

void matmul(Tensor *restrict x, Tensor *restrict y, Tensor *restrict out);
void matmul_at(Tensor *restrict x, Tensor *restrict y, Tensor *restrict out);
void matmul_bt(Tensor *restrict x, Tensor *restrict y, Tensor *restrict out);
void softmax(Tensor *in, Tensor *out);
void soft_grad(Tensor *dS, Tensor *S, Tensor *out);
void triu_mask(Tensor *in, Tensor *out, float val);
void gelu(Tensor *in, Tensor *out);
void lnstats(Tensor *in, Tensor *mean, Tensor *var, float eps);
void batch_mean(Tensor *in, Tensor *out);
void gelu_grad(Tensor *dG, Tensor *in, Tensor *out);
void rms_grad(Tensor *dX, Tensor *safevar, Tensor *X, Tensor *out);
void step(Tensor *x, Tensor *grad, float eta);
float crossentropy(Tensor *X, Tensor *y);
void rms_grad(Tensor *dX, Tensor *safevar, Tensor *X, Tensor *out);
void rms(Tensor *in, Tensor *safevar, Tensor *out, float eps);
size_t round_up_pow2(size_t n);
uint32_t mueller(uint32_t x);
float crossentropy(Tensor *X, Tensor *y);

#endif
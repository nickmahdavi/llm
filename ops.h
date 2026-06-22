#ifndef OPS_H
#define OPS_H

#include "struct.h"

static inline int base_idx(Tensor *t, int i, int j, int k) {
    return i * t->shape[1] * t->shape[2] * t->shape[3] + j * t->shape[2] * t->shape[3] + k * t->shape[3];
}

#define FOR_ROWS(t) \
    for (int i = 0; i < (t)->shape[0]; i++) \
        for (int j = 0; j < (t)->shape[1]; j++) \
            for (int k = 0; k < (t)->shape[2]; k++) \
                 for (int base = base_idx(t, i, j, k), _done = 0; !_done; _done = 1)

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
void gelu_grad(Tensor *dG, Tensor *in, Tensor *out);
void rms_grad(Tensor *dX, Tensor *safevar, Tensor *X, Tensor *out);
void step(Tensor *x, Tensor *grad, float eta);
float crossentropy(Tensor *X, Tensor *y);
void rms_grad(Tensor *dX, Tensor *safevar, Tensor *X, Tensor *out);
void rms(Tensor *in, Tensor *safevar, Tensor *out, float eps);
size_t round_up_pow2(size_t n);
uint32_t mueller(uint32_t x);
float crossentropy(Tensor *X, Tensor *y);
void batch_mean(Tensor *in, Tensor *out, int n);
void step_adamw(Tensor *w, Tensor *g, Tensor *m, Tensor *v, float beta1, float beta2, float b1t, float b2t, float lambda, float eta, float eps, float scale);
float cosine_lr(int t, int n_warmup, int n_decay, float max_lr, float min_lr);
float grad_norm(Weights *grad, int layers);

#endif
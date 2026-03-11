#include <assert.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include "utils.h"
#include "struct.h"

#define BROADCAST_OP(name, op) \
void name(Tensor *x, Tensor *y, Tensor *out) { \
    int s = 1; \
    for (int i = 3; i >= 0; i--) { \
        if (x->shape[i] == y->shape[3]) break; \
        s *= x->shape[i]; \
    } \
    for (size_t i = 0; i < tsize(x); i++) \
        out->data[i] = x->data[i] op y->data[(i/s) % tsize(y)]; \
} 

#define TILE 32

BROADCAST_OP(madd, +)
BROADCAST_OP(msub, -)
BROADCAST_OP(mmult, *)

#define FOR_ROWS(t) \
    for (int i = 0; i < (t)->shape[0]; i++) \
        for (int j = 0; j < (t)->shape[1]; j++) \
            for (int k = 0; k < (t)->shape[2]; k++) \
                 for (int base = base_idx(t, i, j, k), _done = 0; !_done; _done = 1)

static inline int base_idx(Tensor *t, int i, int j, int k) {
    return i * t->shape[1] * t->shape[2] * t->shape[3] + j * t->shape[2] * t->shape[3] + k * t->shape[3];
}

void mscal(Tensor *x, float s, Tensor *out) {
    for (size_t i = 0; i < tsize(x); i++)
        out->data[i] = x->data[i] * s;
}

void step(Tensor *x, Tensor *grad, float eta) {
    for (size_t i = 0; i < tsize(x); i++) \
        x->data[i] -= grad->data[i] * eta;
}

size_t round_up_pow2(size_t n) {
    if (n == 0) return 1;
    if (n > SIZE_MAX / 2 + 1) return 0;
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n |= n >> 32;
    return n + 1;
}

uint32_t mueller(uint32_t x) {
    x ^= x >> 16;
    x *= 0x45d9f3b;
    x ^= x >> 16;
    x *= 0x45d9f3b;
    x ^= x >> 16;
    return x;
}

void matmul(Tensor *restrict x, Tensor *restrict y, Tensor *restrict out) {
    assert(x->shape[3] == y->shape[2] && out->shape[2] == x->shape[2] && out->shape[3] == y->shape[3] && "matmul shape mismatch");
    int xbatch = x->shape[0] * x->shape[1];
    int ybatch = y->shape[0] * y->shape[1];
    int y_do_batch = (ybatch != 1);
    int xsize = x->shape[2] * x->shape[3];
    int ysize = y->shape[2] * y->shape[3];
    int osize = x->shape[2] * y->shape[3];
    int ii_max = x->shape[2];
    int jj_max = y->shape[3];
    int kk_max = x->shape[3];
    float acc[TILE][TILE];
    for (int b = 0; b < xbatch; b++) {
        for (int ii = 0; ii < ii_max; ii += TILE) {
            int i_max = ii + TILE > ii_max ? ii_max : ii + TILE;
            for (int jj = 0; jj < jj_max; jj += TILE) {
                int j_max = jj + TILE > jj_max ? jj_max : jj + TILE;
                memset(acc, 0, sizeof(acc));
                for (int kk = 0; kk < kk_max; kk += TILE) {
                    int k_max = kk + TILE > kk_max ? kk_max : kk + TILE;
                    for (int i = ii; i < i_max; i++) {
                        for (int k = kk; k < k_max; k++) {
                            float a = x->data[b * xsize + i * kk_max + k];
                            for (int j = jj; j < j_max; j++) {
                                acc[i - ii][j - jj] += a * y->data[b * ysize * y_do_batch + k * jj_max + j];
                            }
                        }
                    }
                }
                for (int i = ii; i < i_max; i ++) {
                    for (int j = jj; j < j_max; j++) {
                        out->data[b * osize + i * jj_max + j] = acc[i - ii][j - jj];
                    }
                }
            }
        }
    }
}

void matmul_at(Tensor *restrict x, Tensor *restrict y, Tensor *restrict out) {
    assert(x->shape[2] == y->shape[2] && out->shape[2] == x->shape[3] && out->shape[3] == y->shape[3] && "matmul_at shape mismatch");
    int xbatch = x->shape[0] * x->shape[1];
    int ybatch = y->shape[0] * y->shape[1];
    int y_do_batch = (ybatch != 1);
    int xsize = x->shape[2] * x->shape[3];
    int ysize = y->shape[2] * y->shape[3];
    int osize = x->shape[3] * y->shape[3];
    int ii_max = x->shape[3];
    int jj_max = y->shape[3];
    int kk_max = x->shape[2];
    float acc[TILE][TILE];
    for (int b = 0; b < xbatch; b++) {
        for (int ii = 0; ii < ii_max; ii += TILE) {
            int i_max = ii + TILE > ii_max ? ii_max : ii + TILE;
            for (int jj = 0; jj < jj_max; jj += TILE) {
                int j_max = jj + TILE > jj_max ? jj_max : jj + TILE;
                memset(acc, 0, sizeof(acc));
                for (int kk = 0; kk < kk_max; kk += TILE) {
                    int k_max = kk + TILE > kk_max ? kk_max : kk + TILE;
                    for (int i = ii; i < i_max; i++) {
                        for (int k = kk; k < k_max; k++) {
                            float a = x->data[b * xsize + k * ii_max + i];
                            for (int j = jj; j < j_max; j++) {
                                acc[i - ii][j - jj] += a * y->data[b * ysize * y_do_batch + k * jj_max + j];
                            }
                        }
                    }
                }
                for (int i = ii; i < i_max; i ++) {
                    for (int j = jj; j < j_max; j++) {
                        out->data[b * osize + i * jj_max + j] = acc[i - ii][j - jj];
                    }
                }
            }
        }
    }
}

void matmul_bt(Tensor *restrict x, Tensor *restrict y, Tensor *restrict out) {
    assert(x->shape[3] == y->shape[3] && out->shape[2] == x->shape[2] && out->shape[3] == y->shape[2] && "matmul_bt shape mismatch");
    int xbatch = x->shape[0] * x->shape[1];
    int ybatch = y->shape[0] * y->shape[1];
    int y_do_batch = (ybatch != 1);
    int xsize = x->shape[2] * x->shape[3];
    int ysize = y->shape[2] * y->shape[3];
    int osize = x->shape[2] * y->shape[2];
    int ii_max = x->shape[2];
    int jj_max = y->shape[2];
    int kk_max = x->shape[3];
    float acc[TILE][TILE];
    for (int b = 0; b < xbatch; b++) {
        for (int ii = 0; ii < ii_max; ii += TILE) {
            int i_max = ii + TILE > ii_max ? ii_max : ii + TILE;
            for (int jj = 0; jj < jj_max; jj += TILE) {
                int j_max = jj + TILE > jj_max ? jj_max : jj + TILE;
                memset(acc, 0, sizeof(acc));
                for (int kk = 0; kk < kk_max; kk += TILE) {
                    int k_max = kk + TILE > kk_max ? kk_max : kk + TILE;
                    for (int i = ii; i < i_max; i++) {
                        for (int j = jj; j < j_max; j++) {
                            for (int k = kk; k < k_max; k++) {
                                acc[i - ii][j - jj] += x->data[b * xsize + i * kk_max + k] * y->data[b * ysize * y_do_batch + j * kk_max + k];
                            }
                        }
                    }
                }
                for (int i = ii; i < i_max; i ++) {
                    for (int j = jj; j < j_max; j++) {
                        out->data[b * osize + i * jj_max + j] = acc[i - ii][j - jj];
                    }
                }
            }
        }
    }
}

void softmax(Tensor *in, Tensor *out) {
    FOR_ROWS(in) {
        float max = -FLT_MAX;
        for (int col = 0; col < in->shape[3]; col++) {
            if (in->data[base + col] > max) max = in->data[base + col];
        }

        float acc = 0;
        for (int col = 0; col < in->shape[3]; col++) {
            out->data[base + col] = expf(in->data[base + col] - max);
            acc += out->data[base + col];
        }

        acc = 1.0 / acc;
        for (int col = 0; col < in->shape[3]; col++) {
            out->data[base + col] *= acc;
        }
    }
}

void triu_mask(Tensor *in, Tensor *out, float val) {
    FOR_ROWS(in) {
        for (int col = k + 1; col < in->shape[3]; col++) {
            out->data[base + col] = val;
        }
    }
}

#define GELU_COEFF 0.797884561f
#define GELU_CONST 0.044715f

void gelu(Tensor *in, Tensor *out) {
    for (size_t i = 0; i < tsize(in); i++) {
        float x = in->data[i];
        out->data[i] = x * 0.5f * (1 + tanhf(GELU_COEFF * (x + GELU_CONST * x * x * x)));
    }
}

void gelu_grad(Tensor *dG, Tensor *in, Tensor *out) {
    for (size_t i = 0; i < tsize(in); i++) {
        float x = in->data[i];
        float z = GELU_COEFF * (x + GELU_CONST * x * x * x);
        float t = tanhf(z);
        float grad = 0.5f * (1.0f + t) + 0.5f * x * (1.0f - t * t) * GELU_COEFF * (1.0f + 3 * GELU_CONST * x * x);
        out->data[i] = dG->data[i] * grad;
    }
}

void lnstats(Tensor *in, Tensor *mean, Tensor *var, float eps) {
    FOR_ROWS(in) {
        float x = 0, acc = 0, acc2 = 0, delta = 0, delta2 = 0;
        for (int col = 0; col < in->shape[3]; col++) {
            x = in->data[base + col];
            delta = x - acc;
            acc += delta / (col + 1);
            delta2 = x - acc;
            acc2 += delta * delta2;
        }
        mean->data[i * in->shape[2] + k] = acc;
        var->data[i * in->shape[2] + k] = 1.0f / sqrtf(acc2 / in->shape[3] + eps);
    }
}

void rms(Tensor *in, Tensor *safevar, Tensor *out, float eps) {
    FOR_ROWS(in) {
        double acc = 0.0;
        for (int col = 0; col < in->shape[3]; col++) {
            acc += (double)(in->data[base + col] * in->data[base + col]);
        }
        float inv_rms = 1.0f / sqrtf((float)acc / in->shape[3] + eps);
        safevar->data[i * in->shape[2] + k] = inv_rms;
        for (int col = 0; col < in->shape[3]; col++) {
            out->data[base + col] = in->data[base + col] * inv_rms;
        }
    }
}

void batch_mean(Tensor *in, Tensor *out) {
    memset(out->data, 0, tsizeof(out));
    int s = 1;
    for (int i = 3; i >= 0; i--) {
        if (in->shape[i] != out->shape[i]) break;
        s *= in->shape[i];
    }
    FOR_ROWS(in) {
        for (int col = 0; col < in->shape[3]; col++) {
            out->data[base % s + col] += in->data[base + col];
        }
    }
    mscal(out, 1.0f * s / tsize(in), out);
}

void soft_grad(Tensor *dS, Tensor *S, Tensor *out) {
    FOR_ROWS(S) {
        float acc = 0;
        for (int col = 0; col < S->shape[3]; col++) {
            acc += S->data[base + col] * dS->data[base + col];
        }
        for (int col = 0; col < S->shape[3]; col++) {
            out->data[base + col] = S->data[base + col] * (dS->data[base + col] - acc);
        }
    }
}

void ln_grad(Tensor *dX, Tensor *safevar, Tensor *X, Tensor *out) {
    FOR_ROWS(dX) {
        float acc = 0, acc2 = 0;
        float var = safevar->data[i * dX->shape[2] + k];
        for (int col = 0; col < dX->shape[3]; col++) {
            acc += dX->data[base + col];
            acc2 += dX->data[base + col] * X->data[base + col];
        }
        acc /= X->shape[3];
        acc2 /= X->shape[3];
        for (int col = 0; col < dX->shape[3]; col++) {
            out->data[base + col] = var * (dX->data[base + col] - acc - X->data[base + col] * acc2);
        }
    }
}

void rms_grad(Tensor *dX, Tensor *safevar, Tensor *X, Tensor *out) {
    FOR_ROWS(dX) {
        float acc = 0;
        float var = safevar->data[i * dX->shape[2] + k];
        for (int col = 0; col < dX->shape[3]; col++) {
            acc += dX->data[base + col] * X->data[base + col];
        }
        acc /= X->shape[3];
        for (int col = 0; col < dX->shape[3]; col++) {
            out->data[base + col] = var * (dX->data[base + col] - X->data[base + col] * acc);
        }
    }
}

float crossentropy(Tensor *X, Tensor *y) {
    float loss = 0;
    FOR_ROWS(X) {
        int col = -1;
        while (y->data[base + ++col] == 0);
        loss -= logf(X->data[base + col] + 1e-06);
    }
    return loss / (X->shape[0] * X->shape[1] * X->shape[2]);
}

static inline float sqacc(Tensor *t) {
    float acc = 0;
    for (int i = 0; i < tsize(t); i++) acc += t->data[i] * t->data[i];
    return acc;
}

float grad_norm(Weights *grad, int layers) {
    float acc = 0;
    acc += sqacc(&grad->token_emb);
    acc += sqacc(&grad->pos_emb);
    for (int i = 0; i < layers; i++) {
        DecoderWeights dgrad = grad->layers[i];
        acc += sqacc(&dgrad.attn.WQ);
        acc += sqacc(&dgrad.attn.WK);
        acc += sqacc(&dgrad.attn.WV);
        acc += sqacc(&dgrad.attn.WO);
        acc += sqacc(&dgrad.ff.b1);
        acc += sqacc(&dgrad.ff.W1);
        acc += sqacc(&dgrad.ff.b2);
        acc += sqacc(&dgrad.ff.W2);
        acc += sqacc(&dgrad.rms1.gamma);
        acc += sqacc(&dgrad.rms2.gamma);
    }
    acc += sqacc(&grad->last_rms.gamma);
    acc += sqacc(&grad->token_unemb);
    return powf(acc, 0.5f);
}
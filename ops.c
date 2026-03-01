#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <string.h>
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

void mscal(Tensor *x, float s, Tensor *out) {
    for (size_t i = 0; i < tsize(x); i++)
        out->data[i] = x->data[i] * s;
}

void step(Tensor *x, Tensor *grad, float eta) {
    for (size_t i = 0; i < tsize(x); i++) \
        x->data[i] -= grad->data[i] * eta;
}

void matmul(Tensor *restrict x, Tensor *restrict y, Tensor *restrict out) {
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
    int batch = in->shape[0];
    int nheads = in->shape[1];
    int rows = in->shape[2];
    int cols  = in->shape[3];
    float acc;
    float max;
    size_t base;
    for (int i = 0; i < batch; i++) {
        for (int j = 0; j < nheads; j++) {
            for (int k = 0; k < rows; k++) {
                base = i * nheads * rows * cols + j * rows * cols + k * cols;
                max = -FLT_MAX;
                for (int l = 0; l < cols; l++) {
                    if (in->data[base + l] > max) max = in->data[base + l];
                }
                acc = 0;
                for (int l = 0; l < cols; l++) {
                    out->data[base + l] = expf(in->data[base + l] - max);
                    acc += out->data[base + l];
                }
                acc = 1.0 / acc;
                for (int l = 0; l < cols; l++) {
                    out->data[base + l] *= acc;
                }
            }
        }
    }
}

void triu_mask(Tensor *in, Tensor *out, float val) {
    size_t b = tsize(in) / in->shape[0];
    int seq = in->shape[2];
    for (int i = 0; i < in->shape[0]; i++) {
        for (int j = 0; j < in->shape[1]; j++) {
            for (int k = 0; k < seq; k++) {
                int base = i * b + j * seq * seq + k * seq;
                for (int l = k + 1; l < seq; l++) {
                    out->data[base + l] = val;
                }
            }
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
    float x, acc, acc2, delta, delta2;
    int seq = in->shape[2];
    int dmodel = in->shape[3];
    for (int i = 0; i < in->shape[0]; i++) {
        for (int j = 0; j < seq; j++) {
            acc = 0;
            acc2 = 0;
            delta = 0;
            delta2 = 0;
            for (int k = 0; k < dmodel; k++) {
                x = in->data[i * seq * dmodel + j * dmodel + k];
                delta = x - acc;
                acc += delta / (k + 1);
                delta2 = x - acc;
                acc2 += delta * delta2;
            }
            mean->data[i * seq + j] = acc;
            var->data[i * seq + j] = 1.0f / sqrtf(acc2 / dmodel + eps);
        }
    }
}

void soft_grad(Tensor *dS, Tensor *S, Tensor *out) {
    int d = S->shape[0];
    int h = S->shape[1];
    int seq = S->shape[2];
    for (int i = 0; i < d; i++) {
        for (int j = 0; j < h; j++) {
            for (int k = 0; k < seq; k++) {
                float acc = 0;
                int base = i * h * seq * seq + j * seq * seq + k * seq;
                for (int l = 0; l < seq; l++) {
                    acc += S->data[base + l] * dS->data[base + l];
                }
                for (int l = 0; l < seq; l ++) {
                    out->data[base + l] = S->data[base + l] * (dS->data[base + l] - acc);
                }
            }
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
    for (int i = 0; i < in->shape[0]; i++) {
        for (int j = 0; j < in->shape[1]; j++) {
            for (int k = 0; k < in->shape[2]; k++) {
                int base = i * in->shape[1] * in->shape[2] * in->shape[3] + j * in->shape[2] * in->shape[3] + k * in->shape[3];
                for (int l = 0; l < in->shape[3]; l++) {
                    out->data[base % s + l] += in->data[base + l];
                }
            }
        }
    }
    mscal(out, 1.0f * s / tsize(in), out);
}

void ln_grad(Tensor *dX, Tensor *safevar, Tensor *X, Tensor *out) {
    for (int i = 0; i < dX->shape[0]; i++) {
        for (int j = 0; j < dX->shape[2]; j++) {
            int base = i * dX->shape[2] * dX->shape[3] + j * dX->shape[3];
            float acc = 0, acc2 = 0;
            float var = safevar->data[i * dX->shape[2] + j];
            for (int k = 0; k < dX->shape[3]; k++) {
                acc += dX->data[base + k];
                acc2 += dX->data[base + k] * X->data[base + k];
            }
            acc /= X->shape[3];
            acc2 /= X->shape[3];
            for (int k = 0; k < dX->shape[3]; k++) {
                float cur = X->data[base + k];
                float dcur = dX->data[base + k];
                out->data[base + k] = var * (dcur - acc - cur * acc2);
            }
        }
    }
}

float crossentropy(Tensor *X, Tensor *y) {
    float loss = 0;

    for (int i = 0; i < X->shape[0]; i++) {
        for (int j = 0; j < X->shape[1]; j++) {
            for (int k = 0; k < X->shape[2]; k++) {
                int base = i * X->shape[1] * X->shape[2] * X->shape[3] + j * X->shape[2] * X->shape[3] + k * X->shape[3];
                int l = -1;
                while (y->data[base + ++l] == 0);
                loss -= logf(X->data[base + l] + 1e-06);
            }
        }
    }
    return loss / (X->shape[0] * X->shape[1] * X->shape[2]);
}
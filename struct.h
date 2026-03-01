#ifndef STRUCTS_H
#define STRUCTS_H

#include <stdlib.h>

#define DIMS(t) (t)->shape[0], (t)->shape[1], (t)->shape[2], (t)->shape[3]

typedef struct {
    float *data;
    int shape[4];
} Tensor;

typedef struct { Tensor WQ, WK, WV, WO; } AttnWeights;
typedef struct { Tensor W1, W2, b1, b2; } FFWeights;
typedef struct { Tensor gamma; } RMSWeights;

typedef struct { Tensor X, Q, K, V, S, heads; } AttnActivations;
typedef struct { Tensor X, h, hg; } FFActivations;
typedef struct { Tensor safevar, xhat; } RMSActivations;

typedef struct {
    AttnWeights attn;
    FFWeights ff;
    RMSWeights rms1;
    RMSWeights rms2;
} DecoderWeights;

typedef struct {
    AttnActivations attn;
    FFActivations ff;
    RMSActivations rms1;
    RMSActivations rms2;
    Tensor res1, res2;
} DecoderActivations;

typedef struct {
    Tensor token_emb, pos_emb;
    DecoderWeights *layers;
    RMSWeights last_rms;
} Weights;

typedef struct {
    DecoderActivations *layers;
    RMSActivations last_rms;
    Tensor model_in, model_out, logits, probs;
} Activations;

typedef struct {
    char *data;
    size_t size;
    size_t off;
} Pool;

typedef struct {
    int max_batch, max_seq, nvocab, nheads, dmodel, dff, nlayers;
    float eps, learning_rate;
    Pool *pool;
} Config;

typedef enum {
    ZERO,
    ONE,
    XAVIER,
    KAIMING,
    NORMAL,
    UNIF,
    NONE,
} init_t;

void *palloc(Pool *pool, size_t n_bytes);
size_t pmark(Pool *pool);
void prollback(Pool *pool, size_t index);
Tensor *palloct(Pool *pool, int d0, int d1, int d2, int d3);

void fill_gaussian(float *buf, size_t n, float mean, float std);
void fill_xavier(float *buf, size_t n, int fan_in, int fan_out);
void fill_kaiming(float *buf, size_t n, int fan_in);
void tinit(Pool *pool, Tensor *t, int d0, int d1, int d2, int d3, init_t init_type);

size_t tsize_dims(int d0, int d1, int d2, int d3);
size_t tsize(const Tensor *t);
size_t tsizeof(const Tensor *t);

void transpose(Tensor *in, Tensor *out, int perm[4], Pool *pool);
void reshape(Tensor *t, int d0, int d1, int d2, int d3);

void split_heads(Tensor *in, Tensor *out, int nheads, Pool *pool);
void concat_heads(Tensor *in, Tensor *out, Pool *pool);
void proj_heads(Tensor *x, Tensor *y, Tensor *out, int nheads, Pool *pool);

void tprint(Tensor *in);

Tensor *palloctQ(Pool *pool, int d0, int d1, int d2, int d3);

#endif
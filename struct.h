#ifndef STRUCTS_H
#define STRUCTS_H

#include <stdlib.h>

// use e.g. palloct(config->pool, DIMS(t))
#define DIMS(t) (t)->shape[0], (t)->shape[1], (t)->shape[2], (t)->shape[3]


typedef struct {
    float *data;
    int shape[4];
} Tensor;


typedef struct { Tensor WQ, WK, WV, WO; } AttnWeights;
typedef struct { Tensor W1, W2, b1, b2; } FFWeights;
typedef struct { Tensor gamma, beta; } LNWeights;

typedef struct { Tensor X, Q, K, V, S, heads; } AttnActivations;
typedef struct { Tensor X, h, hg; } FFActivations;
typedef struct { Tensor mean, safevar, xhat; } LNActivations;


typedef struct {
    AttnWeights attn;
    FFWeights ff;
    LNWeights ln1;
    LNWeights ln2;
} DecoderWeights;

typedef struct {
    AttnActivations attn;
    FFActivations ff;
    LNActivations ln1;
    LNActivations ln2;
    Tensor res1, res2;
} DecoderActivations;


typedef struct {
    Tensor token_emb, pos_emb;
    DecoderWeights *layers;
    LNWeights last_ln;
} Weights;

typedef struct {
    Tensor model_in;
    DecoderActivations *layers;
    LNActivations last_ln;
    Tensor model_out;
    Tensor logits;
    Tensor probs;
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

void *palloc(Pool *pool, size_t n_bytes);
size_t pmark(Pool *pool);
void prollback(Pool *pool, size_t index);

Tensor *palloct(Pool *pool, int d0, int d1, int d2, int d3);
Tensor *palloctw(Pool *pool, int d0, int d1, int d2, int d3);
void tinit(Pool *pool, Tensor *t, int d0, int d1, int d2, int d3, int init_weights);

size_t tsize_dims(int d0, int d1, int d2, int d3);
size_t tsize(const Tensor *t);
size_t tsizeof(const Tensor *t);

void transpose(Tensor *in, Tensor *out, int perm[4], Pool *pool);
void reshape(Tensor *t, int d0, int d1, int d2, int d3);

void split_heads(Tensor *in, Tensor *out, int nheads, Pool *pool);
void concat_heads(Tensor *in, Tensor *out, Pool *pool);
void proj_heads(Tensor *x, Tensor *y, Tensor *out, int nheads, Pool *pool);

void tprint(Tensor *in);

#endif
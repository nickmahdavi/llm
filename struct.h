#ifndef STRUCTS_H
#define STRUCTS_H

#include <stdlib.h>

#define DIMS(t) (t)->shape[0], (t)->shape[1], (t)->shape[2], (t)->shape[3]

typedef struct {
    char *data;
    size_t size;
    size_t off;
} Pool;

Pool *pinit(char *chunk, size_t nbytes);
void *palloc(Pool *pool, size_t n_bytes);
size_t pmark(Pool *pool);
void prollback(Pool *pool, size_t index);


// Iterate through corpus.
// Go one byte at a time, 2-wide; use byte val as id.
// For each token:
//    - Link to the adjacent tokens.
// For each pair:
//    - Add the head to the hash if it's not there.
//    - Link to the adjacent pairs.
//    - Bump the count.
// Heapify.
// While vocab left:
//    - Get max.
//    - Create new pair token.
//    - For every left, check right. If there:
//        - Bump new count
//        - Decrement neighbor pair counts
//        - Unlink from adjacent tokens
//        - Unlink from pair lists
//    - Reheapify

typedef struct Token Token;
struct Token {
    size_t idx;
    Token *prev, *next;
    Token *prev_pair, *next_pair;
};

typedef struct {
    Token *head;
    Token *tail;
    uint16_t id;
    uint16_t left;
    uint16_t right;
    size_t count;
    size_t idx;
    int active;
    int heap_idx;
} TokenInfo;

typedef struct Bucket Bucket;
struct Bucket {
    uint32_t key;
    TokenInfo *val;
    Bucket *next;
};

typedef struct {
    Bucket *buckets;
    size_t size;
    size_t used;
    uint32_t (*hash)(uint32_t);
} TokenHashMap;

typedef struct {
    TokenInfo **tokens;
    size_t size;
    size_t used;
} TokenHeap;

typedef struct {
    TokenInfo **tokens;
    size_t size;
    size_t used;
    size_t active;
} Vocab;

typedef struct {
    char **tokens;
    int *lens;
    int count;
} TokenTable;

typedef struct {
    uint8_t *data;
    int len;
    int max_vocab;
    Token *BOT;
    Token *EOT;
    TokenHashMap *map;
    TokenHeap *heap;
    Vocab *vocab;
} BPE;

typedef struct {
    float *data;
    int shape[4];
} Tensor;

typedef enum {
    ZERO, ONE, XAVIER, KAIMING, NORMAL, UNIF, NONE,
} init_t;

size_t tsize_dims(int d0, int d1, int d2, int d3);
size_t tsize(const Tensor *t);
size_t tsizeof(const Tensor *t);
void tinit(Pool *pool, Tensor *t, int d0, int d1, int d2, int d3, init_t init_type);
void tprint(Tensor *in);
Tensor *palloct(Pool *pool, int d0, int d1, int d2, int d3);

void fill(void *buf, void *val, size_t n, size_t val_size);
void fill_gaussian(float *buf, size_t n, float mean, float std);
void fill_xavier(float *buf, size_t n, int fan_in, int fan_out);
void fill_kaiming(float *buf, size_t n, int fan_in);

void transpose(Tensor *in, Tensor *out, int perm[4], Pool *pool);
void reshape(Tensor *t, int d0, int d1, int d2, int d3);

void split_heads(Tensor *in, Tensor *out, int nheads, Pool *pool);
void concat_heads(Tensor *in, Tensor *out, Pool *pool);
void proj_heads(Tensor *x, Tensor *y, Tensor *out, int nheads, Pool *pool);

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
    int max_batch, max_seq, nvocab, nheads, dmodel, dff, nlayers;
    float eps, learning_rate;
    Pool *pool;
} Config;

#endif
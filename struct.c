#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include "struct.h"
#include "ops.h"

Pool *pinit(char *chunk, size_t nbytes) {
    if (nbytes < sizeof(Pool) + 63) return NULL;
    Pool *pool = (Pool *)chunk;
    uintptr_t base = (uintptr_t)(chunk + sizeof(Pool));
    uintptr_t aligned = (base + 63) & ~((uintptr_t)63);
    pool->data = (char *)aligned;
    pool->size = nbytes - (pool->data - chunk);
    pool->off = 0;
    pool->base = chunk;
    return pool;
}

void *palloc(Pool *pool, size_t n_bytes) {
    uintptr_t base = (uintptr_t)pool->data;
    uintptr_t aligned = (uintptr_t)(base + pool->off + 15) & ~((uintptr_t)15);
    pool->off = (size_t)(aligned - base) + n_bytes;
    assert(pool->off < pool->size && "pool overflow");
    return (void *)aligned;
}

size_t pmark(Pool *pool) {
    return pool->off;
}

void prollback(Pool *pool, size_t idx) {
    if (idx > pool->off) return;
    pool->off = idx;
}

Tensor *palloct(Pool *pool, int d0, int d1, int d2, int d3) {
    Tensor *tensor = (Tensor *) palloc(pool, sizeof(Tensor));
    tinit(pool, tensor, d0, d1, d2, d3, NONE);
    return tensor;
}

void fill(void *buf, void *val, size_t n, size_t val_size) {
    char *p = (char *)buf;
    char *q = (char *)val;
    char *cur = p;

    size_t buf_size = n * val_size;
    size_t min = round_up_pow2(256 / val_size) * val_size;
    size_t chunk_size = min < buf_size ? min : buf_size;

    if (buf_size == 0 || val_size == 0) return;

    for (; cur < p + chunk_size; cur += val_size) memcpy(cur, q, val_size);

    while (cur + chunk_size < p + buf_size) {
        memcpy(cur, p, chunk_size);
        cur += chunk_size;
        chunk_size <<= 1;
    }

    memcpy(cur, p, p + buf_size - cur);
}

void fill_gaussian(float *buf, size_t n, float mean, float std) {
    for (size_t i = 0; i + 1 < n; i += 2) {
        float u1 = 0.0f, u2 = 0.0f;
        while (u1 == 0.0f) u1 = (float)drand48();
        while (u2 == 0.0f) u2 = (float)drand48();
        float r = sqrt(-2.0 * log(u1));
        buf[i] = mean + std * r * cos(2.0 * M_PI * u2);
        buf[i + 1] = mean + std * r * sin(2.0 * M_PI * u2);
    }
    if (n % 2) {
        float u1 = (float)drand48();
        float u2 = (float)drand48();
        buf[n - 1] = mean + std * sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
    }
}

void fill_xavier(float *buf, size_t n, int fan_in, int fan_out) {
    float limit = 2.0f * sqrtf(6.0f / (fan_in + fan_out));
    for (size_t i = 0; i < n; i++) buf[i] = (drand48() - 0.5f) * limit;
}

void fill_kaiming(float *buf, size_t n, int fan_in) {
    float limit = sqrtf(6.0f / fan_in);
    for (size_t i = 0; i < n; i++) buf[i] = (drand48() - 0.5f) * limit;
}

void tinit(Pool *pool, Tensor *t, int d0, int d1, int d2, int d3, init_t init_type) {
    size_t size = (size_t) d0 * d1 * d2 * d3;
    t->data = (float *)palloc(pool, sizeof(float) * size);
    t->shape[0] = d0;
    t->shape[1] = d1;
    t->shape[2] = d2;
    t->shape[3] = d3;

    switch (init_type) {
        case ZERO:
            memset(t->data, 0, sizeof(float) * size);
            break;
        case ONE:
            fill(t->data, (float[]){1.0f}, size, sizeof(float));
            break;
        case XAVIER:
            fill_xavier(t->data, size, d2, d3);
            break;
        case KAIMING:
            fill_kaiming(t->data, size, d2);
            break;
        case NORMAL:
            fill_gaussian(t->data, size, 0, 0.01);
            break;
        case NONE:
            break;
        default:
            break;
    }
}

size_t tsize_dims(int d0, int d1, int d2, int d3) {
    return d0 * d1 * d2 * d3;
}

size_t tsize(const Tensor *t) {
    return tsize_dims(DIMS(t));
}

size_t tsizeof(const Tensor *t) {
    return sizeof(float) * tsize(t);
}

void reshape(Tensor *t, int d0, int d1, int d2, int d3) {
    if (d0 >= 0) t->shape[0] = d0;
    if (d1 >= 0) t->shape[1] = d1;
    if (d2 >= 0) t->shape[2] = d2;
    if (d3 >= 0) t->shape[3] = d3;
}

void transpose(Tensor *in, Tensor *out, int perm[4], Pool *pool) {
    size_t off = pmark(pool);
    float *dst = (out == in) ? palloc(pool, tsizeof(in)) : out->data;

    int s[4] = { DIMS(in) };

    int str[3] = {s[1] * s[2] * s[3], s[2] * s[3], s[3]};
    int nstr[3] = {s[perm[1]] * s[perm[2]] * s[perm[3]], s[perm[2]] * s[perm[3]], s[perm[3]]};

    int idx[4];

    for (idx[0] = 0; idx[0] < s[0]; idx[0]++) {
        for (idx[1] = 0; idx[1] < s[1]; idx[1]++) {
            for (idx[2] = 0; idx[2] < s[2]; idx[2]++) {
                for (idx[3] = 0; idx[3] < s[3]; idx[3]++) {
                    int old_i = idx[0] * str[0] + idx[1] * str[1] + idx[2] * str[2] + idx[3];
                    int new_i = idx[perm[0]] * nstr[0] + idx[perm[1]] * nstr[1] + idx[perm[2]] * nstr[2] + idx[perm[3]];
                    dst[new_i] = in->data[old_i];
                }
            }
        }
    }
    reshape(out, s[perm[0]], s[perm[1]], s[perm[2]], s[perm[3]]);
    if (out == in) {
        memcpy(in->data, dst, tsizeof(in));
        prollback(pool, off);
    }
}

void split_heads(Tensor *in, Tensor *out, int nheads, Pool *pool) {
    int seq = in->shape[2], dmodel = in->shape[3];
    reshape(in, -1, in->shape[2], nheads, in->shape[3] / nheads);
    transpose(in, out, (int[]){0, 2, 1, 3}, pool);
    if (in != out) reshape(in, -1, 1, seq, dmodel);
}

void proj_heads(Tensor *x, Tensor *y, Tensor *out, int nheads, Pool *pool) {
    size_t off = pmark(pool);
    Tensor *tmp = palloct(pool, x->shape[0], x->shape[1], x->shape[2], y->shape[3]);
    matmul(x, y, tmp);
    split_heads(tmp, out, nheads, pool);
    prollback(pool, off);
}

void concat_heads(Tensor *in, Tensor *out, Pool *pool) {
    transpose(in, out, (int[]) {0, 2, 1, 3}, pool);
    reshape(out, -1, 1, out->shape[1], out->shape[2] * out->shape[3]);
}
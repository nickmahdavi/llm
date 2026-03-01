#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "struct.h"
#include "ops.h"

/*
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
} */

void *palloc(Pool *pool, size_t n_bytes) {
    /* I'll figure this out later -- for now just assume allocated enough space
    if (pool->off + n_bytes > pool->size) {
        size_t new_size = round_up_pow2(pool->off + n_bytes);
        pool->data = realloc(pool->data, new_size);
        pool->size = new_size;
    } */
    void *ptr = pool->data + pool->off;
    pool->off += n_bytes;
    return ptr;
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
    tinit(pool, tensor, d0, d1, d2, d3, 0);
    return tensor;
}

Tensor *palloctw(Pool *pool, int d0, int d1, int d2, int d3) {
    Tensor *tensor = (Tensor *) palloc(pool, sizeof(Tensor));
    tinit(pool, tensor, d0, d1, d2, d3, 1);
    return tensor;
}

void fill_randn(float *buf, size_t n, float mean, float std) {
    for (size_t i = 0; i + 1 < n; i += 2) {
        float u1 = (float)drand48();
        float u2 = (float)drand48();
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

void tinit(Pool *pool, Tensor *t, int d0, int d1, int d2, int d3, int init_weights) {
    size_t size = (size_t) d0 * d1 * d2 * d3;
    t->data = (float *)palloc(pool, sizeof(float) * size);
    t->shape[0] = d0;
    t->shape[1] = d1;
    t->shape[2] = d2;
    t->shape[3] = d3;
    if (init_weights) {
        fill_randn(t->data, tsize(t), 0, 0.01);
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

void tprint(Tensor *in) {
    for (int i = 0; i < tsize(in); i++) {
        printf("%f ", in->data[i]);
    }
    printf("\n\n");
}
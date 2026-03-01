#include <stdlib.h>
#include <math.h>
#include "struct.h"
#include "ops.h"

void embedding(Tensor *in, Tensor *out, Tensor *WE, Tensor *WP) {
    matmul(in, WE, out);
    madd(out, WP, out);
}

void attention(Tensor *in, Tensor *out, AttnWeights *weights, AttnActivations *acts, Config *config) {
    proj_heads(in, &weights->WQ, &acts->Q, config->nheads, config->pool);
    proj_heads(in, &weights->WK, &acts->K, config->nheads, config->pool);
    proj_heads(in, &weights->WV, &acts->V, config->nheads, config->pool);

    matmul_bt(&acts->Q, &acts->K, &acts->S);
    mscal(&acts->S, 1.0f / sqrtf((float)config->dmodel / (float)config->nheads), &acts->S);
    triu_mask(&acts->S, &acts->S, -INFINITY);
    softmax(&acts->S, &acts->S);

    reshape(&acts->heads, DIMS(&acts->Q));
    matmul(&acts->S, &acts->V, &acts->heads);
    concat_heads(&acts->heads, &acts->heads, config->pool);
    matmul(&acts->heads, &weights->WO, out);
}

void ffn(Tensor *in, Tensor *out, FFWeights *weights, FFActivations *acts, Config *config) {
    matmul(in, &weights->W1, &acts->h);
    madd(&acts->h, &weights->b1, &acts->h);
    gelu(&acts->h, &acts->hg);
    matmul(&acts->hg, &weights->W2, out);
    madd(out, &weights->b2, out);
}

void layernorm(Tensor *in, Tensor *out, LNWeights *weights, LNActivations *acts, Config *config) {
    lnstats(in, &acts->mean, &acts->safevar, config->eps);
    msub(in, &acts->mean, &acts->xhat);
    mmult(&acts->xhat, &acts->safevar, &acts->xhat);
    mmult(&acts->xhat, &weights->gamma, out);
    madd(out, &weights->beta, out);
}

void decoder(Tensor **in, DecoderWeights *weights, DecoderActivations *acts, Config *config) {
    layernorm(*in, &acts->attn.X, &weights->ln1, &acts->ln1, config);
    attention(&acts->attn.X, &acts->res1, &weights->attn, &acts->attn, config);
    madd(*in, &acts->res1, &acts->res1);
    layernorm(&acts->res1, &acts->ff.X, &weights->ln2, &acts->ln2, config);
    ffn(&acts->ff.X, &acts->res2, &weights->ff, &acts->ff, config);
    madd(&acts->res1, &acts->res2, &acts->res2);
    *in = &acts->res2;
}

void unembedding(Tensor *in, Tensor *out, Tensor *WE, Config *config) {
    size_t off = pmark(config->pool);
    Tensor *WU = palloct(config->pool, DIMS(WE));
    transpose(WE, WU, (int[]){0, 1, 3, 2}, config->pool);
    matmul(in, WU, out);
    prollback(config->pool, off);
}

void forward(Tensor *in, Weights *weights, Activations *acts, Config *config) {
    Tensor *ptr = &acts->model_in;
    embedding(in, ptr, &weights->token_emb, &weights->pos_emb);
    for (int i = 0; i < config->nlayers; i++) {
        DecoderWeights *d_weights = &weights->layers[i];
        DecoderActivations *d_acts = &acts->layers[i];
        decoder(&ptr, d_weights, d_acts, config);
    }
    layernorm(ptr, &acts->model_out, &weights->last_ln, &acts->last_ln, config);
    unembedding(&acts->model_out, &acts->logits, &weights->token_emb, config);
    softmax(&acts->logits, &acts->probs);
}
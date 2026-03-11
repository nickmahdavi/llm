#include <stdio.h>
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

void rmsnorm(Tensor *in, Tensor *out, RMSWeights *weights, RMSActivations *acts, Config *config) {
    rms(in, &acts->safevar, &acts->xhat, config->eps);
    mmult(&acts->xhat, &weights->gamma, out);
}

void decoder(Tensor **in, DecoderWeights *weights, DecoderActivations *acts, Config *config) {
    rmsnorm(*in, &acts->attn.X, &weights->rms1, &acts->rms1, config);
    attention(&acts->attn.X, &acts->res1, &weights->attn, &acts->attn, config);
    madd(*in, &acts->res1, &acts->res1);
    rmsnorm(&acts->res1, &acts->ff.X, &weights->rms2, &acts->rms2, config);
    ffn(&acts->ff.X, &acts->res2, &weights->ff, &acts->ff, config);
    madd(&acts->res1, &acts->res2, &acts->res2);
    *in = &acts->res2;
}

void unembedding(Tensor *in, Tensor *out, Tensor *WE, Config *config) {
    matmul(in, WE, out);
}

void forward_from(int layer, Tensor *in, Weights *weights, Activations *acts, Config *config) {
    Tensor *ptr;

    if (layer == 0 || layer == 1) {
        ptr = &acts->model_in;
        embedding(in, ptr, &weights->token_emb, &weights->pos_emb);
        layer = 1;
    } else if (layer <= config->nlayers + 1) {
        ptr = &acts->layers[layer - 2].res2;
    }

    for (; layer <= config->nlayers; layer++) {
        DecoderWeights *d_weights = &weights->layers[layer - 1];
        DecoderActivations *d_acts = &acts->layers[layer - 1];
        decoder(&ptr, d_weights, d_acts, config);
    }

    if (layer == config->nlayers + 1) {
        rmsnorm(ptr, &acts->model_out, &weights->last_rms, &acts->last_rms, config);
        layer++;
    }

    if (layer == config->nlayers + 2) {
        unembedding(&acts->model_out, &acts->logits, &weights->token_unemb, config);
        layer++;
    }

    if (layer == config->nlayers + 3) {
        softmax(&acts->logits, &acts->probs);
    }
}

void forward(Tensor *in, Weights *weights, Activations *acts, Config *config) {
    forward_from(0, in, weights, acts, config);
}
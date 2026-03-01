#include <string.h>
#include <math.h>
#include "struct.h"
#include "ops.h"

void logits_back(Tensor *logits, Tensor *labels, Tensor *grad_out) {
    msub(logits, labels, grad_out);
}

void unembedding_back(Tensor *grad_in, Tensor *grad_out, Tensor *X, Tensor *WU, Tensor *WU_grad, Config *config) {
    size_t off = pmark(config->pool);
    Tensor *grad_acc_WU = palloct(config->pool, X->shape[0], WU->shape[1], WU->shape[2], WU->shape[3]);

    matmul_at(X, grad_in, grad_acc_WU);
    batch_mean(grad_acc_WU, WU_grad);
    matmul_bt(grad_in, WU, grad_out);

    prollback(config->pool, off);
}

void ff_back(Tensor *grad_in, Tensor *grad_out, FFActivations *acts, FFWeights *weights, FFWeights *updates, Config *config) {
    size_t off = pmark(config->pool);

    Tensor *dG = palloct(config->pool, DIMS(&acts->hg));
    #define BDIMS(t) acts->h.shape[0], (t)->shape[1], (t)->shape[2], (t)->shape[3]
    Tensor *grad_acc_W1 = palloct(config->pool, BDIMS(&weights->W1));
    Tensor *grad_acc_W2 = palloct(config->pool, BDIMS(&weights->W2));
    Tensor *grad_acc_b1 = palloct(config->pool, BDIMS(&weights->b1));
    #undef BDIMS

    matmul_at(&acts->hg, grad_in, grad_acc_W2);
    matmul_bt(grad_in, &weights->W2, dG);
    gelu_grad(dG, &acts->h, grad_acc_b1);
    matmul_at(&acts->X, grad_acc_b1, grad_acc_W1);
    matmul_bt(grad_acc_b1, &weights->W1, grad_out);

    batch_mean(grad_in, &updates->b2);
    batch_mean(grad_acc_W2, &updates->W2);
    batch_mean(grad_acc_b1, &updates->b1);
    batch_mean(grad_acc_W1, &updates->W1);

    prollback(config->pool, off);
}

void attention_back(Tensor *grad_in, Tensor *grad_out, AttnActivations *acts, AttnWeights *weights, AttnWeights *updates, Config *config) {
    size_t off = pmark(config->pool);

    Tensor *dA = palloct(config->pool, DIMS(grad_in));
    Tensor *dQ = palloct(config->pool, DIMS(&acts->Q));
    Tensor *dK = palloct(config->pool, DIMS(&acts->K));
    Tensor *dV = palloct(config->pool, DIMS(&acts->V));
    Tensor *dS = palloct(config->pool, DIMS(&acts->S));
    Tensor *dXQ = palloct(config->pool, DIMS(&acts->X));
    Tensor *dXK = palloct(config->pool, DIMS(&acts->X));
    Tensor *dXV = palloct(config->pool, DIMS(&acts->X));

    #define BDIMS(t) acts->X.shape[0], (t)->shape[1], (t)->shape[2], (t)->shape[3]
    Tensor *grad_acc_WO = palloct(config->pool, BDIMS(&weights->WO));
    Tensor *grad_acc_WQ = palloct(config->pool, BDIMS(&weights->WQ));
    Tensor *grad_acc_WK = palloct(config->pool, BDIMS(&weights->WK));
    Tensor *grad_acc_WV = palloct(config->pool, BDIMS(&weights->WV));
    #undef BDIMS

    matmul_bt(grad_in, &weights->WO, dA);
    split_heads(dA, dA, config->nheads, config->pool);
    
    matmul_bt(dA, &acts->V, dS);
    soft_grad(dS, &acts->S, dS);
    mscal(dS, 1.0f / sqrtf((float)config->dmodel / (float)config->nheads), dS);
    triu_mask(dS, dS, 0.0f);

    matmul(dS, &acts->K, dQ);
    matmul_at(dS, &acts->Q, dK);
    matmul_at(&acts->S, dA, dV);

    concat_heads(dQ, dQ, config->pool);
    concat_heads(dK, dK, config->pool);
    concat_heads(dV, dV, config->pool);

    matmul_bt(dQ, &weights->WQ, dXQ);
    matmul_bt(dK, &weights->WK, dXK);
    matmul_bt(dV, &weights->WV, dXV);

    matmul_at(&acts->heads, grad_in, grad_acc_WO);
    matmul_at(&acts->X, dQ, grad_acc_WQ);
    matmul_at(&acts->X, dK, grad_acc_WK);
    matmul_at(&acts->X, dV, grad_acc_WV);

    batch_mean(grad_acc_WO, &updates->WO);
    batch_mean(grad_acc_WQ, &updates->WQ);
    batch_mean(grad_acc_WK, &updates->WK);
    batch_mean(grad_acc_WV, &updates->WV);

    madd(dXQ, dXK, grad_out);
    madd(grad_out, dXV, grad_out);

    prollback(config->pool, off);
}

void ln_back(Tensor *grad_in, Tensor *grad_out, LNActivations *acts, LNWeights *weights, LNWeights *updates, Config *config) {
    size_t off = pmark(config->pool);

    Tensor *grad_acc_gamma = palloct(config->pool, acts->xhat.shape[0], 1, acts->xhat.shape[2], weights->gamma.shape[3]);
    Tensor *dXhat = palloct(config->pool, DIMS(&acts->xhat));

    mmult(&acts->xhat, grad_in, grad_acc_gamma);
    mmult(grad_in, &weights->gamma, dXhat);
    ln_grad(dXhat, &acts->safevar, &acts->xhat, grad_out);

    batch_mean(grad_acc_gamma, &updates->gamma);
    batch_mean(grad_in, &updates->beta);

    prollback(config->pool, off);
}

void decoder_back(Tensor *grad_in, Tensor *grad_out, DecoderActivations *acts, DecoderWeights *weights, DecoderWeights *updates, Config *config) {
    size_t off = pmark(config->pool);
    
    Tensor *t1 = palloct(config->pool, DIMS(grad_in));
    Tensor *t2 = palloct(config->pool, DIMS(grad_in));

    ff_back(grad_in, t1, &acts->ff, &weights->ff, &updates->ff, config);
    ln_back(t1, t2, &acts->ln2, &weights->ln2, &updates->ln2, config);
    madd(grad_in, t2, grad_in);

    attention_back(grad_in, t1, &acts->attn, &weights->attn, &updates->attn, config);
    ln_back(t1, t2, &acts->ln1, &weights->ln1, &updates->ln1, config);
    madd(grad_in, t2, grad_out);

    prollback(config->pool, off);
}

void embedding_back(Tensor *grad_in, Tensor *X, Tensor *WE_grad, Tensor *WP_grad, Config *config) {
    size_t off = pmark(config->pool);

    Tensor *grad_acc_WE = palloct(config->pool, X->shape[0], WE_grad->shape[1], WE_grad->shape[2], WE_grad->shape[3]);

    matmul_at(X, grad_in, grad_acc_WE);
    batch_mean(grad_acc_WE, WE_grad);
    batch_mean(grad_in, WP_grad);

    prollback(config->pool, off);
}

void backpropagate(Tensor *in, Tensor *labels, Activations *acts, Weights *weights, Weights *grad, Config *config) {
    size_t off = pmark(config->pool);

    float eta = config->learning_rate;

    Tensor *t0 = palloct(config->pool, in->shape[0], 1, in->shape[2], config->nvocab);
    Tensor *t1 = palloct(config->pool, in->shape[0], 1, in->shape[2], config->dmodel);
    Tensor *t2 = palloct(config->pool, in->shape[0], 1, in->shape[2], config->dmodel);

    logits_back(&acts->probs, labels, t0);

    Tensor *WU = palloct(config->pool, DIMS(&weights->token_emb));
    transpose(&weights->token_emb, WU, (int[]){0, 1, 3, 2}, config->pool);
    Tensor *dWU = palloct(config->pool, DIMS(WU));
    unembedding_back(t0, t1, &acts->model_out, WU, dWU, config);
    transpose(dWU, dWU, (int[]){0, 1, 3, 2}, config->pool);
    step(&weights->token_emb, dWU, eta);

    ln_back(t1, t2, &acts->last_ln, &weights->last_ln, &grad->last_ln, config);
    step(&weights->last_ln.beta, &grad->last_ln.beta, eta);
    step(&weights->last_ln.gamma, &grad->last_ln.gamma, eta);

    Tensor **grad_in = &t2;
    Tensor **grad_out = &t1;
    Tensor **tmp;

    for (int i = config->nlayers - 1; i >= 0; i--) {
        DecoderActivations *d_acts = &acts->layers[i];
        DecoderWeights *d_weights = &weights->layers[i];
        DecoderWeights *d_grad = &grad->layers[i];

        decoder_back(*grad_in, *grad_out, d_acts, d_weights, d_grad, config);

        #define STEP(fld, t) step(&d_weights->fld.t, &d_grad->fld.t, eta)
        STEP(attn, WQ);
        STEP(attn, WK);
        STEP(attn, WV);
        STEP(attn, WO);
        STEP(ff, b1);
        STEP(ff, b2);
        STEP(ff, W1);
        STEP(ff, W2);
        STEP(ln1, beta);
        STEP(ln1, gamma);
        STEP(ln2, beta);
        STEP(ln2, gamma);
        #undef STEP

        tmp = grad_in;
        grad_in = grad_out;
        grad_out = tmp;
    }

    embedding_back(*grad_in, in, &grad->token_emb, &grad->pos_emb, config);
    step(&weights->pos_emb, &grad->pos_emb, eta);
    step(&weights->token_emb, &grad->token_emb, eta);

    prollback(config->pool, off);
}
#include <string.h>
#include <math.h>
#include "struct.h"
#include "ops.h"

#define BS (grad_in->shape[0] * grad_in->shape[2])

void logits_back(Tensor *logits, Tensor *labels, Tensor *grad_out) {
    msub(logits, labels, grad_out);
}

void unembedding_back(Tensor *grad_in, Tensor *grad_out, Tensor *X, Tensor *WU, Tensor *WU_grad, Config *config) {
    size_t off = pmark(config->pool);
    Tensor *grad_acc_WU = palloct(config->pool, X->shape[0], WU->shape[1], WU->shape[2], WU->shape[3]);

    matmul_at(X, grad_in, grad_acc_WU);
    batch_mean(grad_acc_WU, WU_grad, BS);
    matmul_bt(grad_in, WU, grad_out);

    prollback(config->pool, off);
}

void ff_back(Tensor *grad_in, Tensor *grad_out, FFActivations *acts, FFWeights *weights, FFWeights *updates, Config *config) {
    size_t off = pmark(config->pool);

    Tensor *dG = palloct(config->pool, DIMS(&acts->hg));
    Tensor *dH = palloct(config->pool, DIMS(&acts->h));

    #define BDIMS(t) acts->h.shape[0], (t)->shape[1], (t)->shape[2], (t)->shape[3]
    Tensor *grad_acc_W1 = palloct(config->pool, BDIMS(&weights->W1));
    Tensor *grad_acc_W2 = palloct(config->pool, BDIMS(&weights->W2));
    #undef BDIMS

    matmul_at(&acts->hg, grad_in, grad_acc_W2);
    matmul_bt(grad_in, &weights->W2, dG);
    gelu_grad(dG, &acts->h, dH);
    matmul_at(&acts->X, dH, grad_acc_W1);
    matmul_bt(dH, &weights->W1, grad_out);

    batch_mean(grad_in, &updates->b2, BS);
    batch_mean(grad_acc_W2, &updates->W2, BS);
    batch_mean(dH, &updates->b1, BS);
    batch_mean(grad_acc_W1, &updates->W1, BS);

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

    batch_mean(grad_acc_WO, &updates->WO, BS);
    batch_mean(grad_acc_WQ, &updates->WQ, BS);
    batch_mean(grad_acc_WK, &updates->WK, BS);
    batch_mean(grad_acc_WV, &updates->WV, BS);

    madd(dXQ, dXK, grad_out);
    madd(grad_out, dXV, grad_out);

    prollback(config->pool, off);
}

void rms_back(Tensor *grad_in, Tensor *grad_out, RMSActivations *acts, RMSWeights *weights, RMSWeights *updates, Config *config) {
    size_t off = pmark(config->pool);

    Tensor *grad_acc_gamma = palloct(config->pool, acts->xhat.shape[0], 1, acts->xhat.shape[2], weights->gamma.shape[3]);
    Tensor *dXhat = palloct(config->pool, DIMS(&acts->xhat));

    mmult(&acts->xhat, grad_in, grad_acc_gamma);
    mmult(grad_in, &weights->gamma, dXhat);
    rms_grad(dXhat, &acts->safevar, &acts->xhat, grad_out);

    batch_mean(grad_acc_gamma, &updates->gamma, BS);

    prollback(config->pool, off);
}

void decoder_back(Tensor *grad_in, Tensor *grad_out, DecoderActivations *acts, DecoderWeights *weights, DecoderWeights *updates, Config *config) {
    size_t off = pmark(config->pool);
    
    Tensor *t1 = palloct(config->pool, DIMS(grad_in));
    Tensor *t2 = palloct(config->pool, DIMS(grad_in));
    Tensor *res_grad = palloct(config->pool, DIMS(grad_in));

    ff_back(grad_in, t1, &acts->ff, &weights->ff, &updates->ff, config);
    rms_back(t1, t2, &acts->rms2, &weights->rms2, &updates->rms2, config);
    madd(grad_in, t2, res_grad);

    attention_back(res_grad, t1, &acts->attn, &weights->attn, &updates->attn, config);
    rms_back(t1, t2, &acts->rms1, &weights->rms1, &updates->rms1, config);
    madd(res_grad, t2, grad_out);

    prollback(config->pool, off);
}

void embedding_back(Tensor *grad_in, Tensor *X, Tensor *WE_grad, Tensor *WP_grad, Config *config) {
    size_t off = pmark(config->pool);

    Tensor *grad_acc_WE = palloct(config->pool, X->shape[0], WE_grad->shape[1], WE_grad->shape[2], WE_grad->shape[3]);

    matmul_at(X, grad_in, grad_acc_WE);
    batch_mean(grad_acc_WE, WE_grad, BS);
    batch_mean(grad_in, WP_grad, BS);

    prollback(config->pool, off);
}

void backpropagate(Tensor *in, Tensor *labels, Activations *acts, Weights *weights, Weights *grad, Config *config) {
    size_t off = pmark(config->pool);

    Tensor *t0 = palloct(config->pool, in->shape[0], 1, in->shape[2], config->nvocab);
    Tensor *t1 = palloct(config->pool, in->shape[0], 1, in->shape[2], config->dmodel);
    Tensor *t2 = palloct(config->pool, in->shape[0], 1, in->shape[2], config->dmodel);

    logits_back(&acts->probs, labels, t0);

    unembedding_back(t0, t1, &acts->model_out, &weights->token_unemb, &grad->token_unemb, config);

    rms_back(t1, t2, &acts->last_rms, &weights->last_rms, &grad->last_rms, config);

    Tensor **grad_in = &t2;
    Tensor **grad_out = &t1;
    Tensor **tmp;

    for (int i = config->nlayers - 1; i >= 0; i--) {
        DecoderActivations *d_acts = &acts->layers[i];
        DecoderWeights *d_weights = &weights->layers[i];
        DecoderWeights *d_grad = &grad->layers[i];

        decoder_back(*grad_in, *grad_out, d_acts, d_weights, d_grad, config);

        tmp = grad_in;
        grad_in = grad_out;
        grad_out = tmp;
    }

    embedding_back(*grad_in, in, &grad->token_emb, &grad->pos_emb, config);

    prollback(config->pool, off);
}

void adamw(int t, int batch, Weights *weights, Weights *grad, Weights *m, Weights *v, Config *config) {
    float eta = cosine_lr(t, config->nwarmup, config->ndecay, config->eta_max, config->eta_min) / batch;
    float scale = fminf(1.0f, config->max_norm / grad_norm(grad, config->nlayers));
    float b1t = powf(config->beta1, (float)t);
    float b2t = powf(config->beta2, (float)t);

    #define STEP(fld) step_adamw(&weights->fld, &grad->fld, &m->fld, &v->fld, config->beta1, config->beta2, b1t, b2t, config->lambda, eta, config->eps, scale)
    FOR_WEIGHTS(STEP, config->nlayers);
    #undef STEP
}

void zero_grad(Weights *grad, Config *config) {
    #define ZERO(fld) for (int i = 0; i < tsize(&grad->fld); i++) grad->fld.data[i] = 0
    FOR_WEIGHTS(ZERO, config->nlayers);
    #undef ZERO
}
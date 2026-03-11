#include "struct.h"

#define b config->max_batch
#define nh config->nheads
#define s config->max_seq
#define d config->dmodel
#define v config->nvocab
#define dk (config->dmodel / config->nheads)
#define df config->dff
#define INIT_A(arg_a, arg_b, arg_c, arg_d, arg_e) tinit(config->pool, &acts->arg_a, arg_b, arg_c, arg_d, arg_e, NONE)
#define INIT_W(arg_a, arg_b, arg_c, arg_d, arg_e, arg_f) tinit(config->pool, &weights->arg_a, arg_b, arg_c, arg_d, arg_e, arg_f)

void init_attn_acts(AttnActivations *acts, Config *config) {
    INIT_A(X, b, 1, s, d);
    INIT_A(Q, b, nh, s, dk);
    INIT_A(K, b, nh, s, dk);
    INIT_A(V, b, nh, s, dk);
    INIT_A(S, b, nh, s, s);
    INIT_A(heads, b, 1, s, d);
}

void init_attn_weights(AttnWeights *weights, Config *config) {
    INIT_W(WQ, 1, 1, d, d, XAVIER);
    INIT_W(WK, 1, 1, d, d, XAVIER);
    INIT_W(WV, 1, 1, d, d, XAVIER);
    INIT_W(WO, 1, 1, d, d, XAVIER);
}

void init_ff_acts(FFActivations *acts, Config *config) {
    INIT_A(X, b, 1, s, d);
    INIT_A(h, b, 1, s, df);
    INIT_A(hg, b, 1, s, df);
}

void init_ff_weights(FFWeights *weights, Config *config) {
    INIT_W(W1, 1, 1, d, df, KAIMING);
    INIT_W(W2, 1, 1, df, d, KAIMING);
    INIT_W(b1, 1, 1, 1, df, ZERO);
    INIT_W(b2, 1, 1, 1, d, ZERO);
}

void init_rms_acts(RMSActivations *acts, Config *config) {
    INIT_A(safevar, b, 1, 1, s);
    INIT_A(xhat, b, 1, s, d);
}

void init_rms_weights(RMSWeights *weights, Config *config) {
    INIT_W(gamma, 1, 1, 1, d, ONE);
}

void init_decoder_acts(DecoderActivations *acts, Config *config) {
    init_attn_acts(&acts->attn, config);
    init_ff_acts(&acts->ff, config);
    init_rms_acts(&acts->rms1, config);
    init_rms_acts(&acts->rms2, config);
    INIT_A(res1, b, 1, s, d);
    INIT_A(res2, b, 1, s, d);
}

void init_decoder_weights(DecoderWeights *weights, Config *config) {
    init_attn_weights(&weights->attn, config);
    init_ff_weights(&weights->ff, config);
    init_rms_weights(&weights->rms1, config);
    init_rms_weights(&weights->rms2, config);
}

Activations *init_acts(Config *config) {
    Activations *acts = palloc(config->pool, sizeof(Activations));
    acts->layers = palloc(config->pool, config->nlayers * sizeof(DecoderActivations));
    INIT_A(model_in, b, 1, s, d);
    for (int i = 0; i < config->nlayers; i++) {
        DecoderActivations *d_acts = &acts->layers[i];
        init_decoder_acts(d_acts, config);
    }
    init_rms_acts(&acts->last_rms, config);
    INIT_A(model_out, b, 1, s, d);
    INIT_A(logits, b, 1, s, v);
    INIT_A(probs, b, 1, s, v);

    return acts;
}

Weights *init_weights(Config *config) {
    Weights *weights = palloc(config->pool, sizeof(Weights));
    weights->layers = palloc(config->pool, config->nlayers * sizeof(DecoderWeights));

    INIT_W(token_emb, 1, 1, v, d, NORMAL);
    INIT_W(pos_emb, 1, 1, s, d, NORMAL);
    INIT_W(token_unemb, 1, 1, d, v, NORMAL);
    for (int i = 0; i < config->nlayers; i++) {
        DecoderWeights *d_weights = &weights->layers[i];
        init_decoder_weights(d_weights, config);
    }
    init_rms_weights(&weights->last_rms, config);
    return weights;
}

Weights *init_gradient_checker(int n, Config *config) {
    Weights *weights = palloc(config->pool, sizeof(Weights));
    weights->layers = palloc(config->pool, config->nlayers * sizeof(DecoderWeights));

    #define GC(fld) INIT_W(fld, 1, 1, n, 3, NONE)
    FOR_WEIGHTS(GC, config->nlayers);
    #undef GC

    return weights;
}

#undef b
#undef nh
#undef s
#undef d
#undef v
#undef dk
#undef df
#undef INIT_A
#undef INIT_W
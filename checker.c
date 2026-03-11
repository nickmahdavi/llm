#include <string.h>
#include <math.h>
#include "layers.h"
#include "struct.h"
#include "grad.h"
#include "init.h"
#include "ops.h"

#define FD_EPS 0.005f
#define PARAMS_CHECK 3

void get_perturbations(size_t max, size_t *indices, int n) {
    for (int i = 0; i < n; i++) {
        retry:
        indices[i] = (size_t)random() % max;
        for (int j = 0; j < i; j++) {
            if (indices[j] == indices[i]) goto retry;
        }
    }
}

void test_gradient(Tensor *results, int layer, Tensor *in, Tensor *w, Tensor *wgrad, Tensor *labels, Activations *acts, Weights *weights, Config *config) {
    // w should point into weights
    int n = results->shape[2];
    size_t indices[n];
    get_perturbations(tsize(w), indices, n);
    for (int i = 0; i < n; i++) {
        float s = w->data[indices[i]];
        w->data[indices[i]] = s + FD_EPS;
        forward_from(layer, in, weights, acts, config);
        float a = crossentropy(&acts->probs, labels);
        w->data[indices[i]] = s - FD_EPS;
        forward_from(layer, in, weights, acts, config);
        a -= crossentropy(&acts->probs, labels);
        a /= 2 * FD_EPS;
        w->data[indices[i]] = s;
        float b = wgrad->data[indices[i]];
        float c = fabsf(a - b) / fmaxf(fabsf(a) + fabsf(b), config->eps);
        results->data[3 * i] = a;
        results->data[3 * i + 1] = b;
        results->data[3 * i + 2] = c;
    }
}

void check_gradients(Weights *results, Tensor *in, Tensor *labels, Activations *acts, Weights *weights, Weights *grad, Config *config) {
    #define TEST(l, fld) test_gradient(&results->fld, l, in, &weights->fld, &grad->fld, labels, acts, weights, config)

    // results should be (config->nlayers + 2) x n x 3

    backpropagate(in, labels, acts, weights, grad, config);

    TEST(config->nlayers + 2, token_unemb);
    TEST(config->nlayers + 1, last_rms.gamma);

    for (int i = config->nlayers; i > 0; i--) {
        TEST(i, layers[i - 1].attn.WQ);
        TEST(i, layers[i - 1].attn.WK);
        TEST(i, layers[i - 1].attn.WV);
        TEST(i, layers[i - 1].attn.WO);
        TEST(i, layers[i - 1].ff.b1);
        TEST(i, layers[i - 1].ff.b2);
        TEST(i, layers[i - 1].ff.W1);
        TEST(i, layers[i - 1].ff.W2);
        TEST(i, layers[i - 1].rms1.gamma);
        TEST(i, layers[i - 1].rms2.gamma);
    }

    TEST(0, pos_emb);
    TEST(0, token_emb);

    #undef TEST
}
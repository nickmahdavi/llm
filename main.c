#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <math.h>
#include "init.h"
#include "struct.h"
#include "layers.h"
#include "grad.h"
#include "ops.h"

#define MEM_POW_2 33

#define VOCAB 10
#define DMODEL 64
#define DFF (4 * DMODEL)
#define EPS 1e-05
#define ETA 0.05
#define BATCH 25
#define SEQ 5
#define HEADS 4
#define LAYERS 8
#define DK (DMODEL / HEADS)

#define SAMPLES 100
#define EPOCHS 50

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
    Pool *pool = malloc(sizeof(Pool));
    pool->size = (size_t)1 << MEM_POW_2;
    pool->off = 0;
    pool->data = malloc((size_t)1 << MEM_POW_2);

    Config *config = malloc(sizeof(Config));
    config->pool = pool;
    config->nvocab = VOCAB;
    config->dmodel = DMODEL;
    config->dff = DFF;
    config->eps = EPS;
    config->learning_rate = ETA;
    config->max_batch = BATCH;
    config->max_seq = SEQ;
    config->nheads = HEADS;
    config->nlayers = LAYERS;

    unsigned long long x_size = (BATCH * SEQ * DMODEL);

    unsigned long long aa_size = 2 * x_size + (3 * DK + SEQ) + x_size;
    unsigned long long aw_size = (HEADS * DMODEL * DK) * 3 + (DMODEL * DMODEL);

    unsigned long long fa_size = (BATCH * SEQ) * (9 * DMODEL);
    unsigned long long fw_size = 2 * DMODEL * DFF + DFF + DMODEL;

    unsigned long long la_size = (BATCH * SEQ) * (2 + DMODEL);
    unsigned long long lw_size = 2 * DMODEL;

    unsigned long long deca_size = aa_size + fa_size + 2 * la_size + 2 * x_size;
    unsigned long long decw_size = aw_size + fw_size + 2 * lw_size;

    unsigned long long w_size = (sizeof(float)) * ((VOCAB * DMODEL) + (DMODEL * DMODEL) + LAYERS * decw_size + lw_size);
    unsigned long long a_size = (sizeof(float)) * (2 * x_size + LAYERS * deca_size + la_size + 2 * (BATCH * SEQ * VOCAB));

    unsigned long long wc = w_size, ac = a_size;
    int wlog = 0, alog = 0;

    while ((wc >>= 1)) wlog++;
    while ((ac >>= 1)) alog++;

    if ( wlog >= 50 || alog >= 50) {
        printf("Too big (weights 2^%d GB, acts 2^%d GB)\n", wlog / 10, alog / 10);
        return 1;
    } else {
        long double tot = (w_size * 2 + a_size);
        if (wlog > 28 || alog > 28) {
            printf("Allocating %Lf GB. Continue? ", tot / (1 << 30));
        } else if (wlog > 18 || alog > 18){
            printf("Allocating %Lf MB. Continue? ", tot / (1 << 20));
        } else {
            printf("Allocating %Lf KB. Continue? ", tot / (1 << 10));
        }
        fflush(stdout);
        while (getchar() != '\n');
    }

    printf("Ok\n\n");

    printf("allocating activations cache\n");
    Activations *acts = init_acts(config);
    printf("allocating weights\n");
    Weights *weights = init_weights(config);
    printf("allocating grads\n");
    Weights *grad = init_weights(config);

    Tensor *X[SAMPLES * 2];
    Tensor *y[SAMPLES * 2];

    printf("datagen\n");
    for (int i = 0; i < SAMPLES * 2; i++) {
        X[i] = palloct(pool, config->max_batch, 1, config->max_seq, config->nvocab);
        y[i] = palloct(pool, config->max_batch, 1, config->max_seq, config->nvocab);
        for (int b = 0; b < config->max_batch; b++) {
            int tok = rand() % config->nvocab;
            for (int j = 0; j < config->max_seq; j++) {
                int next = (tok + 1) % config->nvocab;
                int base = b * config->max_seq * config->nvocab + j * config->nvocab;
                X[i]->data[base + tok] = 1.0f;
                y[i]->data[base + next] = 1.0f;
                tok = next;
            }
        }
    }

    printf("\ninitial loss: ");
    float loss = 0;
    for (int i = SAMPLES; i < SAMPLES * 2; i++) {
        forward(X[i], weights, acts, config);
        loss += crossentropy(&acts->probs, y[i]);
    }
    loss /= SAMPLES;
    printf("%f \n", loss);

    printf("\ntraining\n\n");
    for (int e = 1; e <= EPOCHS; e++) {
        printf("epoch %d: ", e);
        double fwd = 0, back = 0;
        for (int i = 0; i < SAMPLES; i++) {
            double t1 = get_time();
            forward(X[i], weights, acts, config);
            double t2 = get_time();
            backpropagate(X[i], y[i], acts, weights, grad, config);
            double t3 = get_time();

            fwd += t2 - t1;
            back += t3 - t2;
        }

        printf("avg forward pass %.3f ms, backwards pass %.3f ms // ", fwd, back);

        loss = 0;
        for (int i = SAMPLES; i < SAMPLES * 2; i++) {
            forward(X[i], weights, acts, config);
            loss += crossentropy(&acts->probs, y[i]);
        }
        loss /= SAMPLES;
        printf("loss %f \n", loss);
    }

    //forward(X[0], weights, acts, config);

    //tprint(&acts->probs);
    //tprint(y[0]);


    free(pool->data);
    free(pool);
    free(config);
}
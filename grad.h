#ifndef GRAD_H
#define GRAD_H

#include "struct.h"

void backpropagate(Tensor *in, Tensor *labels, Activations *acts, Weights *weights, Weights *grad, Config *config);
void adamw(int t, int batch, Weights *weights, Weights *grad, Weights *m, Weights *v, Config *config);
void zero_grad(Weights *grad, Config *config);

#endif